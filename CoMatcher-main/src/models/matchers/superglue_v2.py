import warnings
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch import nn
from copy import deepcopy
import logging
from torch.utils.checkpoint import checkpoint

from src.models.base_model import BaseModel
from src.models.utils.metrics import matcher_metrics
import torch.nn.functional as F


FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class Attention(nn.Module):
    def __init__(self, flash=False):
        super().__init__()
        if flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.flash = flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(flash)

    def forward(self, q, k, v):
        if self.flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            args = [x.half().contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args).to(q.dtype)
            return v
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args)
            return v
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(self, d, num_heads, flash=False):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        assert self.d % num_heads == 0
        self.head_dim = self.d // num_heads

        self.Wqkv = nn.Linear(d, 3 * d, bias=True)

        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.LayerNorm(2 * d, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, x, encoding):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)

        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, d, num_heads, flash=False):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        assert self.d % num_heads == 0
        self.head_dim = self.d // num_heads

        self.Wq = nn.Linear(d, d, bias=True)
        self.Wkv = nn.Linear(d, 2 * d, bias=True)

        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.LayerNorm(2 * d, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, x1, x2):
        q = self.Wq(x1)
        kv = self.Wkv(x2)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        kv = kv.unflatten(-1, (self.num_heads, -1, 2)).transpose(1, 2)
        k, v = kv[..., 0], kv[..., 1]

        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x1 + self.ffn(torch.cat([x1, message], -1))



class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
    ):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        c_desc0 = self.cross_attn(desc0, desc1)
        c_desc1 = self.cross_attn(desc1, desc0)

        return c_desc0, c_desc1

def filter_matches(scores: torch.Tensor, th: float):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices     # b,n
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    # mutual matches
    mutual0 = indices0 == m1.gather(1, m0)  # b,n
    mutual1 = indices1 == m0.gather(1, m1)

    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero) # b,n
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)  # b,n
    valid1 = mutual1 & valid0.gather(1, m1)

    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class SuperGlueV2(BaseModel):
    default_conf = {
        "descriptor_dim": 256,
        # transformer
        "num_heads": 4,
        "n_layers": 9,
        "flash": False,
        "checkpointed": False,
        # sinkhorn
        "num_sinkhorn_iterations": 50,
        # compute result
        "filter_threshold": 0.1,
        "loss": {
            "nll_balancing": 0.5
        },
    }
    required_data_keys = [
        "keypoints0", "keypoints1", "descriptors0", "descriptors1"
    ]

    def _init(self, conf):
        # position encoding
        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2, head_dim, head_dim
        )

        # transformer
        h, l, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(l)]
        )

        # final_proj
        self.final_proj = nn.Linear(
            conf.descriptor_dim, conf.descriptor_dim, bias=True
        )

        # sinkhorn
        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

    def _forward(self, data):
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.descriptor_dim
        assert desc1.shape[-1] == self.conf.descriptor_dim

        # transformer
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)


        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        cost = scores / self.conf.descriptor_dim ** 0.5

        scores = log_optimal_transport(
            cost, self.bin_score, iters=self.conf.num_sinkhorn_iterations
        )
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)


        return {
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }

    def loss(self, pred, data):
        losses = {"total": 0}

        positive = data["gt_assignment"].float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))

        log_assignment = pred["log_assignment"]
        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = (
                self.conf.loss.nll_balancing * nll_pos
                + (1 - self.conf.loss.nll_balancing) * nll_neg
        )
        losses["assignment_nll"] = nll
        losses["total"] = nll

        losses["nll_pos"] = nll_pos
        losses["nll_neg"] = nll_neg

        # Some statistics
        losses["num_matchable"] = num_pos
        losses["num_unmatchable"] = num_neg
        losses["bin_score"] = self.bin_score[None]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics
