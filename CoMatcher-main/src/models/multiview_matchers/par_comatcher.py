import warnings
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models import BaseModel
from src.models.utils.metrics import multiview_matcher_metrics

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


# ----------KP_Norm, Positional Encoding and Rotary Embedding----------
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


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        # 2, 64, 64(head_dim)
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim  # 64
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)  # 2->32
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        # head dim = 2
        projected = self.Wr(x)  # 2->32
        cosines, sines = torch.cos(projected), torch.sin(projected)  # b,n,32
        emb = torch.stack([cosines, sines], 0).unsqueeze(2)  # 2,b,1,n,32
        return emb.repeat_interleave(2, dim=-1)  # 2,b,1,n,64


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


def get_q_kpts(s_kpts, s2s_matches):
    """
    Params:
        s_kpts: (b, N, n, 2)
        s2s_matches: (b, N, N, n), -1 means no match
    Return:
        return: q_kpts (b, N, N, n, 2), dim 1 means key; dim 2 means query
    """
    b, N, n, _ = s_kpts.size()
    q_kpts = s_kpts.unsqueeze(1).expand(b, N, N, n, 2)
    s2s_matches = s2s_matches.unsqueeze(-1).expand(b, N, N, n, 2)
    # -1 means no match, so we need to filter them
    mask = s2s_matches != -1
    s2s_matches = torch.where(mask, s2s_matches, s2s_matches.new_tensor(0))
    q_kpts = torch.gather(q_kpts, 3, s2s_matches)
    q_kpts = torch.where(mask, q_kpts, q_kpts.new_tensor(0))
    return q_kpts.transpose(1,2)


def get_AP_mask(confidence: torch.Tensor, s2s_matches: torch.Tensor, th: float = 0.75) -> torch.Tensor:
    """
    Generate AP mask based on confidence and S2S matches.
    Args:
        confidence: (b, N, n) confidence scores
        s2s_matches: (b, N, N, n) cross-view matches
        th: confidence threshold
    Returns:
        AP_mask: (b, N*n) indices for attention propagation
    """
    b, N, n = confidence.shape
    device = confidence.device

    # Step 1: Filter low-confidence points
    conf_mask = (confidence > th).float()  # (b, N, n)

    # Step 2: Initialize index table (image_idx * n + point_idx)
    index_map = torch.arange(N * n, device=device).view(N, n)  # (N, n)

    # Step 3: Propagate indices through matches
    AP_mask = index_map.view(1, N, n).expand(b, -1, -1).clone()  # (b, N, n)

    for i in range(N):
        for j in range(N):
            matches_ij = s2s_matches[:, i, j]  # (b, n)
            valid = (matches_ij != -1) & (conf_mask[:, j] > 0)  # (b, n)

            # Update AP_mask where matches are valid
            src_indices = matches_ij[valid] + j * n  # (b, n) -> flat indices
            tgt_indices = torch.where(valid,
                                      torch.arange(n, device=device).expand(b, -1),
                                      AP_mask[:, i])
            AP_mask[:, i] = torch.where(valid, src_indices, tgt_indices)

    return AP_mask.view(b, N * n)  # Flatten to (b, N*n)


# ----------attention layers----------
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

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        if self.flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            args = [x.half().contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
            return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
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
        if encoding.dim() == 6:
            encoding = encoding.transpose(2, 3)

        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(-4, -3)  # b, h, n, d/h, 3
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)

        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(-3, -2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, d, num_heads, flash=False):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        assert self.d % num_heads == 0
        self.head_dim = self.d // num_heads

        self.q = nn.Linear(d, d, bias=True)
        self.kv = nn.Linear(d, 2 * d, bias=True)

        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.LayerNorm(2 * d, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, desc0, desc1, AP_mask=None):
        b, N, m, d = desc0.size()
        q = self.q(desc0)  # b, N, m, h, d
        kv = self.kv(desc1)
        q = q.unflatten(-1, (self.num_heads, -1, 1)).transpose(2, 3)
        kv = kv.unflatten(-1, (self.num_heads, -1, 2)).transpose(2, 3)

        q, k, v = q[..., 0], kv[..., 0], kv[..., 1]

        # Apply AP if enabled
        if self.use_ap and AP_mask is not None:
            k = self._apply_ap(k, AP_mask)  # (b, h, N*n, d/h)
            v = self._apply_ap(v, AP_mask)

        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(2, 3).flatten(start_dim=-2))
        return desc0 + self.ffn(torch.cat([desc0, message], -1))

    def _apply_ap(self, x: torch.Tensor, AP_mask: torch.Tensor) -> torch.Tensor:
        """Propagate features via AP mask"""
        b, h, _, c = x.shape
        x = x.transpose(2, 3).reshape(b, h * c, -1)  # (b, h*c, N*n)
        x = torch.gather(x, 2, AP_mask.unsqueeze(1).expand(-1, h * c, -1))
        return x.view(b, h, c, -1).transpose(2, 3)


class T2TCrossBlock(nn.Module):
    def __init__(self, d, num_heads, flash=False):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        assert self.d % num_heads == 0
        self.head_dim = self.d // num_heads

        self.qkv = nn.Linear(d, 3 * d, bias=True)

        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.LayerNorm(2 * d, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, t_desc):
        b, N, m, d = t_desc.size()
        t_desc = t_desc.transpose(1, 2)
        qkv = self.qkv(t_desc)  # b, m, N, 3d
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(3, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        context = self.inner_attn(q, k, v)  # b, m, h, N, d/h
        message = self.out_proj(context.transpose(3, 2).flatten(start_dim=-2))
        return (t_desc + self.ffn(torch.cat([t_desc, message], -1))).transpose(1, 2)


class S2SCrossBlock(nn.Module):
    def __init__(self, d, num_heads, flash=False):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        assert self.d % num_heads == 0
        self.head_dim = self.d // num_heads

        self.qkv = nn.Linear(d, 3 * d, bias=True)

        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.LayerNorm(2 * d, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, s_desc, q_encoding, k_encoding):
        b, N, n, d = s_desc.size()

        qkv = self.qkv(s_desc)  # b, N, n, 3d
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(-4, -3).transpose(-4, -5)  # b, h, N, n, c, 3
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # b, h, N, n, c

        q = q.unsqueeze(2).expand(b, self.num_heads, N, N, n, d // self.num_heads)  # b, h, N, N, n, c
        q = apply_cached_rotary_emb(q_encoding, q)
        q = q.contiguous().view(b, self.num_heads, N, N * n, d // self.num_heads)  # b, h, N, N*n, c

        k = apply_cached_rotary_emb(k_encoding, k)  # b, h, N, n, c

        context = self.inner_attn(q, k, v)  # b, h, N, N*n, c
        context = context.view(b, self.num_heads, N, N, n, d // self.num_heads)  # b, h, N, N, n, c

        # delete the same dim
        mask = torch.ones(N, N, dtype=torch.bool, device=s_desc.device)
        mask = mask.fill_diagonal_(0)
        context = context[:, :, mask].view(b, self.num_heads, N - 1, N, n, d // self.num_heads)
        context = context.mean(2)  # b, h, N, n, c
        context = context.transpose(1, 2).transpose(2, 3).flatten(start_dim=-2)  # b, N, n, c
        message = self.out_proj(context)
        return s_desc + self.ffn(torch.cat([s_desc, message], -1))


# --------confidence estimator and layer match assignment--------
class ConfidenceEstimator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, desc):
        return self.mlp(desc.detach()).squeeze(-1)


# ------------transformer layers------------

class TransformerLayerN2N(nn.Module):
    def __init__(self, dim, head, flash, is_last=False):
        super().__init__()
        self.self_attn = SelfBlock(dim, head, flash)
        self.s2s_cross_attn = S2SCrossBlock(dim, head, flash)
        self.t2t_cross_attn = T2TCrossBlock(dim, head, flash)
        self.cross_attn = CrossBlock(dim, head, flash)
        # confidence
        self.is_last = is_last
        if not is_last:
            self.confidence_estimator = ConfidenceEstimator(dim)

    def forward(self, t_desc, s_desc, t_encoding, s_encoding, q_encoding, k_encoding, AP_mask=None):
        t_desc = self.self_attn(t_desc, t_encoding)
        s_desc = self.self_attn(s_desc, s_encoding)
        t_desc = self.t2t_cross_attn(t_desc)
        s_desc = self.s2s_cross_attn(s_desc, q_encoding, k_encoding)
        # Pass AP_mask to cross-attention
        _t_desc = self.cross_attn(t_desc, s_desc, AP_mask)
        _s_desc = self.cross_attn(s_desc, t_desc, AP_mask)

        if not self.is_last:
            confidence = self.confidence_estimator(s_desc)
            return _t_desc, _s_desc, confidence
        else:
            return _t_desc, _s_desc


# ----------final matching head ----------
def sigmoid_log_double_softmax(sim, zt, zs):
    b, N, m, n = sim.shape
    certainties = F.logsigmoid(zt) + F.logsigmoid(zs).transpose(2, 3)
    scores0 = F.log_softmax(sim, 3)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 3).transpose(-1, -2)
    scores = sim.new_full((b, N, m + 1, n + 1), 0)
    scores[:, :, :m, :n] = scores0 + scores1 + certainties
    scores[:, :, :-1, -1] = F.logsigmoid(-zt.squeeze(-1))
    scores[:, :, -1, :-1] = F.logsigmoid(-zs.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, t_desc, s_desc):
        m_tdesc, m_sdesc = self.final_proj(t_desc), self.final_proj(s_desc)
        b, N, m, d = m_tdesc.shape
        m_tdesc, m_sdesc = m_tdesc / d ** 0.25, m_sdesc / d ** 0.25
        sim = torch.einsum("bNmd,bNnd->bNmn", m_tdesc, m_sdesc)
        zt = self.matchability(t_desc)
        zs = self.matchability(s_desc)
        scores = sigmoid_log_double_softmax(sim, zt, zs)
        return scores, sim


def layer_match_assignment(t_desc, s_desc, th=0.32):
    # boardcast t_desc to s_desc
    b, N, n, d = s_desc.size()
    if t_desc.dim() == 3:  # b,m,d
        t_desc = t_desc.unsqueeze(1).expand(b, N, -1, d)
    # get sim scores
    sim = torch.einsum("bNmd,bNnd->bNmn", t_desc, s_desc)
    sim = sim / d ** 0.5
    scores_t2s = F.log_softmax(sim, -1)
    scores_s2t = F.log_softmax(sim.transpose(-1, -2).contiguous(), -1).transpose(-1, -2)
    scores = scores_t2s + scores_s2t  # b,N,m,n

    # 0: t2s, 1: s2t
    max0, max1 = scores.max(3), scores.max(2)
    m0, m1 = max0.indices, max1.indices  # b,N,m/n

    indices0 = torch.arange(m0.shape[2], device=m0.device)[None]  # 1,m
    indices1 = torch.arange(m1.shape[2], device=m1.device)[None]  # 1,n
    # mutual matches
    mutual0 = indices0 == m1.gather(2, m0)  # m1(m0[i]) == i
    mutual1 = indices1 == m0.gather(2, m1)

    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    # mscores1 = torch.where(mutual1, mscores0.gather(2, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(2, m1)
    # m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m1


def filter_matches(scores: torch.Tensor, th: float):
    # 0: t2s, 1: s2t
    max0, max1 = scores[:, :, :-1, :-1].max(3), scores[:, :, :-1, :-1].max(2)
    m0, m1 = max0.indices, max1.indices

    indices0 = torch.arange(m0.shape[2], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[2], device=m1.device)[None]
    # mutual matches
    mutual0 = indices0 == m1.gather(2, m0)
    mutual1 = indices1 == m0.gather(2, m1)

    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(2, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(2, m1)
    m0 = torch.where(valid0, m0, m0.new_tensor(-1))
    m1 = torch.where(valid1, m1, m1.new_tensor(-1))
    return m0, m1, mscores0, mscores1


# ----------------Main Model------------------------
class ParCoMatcherPrior(BaseModel):
    default_conf = {
        "descriptor_dim": 256,
        # transformer
        "num_heads": 4,
        "n_n2n_layers": 9,
        "flash": False,
        "checkpointed": False,

        # compute result
        "filter_threshold": 0.1,
        "loss": {
            "nll_balancing": 0.5
        },
        # confidence
        "loss_lamda": 0.5,
        "use_ap": True,
        "ap_threshold": 0.7,
    }
    required_data_keys = ["target_keypoints", "target_descriptors", "source_keypoints", "source_descriptors",
                          "S2S_matches"]

    def _init(self, conf):
        # position encoding
        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2, head_dim, head_dim
        )
        self.posenc_s2s = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        # transformer
        h, l_n2n, d = conf.num_heads, conf.n_n2n_layers, conf.descriptor_dim
        self.transformer_n2n = nn.ModuleList(
            [TransformerLayerN2N(d, h, conf.flash, False) for _ in range(l_n2n - 1)] +
            [TransformerLayerN2N(d, h, conf.flash, True)])

        self.log_assignment = MatchAssignment(d)

    def _forward(self, data):
        target_keypoints, source_keypoints = data["target_keypoints"], data["source_keypoints"]
        b, m, _ = target_keypoints.shape
        b, N, n, _ = source_keypoints.shape

        if "target_view" in data.keys() and "source_views" in data.keys():
            target_size = data["target_view"]["image_size"]
            source_size = data["source_views"]["image_size"]

        t_kpts = normalize_keypoints(target_keypoints, target_size).clone()
        s_kpts = normalize_keypoints(source_keypoints, source_size).clone()

        t_kpts = t_kpts.unsqueeze(1).expand(b, N, m, -1)
        t_encoding_n2n = self.posenc(t_kpts)
        s_encoding = self.posenc(s_kpts)

        t_desc = data["target_descriptors"].contiguous()
        s_desc = data["source_descriptors"].contiguous()

        assert t_desc.shape[-1] == self.conf.descriptor_dim
        assert s_desc.shape[-1] == self.conf.descriptor_dim

        q_kpts = get_q_kpts(s_kpts, data["S2S_matches"])
        q_encoding = self.posenc_s2s(q_kpts)
        k_encoding = self.posenc_s2s(s_kpts)

        layer_s2t_matches_results, confidences = [], []

        AP_mask = None
        if not self.training:
            confidence = torch.sigmoid(self.confidence_estimator(s_desc))
            AP_mask = get_AP_mask(confidence, data["S2S_matches"])

        # N2N transformer
        t_desc = t_desc.unsqueeze(1).expand(b, N, m, -1)
        for i in range(self.conf.n_n2n_layers - 1):
            if self.conf.checkpointed and self.training:
                t_desc, s_desc, confidence = checkpoint(
                    self.transformer_n2n[i], t_desc, s_desc, t_encoding_n2n, s_encoding, q_encoding, k_encoding
                )
            else:
                t_desc, s_desc, confidence = self.transformer_n2n[i](t_desc, s_desc, t_encoding_n2n, s_encoding, q_encoding, k_encoding, AP_mask)
            confidences.append(confidence)

            if self.training:
                layer_s2t_matches_results.append(layer_match_assignment(t_desc, s_desc))

        # final layer
        if self.conf.checkpointed and self.training:
            t_desc, s_desc = checkpoint(
                self.transformer_n2n[-1], t_desc, s_desc, t_encoding_n2n, s_encoding, q_encoding, k_encoding
            )
        else:
            t_desc, s_desc = self.transformer_n2n[-1](t_desc, s_desc, t_encoding_n2n, s_encoding, q_encoding, k_encoding)

        scores, _ = self.log_assignment(t_desc, s_desc)
        t2s_matches, s2t_matches, t2s_matches_scores, s2t_matches_scores = filter_matches(scores,
                                                                                          self.conf.filter_threshold)

        pred = {
            "t2s_matches": t2s_matches,
            "s2t_matches": s2t_matches,
            "t2s_matches_scores": t2s_matches_scores,
            "s2t_matches_scores": s2t_matches_scores,
            "log_assignment": scores,
        }

        if self.training:
            pred["layer_s2t_matches_results"] = layer_s2t_matches_results
            pred["confidences"] = confidences
        return pred

    def loss(self, pred, data):
        losses = {"total": 0}

        positive = data["gt_assignment"].float()  # b, N, m, n
        num_pos = torch.max(positive.sum((2, 3)), positive.new_tensor(1))
        neg0 = (data["gt_t2s_matches"] == -1).float()
        neg1 = (data["gt_s2t_matches"] == -1).float()
        num_neg = torch.max(neg0.sum(2) + neg1.sum(2), neg0.new_tensor(1))

        log_assignment = pred["log_assignment"]
        nll_pos = -(log_assignment[:, :, :-1, :-1] * positive).sum((2, 3))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :, :-1, -1] * neg0).sum(2)
        nll_neg1 = -(log_assignment[:, :, -1, :-1] * neg1).sum(2)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg

        # average the nll of all the source views
        nll_pos = nll_pos.mean(dim=1)
        nll_neg = nll_neg.mean(dim=1)

        nll = (
                self.conf.loss.nll_balancing * nll_pos
                + (1 - self.conf.loss.nll_balancing) * nll_neg
        )

        losses["assignment_nll"] = nll
        losses["total"] = nll

        losses["nll_pos"] = nll_pos
        losses["nll_neg"] = nll_neg

        # Some statistics
        losses["num_matchable"] = num_pos.mean(dim=1)
        losses["num_unmatchable"] = num_neg.mean(dim=1)

        # confidence loss
        if self.training:
            losses["confidence"] = 0.0
            final_s2t_matches = pred["s2t_matches"]  # b,N,n
            l = len(pred["layer_s2t_matches_results"])
            for i in range(l):
                s2t_matches = pred["layer_s2t_matches_results"][i]  # b,N,n
                gt = s2t_matches == final_s2t_matches  # b,N,n
                losses["confidence"] += F.binary_cross_entropy_with_logits(
                    pred["confidences"][i], gt.float(), reduction="none"
                ).mean(dim=(1, 2))
            losses["confidence"] = losses["confidence"] / l
            losses["total"] = losses["total"] + self.conf.loss_lamda * losses["confidence"]

        if not self.training:
            # add metrics
            metrics = multiview_matcher_metrics(pred, data)
        else:
            metrics = {}

        return losses, metrics
