import itertools
import os

import torch
from omegaconf import OmegaConf
from . import get_model
from .base_model import BaseModel
from .. import logger
from ..utils.tensor import batch_to_device
from ..visualization.viz2d import cm_RdGn, plot_image_grid, plot_keypoints, plot_matches

to_ctr = OmegaConf.to_container


def check_S2S_matches_correct(data_, pred_):
    # we check [0->1]
    check_i, check_j = 0, 1

    images, kpts, matches, mcolors = [], [], [], []

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    kp0, kp1 = pred["source_keypoints"][2, check_i, :, :].numpy(), pred["source_keypoints"][2, check_j, :, :].numpy()
    m0 = pred["S2S_matches"][2, check_i, check_j, :]

    valid = m0 > -1
    kpm0 = kp0[valid]
    kpm1 = kp1[m0[valid]]
    images.append(
        [data["source_views"]["image"][2, check_i, :, :, :].permute(1, 2, 0), data["source_views"]["image"][2, check_j, :, :, :].permute(1, 2, 0)]
    )
    kpts.append([kp0, kp1])
    matches.append((kpm0, kpm1))

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    [plot_keypoints(kpts[0], axes=axes[0], colors="royalblue")]
    [plot_matches(*matches[0], axes=axes[0], a=0.5, lw=1.0, ps=0.0)]

    # save the figure
    # print(os.getcwd())
    fig.savefig("S2S_matches.png")



class PriorPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "prior_matcher": {"name": None},
        "multiview_matcher": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False
    }
    required_data_keys = ["target_view", "source_views"]
    strict_conf = False
    components = [
        "extractor",
        "multiview_matcher",
        "ground_truth",
        "prior_matcher"
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.prior_matcher.name:
            self.prior_matcher = get_model(conf.prior_matcher.name)(to_ctr(conf.prior_matcher))

        if conf.multiview_matcher.name:
            self.multiview_matcher = get_model(conf.multiview_matcher.name)(to_ctr(conf.multiview_matcher))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_features(self, data, target=True):
        if target:
            data = data["target_view"]
            pred = data.get("cache", {})
            skip_extract = len(pred) > 0 and self.conf.allow_no_extract
            if self.conf.extractor.name and not skip_extract:
                return {**pred, **self.extractor(data)}
            elif skip_extract:
                return pred
            else:
                logger.warning("No extractor found for target view")
                return pred
        else:
            data = data["source_views"]
            pred = data.get("cache", {})
            skip_extract = len(pred) > 0 and self.conf.allow_no_extract
            if self.conf.extractor.name and not skip_extract:
                b, N, *_ = data["image"].size()
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].view(b * N, *data[key].size()[2:])
                pred = {**pred, **self.extractor(data)}
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].view(b, N, *data[key].size()[1:])
                for key in pred.keys():
                    pred[key] = pred[key].view(b, N, *pred[key].size()[1:])
                return pred
            elif skip_extract:
                return pred
            else:
                logger.warning("No extractor found for target view")
                return pred


    def trans2_s2s_matches(self, prior_matches, pairs_ind, N):
        S2S_matches = []
        for i in range(N):
            per_view_matches = []
            # get all pairs that contain i-th view in pairs_ind list
            # sort by the element that is not i-th view
            pairs = sorted([pair for pair in pairs_ind if i in pair], key=lambda x: x[0] if x[0] != i else x[1])
            for pair in pairs:
                # if i-th view is the first element in the pair we get the matches0 else matches1
                id = pairs_ind.index(pair)
                per_view_matches.append(prior_matches["matches0"][:, id, :] if pair[0] == i else prior_matches["matches1"][:, id, :])
            # insert a -1 tensor in i position of the list
            per_view_matches.insert(i, torch.full_like(per_view_matches[0], -1))
            S2S_matches.append(torch.stack(per_view_matches, dim=1)) # b, N, n
        results = torch.stack(S2S_matches, dim=1) # b, N, N, n
        return results

    def get_prior_matches(self, data):
        # 1. sources views to pairs set
        _, N, _, _ = data["source_keypoints"].size()
        N_ind = list(range(N))
        pairs_ind = list(itertools.combinations(N_ind, 2))

        matches_results = []

        pair_data = []
        for pair in pairs_ind:
            # 2. trans 2 target format
            pair_data.append({
                "keypoints0": data["source_keypoints"][:, pair[0], :, :],
                "keypoints1": data["source_keypoints"][:, pair[1], :, :],
                "descriptors0": data["source_descriptors"][:, pair[0], :, :],
                "descriptors1": data["source_descriptors"][:, pair[1], :, :],
                "image_size0": data["source_views"]["image_size"][:, pair[0], :],
                "image_size1": data["source_views"]["image_size"][:, pair[1], :],
            })
        # stack the pair_data
        pair_data = {k: torch.stack([pair[k] for pair in pair_data], dim=1) for k in pair_data[0].keys()}
        b, N_pairs, *_ = pair_data["keypoints0"].size()
        for key in pair_data.keys():
            if isinstance(pair_data[key], torch.Tensor):
                pair_data[key] = pair_data[key].view(b * N_pairs, *pair_data[key].size()[2:])

        view0 = {"image_size": pair_data["image_size0"]}
        view1 = {"image_size": pair_data["image_size1"]}
        pair_data = {**pair_data, "view0": view0, "view1": view1}


        if self.conf.prior_matcher.name:
            matches_results = self.prior_matcher(pair_data)
        for key in matches_results.keys():
            if isinstance(matches_results[key], torch.Tensor):
                matches_results[key] = matches_results[key].view(b, N_pairs, *matches_results[key].size()[1:])

        # 4. organize results
        return self.trans2_s2s_matches(matches_results, pairs_ind, N)


    def _forward(self, data):
        target_pred = self.extract_features(data, target=True)
        source_pred = self.extract_features(data, target=False)
        pred = {
            **{"target_" + k: v for k, v in target_pred.items()},
            **{"source_" + k: v for k, v in source_pred.items()},
        }

        pred = {**pred, "S2S_matches": self.get_prior_matches({**data, **pred})}

        # check_S2S_matches_correct(data, pred)

        if self.conf.multiview_matcher.name:
            pred = {**pred, **self.multiview_matcher({**data, **pred})}

        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics

