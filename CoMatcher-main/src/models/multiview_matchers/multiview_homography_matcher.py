import torch

from src.models import BaseModel
from ...geometry.gt_generation import (
    gt_matches_from_homography,
)


class MultiviewMatcher(BaseModel):
    default_conf = {
        "th_positive": 3.0,
        "th_negative": 3.0,
    }

    required_data_keys = ["Ht2s", "target_keypoints", "source_keypoints"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        result = {}
        b, N, *_ = data["Ht2s"].size()

        result_per_pairs = []
        for i in range(N):
            Ht2s = data["Ht2s"][:, i]
            source_keypoints = data["source_keypoints"][:, i]
            result_per_pairs.append(gt_matches_from_homography(
                data["target_keypoints"],
                source_keypoints,
                Ht2s,
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
                multiview=True
            ))
        for key in result_per_pairs[0].keys():
            result[key] = torch.stack([r[key] for r in result_per_pairs], dim=1)
        return result


    def loss(self, pred, data):
        raise NotImplementedError
