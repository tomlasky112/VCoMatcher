import torch

from ...geometry.gt_generation import (
    multiview_gt_matches_from_pose_depth,
)
from ..base_model import BaseModel


class MultiviewDepthMatcher(BaseModel):
    default_conf = {
        "th_positive": 3.0,
        "th_negative": 5.0,
        "th_epi": None,  # add some more epi outliers
        "th_consistency": None,  # check for projection consistency in px
    }

    required_data_keys = ["target_view", "source_views", "Tt2s", "Ts2t", "target_keypoints", "source_keypoints"]

    def _init(self, conf):
        pass

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _forward(self, data):
        result = {}
        if "depth_keypoints0" in data:
            keys = [
                "depth_keypoints0",
                "valid_depth_keypoints0",
                "depth_keypoints1",
                "valid_depth_keypoints1",
            ]
            kw = {k: data[k] for k in keys}
        else:
            kw = {}
        b, N, _, _ = data["source_keypoints"].size()
        result_per_pairs = []
        for i in range(N):
            keypoints0 = data["target_keypoints"]
            keypoints1 = data["source_keypoints"][:, i]
            camera0 = data["target_view"]["camera"]
            camera1 = data["source_views"]["camera"][:, i]
            T_0to1 = data["Tt2s"][:, i]
            T_1to0 = data["Ts2t"][:, i]
            depth0 = data["target_view"]["depth"]
            depth1 = data["source_views"]["depth"][:, i]

            result_per_pairs.append(multiview_gt_matches_from_pose_depth(
                keypoints0,
                keypoints1,
                camera0,
                camera1,
                T_0to1,
                T_1to0,
                depth0,
                depth1,
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
                epi_th=self.conf.th_epi,
                cc_th=self.conf.th_consistency,
                **kw
            ))

        for key in result_per_pairs[0].keys():
            result[key] = torch.stack([r[key] for r in result_per_pairs], dim=1)
        return result

    def loss(self, pred, data):
        raise NotImplementedError
