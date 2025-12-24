import torch
from omegaconf import OmegaConf
from . import get_model
from .base_model import BaseModel
from .. import logger

to_ctr = OmegaConf.to_container


class MultiviewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "multiview_matcher": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False
    }
    required_data_keys = ["target_view", "source_views"]
    strict_conf = False
    components = [
        "extractor",
        "multiview_matcher",
        "ground_truth"
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

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

    def _forward(self, data):
        target_pred = self.extract_features(data, target=True)
        source_pred = self.extract_features(data, target=False)
        pred = {
            **{"target_" + k: v for k, v in target_pred.items()},
            **{"source_" + k: v for k, v in source_pred.items()},
        }

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

