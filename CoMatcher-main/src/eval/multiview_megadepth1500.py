
import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets import get_dataset
from src.models.cache_loader import CacheLoader
from src.settings import DATA_PATH, EVAL_PATH
from src.utils.export_predictions import export_predictions, export_multiview_predictions_mega
from src.visualization.viz2d import plot_cumulative
from src.eval.eval_pipeline import EvalPipeline
from src.eval.io import get_eval_parser, load_model, parse_eval_args
from src.eval.utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust

logger = logging.getLogger(__name__)
dataset_path = Path(__file__).parent.parent / "datasets"
class MultiviewMegaDepth1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "model_data": {
                "name": "image_quads",
                "extra_data": "relative_pose",
                "root": "MegaDepth/Undistorted_SfM",
                "quads": str(dataset_path / "multiview/megadepth_scene_lists/quads_calibrated.txt"),
                "preprocessing": {
                    "side": "long",
                    "resize": 1600,
                    "square_pad": True,
                },
                "num_workers": 16,
            },
            "eval_data": {
                "name": "image_pairs",
                "pairs": str(dataset_path / "megadepth_scene_lists/pairs_calibrated_valid_pairs_hard.txt"),
                "root": "MegaDepth/Undistorted_SfM",
                "extra_data": "relative_pose",
                "preprocessing": {
                    "side": "long",
                    "square_pad": True,
                    "resize": 1600
                },
                "num_workers": 16,
            }
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": 0.5,
        },

    }

    export_keys = [
        "target_keypoints",
        "target_keypoint_scores",
        "source_keypoints",
        "source_keypoint_scores",
        "t2s_matches",
        "t2s_matches_scores",
        "s2t_matches",
        "s2t_matches_scores"
    ]

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        if data_conf and data_conf.name == "image_quads":
            dataset = get_dataset("image_quads")(data_conf)
        else:
            data_conf = self.default_conf["data"]["eval_data"]
            dataset = get_dataset("image_pairs")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_multiview_predictions_mega(
                self.get_dataloader(self.conf.data.model_data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file, save_fig=False):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results



if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MultiviewMegaDepth1500Pipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/multiview/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = MultiviewMegaDepth1500Pipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval, save_fig=args.save_fig
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()

