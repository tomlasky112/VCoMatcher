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
from src.settings import EVAL_PATH
from src.utils.export_predictions import export_predictions, export_multiview_predictions
from src.utils.tensor import map_tensor
from src.utils.tools import AUCMetric
from src.visualization.viz2d import plot_cumulative
from src.eval.eval_pipeline import EvalPipeline
from src.eval.io import get_eval_parser, load_model, parse_eval_args
from src.eval.utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
    eval_poses,
)
from src.visualization.viz2d import plot_image_grid, plot_keypoints, plot_matches

class HPatchesMultiviewPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "model_data": {
                "name": "multiview_hpatches",
                "batch_size": 1,
                "num_workers": 1,
                "preprocessing": {
                    "resize": 480,
                    "side": "short",
                },
            },
            "eval_data": {
                "name": "hpatches",
                "batch_size": 1,
                "num_workers": 1,
                "preprocessing": {
                    "resize": 480,
                    "side": "short",
                },
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
        if data_conf and data_conf.name == "multiview_hpatches":
            dataset = get_dataset("multiview_hpatches")(data_conf)
        else:
            data_conf = self.default_conf["data"]["eval_data"]
            dataset = get_dataset("hpatches")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_multiview_predictions(
                self.get_dataloader(self.conf.data.model_data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def plot_matches(self, data, pred):
        images, kpts, matches, mcolors = [], [], [], []

        view0, view1 = data["view0"], data["view1"]
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]  # 取batch
        m0 = pred["matches0"]

        valid = (m0 > -1)
        kpm0, kpm1 = kp0[valid], kp1[m0[valid]]
        images.append(
            [view0["image"].permute(1, 2, 0).cpu().numpy(), view1["image"].permute(1, 2, 0).cpu().numpy()]
        )
        kpts.append([kp0.cpu().numpy(), kp1.cpu().numpy()])
        matches.append((kpm0.cpu().numpy(), kpm1.cpu().numpy()))

        fig, axes = plot_image_grid(images, return_fig=True, set_lim=True, dpi=150, figs=4.0,pad=0.25)
        plot_keypoints(kpts[0], axes=axes[0], colors="royalblue")
        plot_matches(*(matches[0]), axes=axes[0], a=0.5, lw=1.0, ps=0.0)

        return fig

    def run_eval(self, loader, pred_file, save_fig=False):
        assert pred_file.exists()
        results = defaultdict(list)

        conf = self.conf.eval

        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )   # 0.5
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        fig_dict = {}
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here
            if "keypoints0" in pred:
                results_i = eval_matches_homography(data, pred)
                results_i = {**results_i, **eval_homography_dlt(data, pred)}
            else:
                results_i = {}
            for th in test_thresholds:
                pose_results_i = eval_homography_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

            # save matching result
            if save_fig:
                fig_dict[data["name"][0]] = self.plot_matches(data, pred)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.median(arr), 3)   # 取中位数

        auc_ths = [1, 3, 5]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
        )
        if "H_error_dlt" in results.keys():
            dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
            for i, ath in enumerate(auc_ths):
                summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]

        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "homography_recall": plot_cumulative(
                {
                    "DLT": results["H_error_dlt"],
                    self.conf.eval.estimator: results["H_error_ransac"],
                },
                [0, 10],
                unit="px",
                title="Homography ",
            ),
            **fig_dict
        }

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HPatchesMultiviewPipeline.default_conf)

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

    pipeline = HPatchesMultiviewPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval, save_fig=args.save_fig
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
