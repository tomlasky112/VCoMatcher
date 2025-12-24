import argparse
import os
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from src.datasets import get_dataset
from src.models import get_model
from src.utils.tensor import batch_to_device

resize = 1024
n_kpts = 2048
configs = {
    "sp": {
        "name": f"r{resize}_SP-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "third_party.superpoint",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "sp_open": {
        "name": f"r{resize}_SP-open-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "extractors.superpoint_open",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "cv2-sift": {
        "name": f"r{resize}_opencv-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": 4096,
            "backend": "opencv",
        },
    },
    "pycolmap-sift": {
        "name": f"r{resize}_pycolmap-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap",
        },
    },
    "pycolmap-sift-gpu": {
        "name": f"r{resize}_pycolmap_SIFTGPU-nms3-fixed-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap_cuda",
            "nms_radius": 3,
        },
    },
    "keynet-affnet-hardnet": {
        "name": f"r{resize}_KeyNetAffNetHardNet-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.keynet_affnet_hardnet",
            "max_num_keypoints": n_kpts,
        },
    },
    "disk": {
        "name": f"r{resize}_DISK-k{n_kpts}-nms5",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.disk_kornia",
            "max_num_keypoints": n_kpts,
        },
    },
    "aliked": {
        "name": f"r{resize}_ALIKED-k{n_kpts}-n16",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.aliked",
            "max_num_keypoints": n_kpts,
        },
    },
}


def export_features(loader, model, keys, export_path):
    Path(export_path).mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)

        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

        for key in pred.keys():
            k = key
            if key == "keypoint_scores":
                k = "scores"
            elif key == "oris":
                k = "angles"
            export_file = export_path / f"{k}.h5"
            with h5py.File(export_file, mode='a') as f:
                name = os.path.splitext(os.path.basename(data["name"][0]))[0]
                f[name] = pred[key]

class ExtractPipeline():
    config = {
        "data": {
            "name": "imc",
            "grayscale": True,
            "num_workers": 16,
            "preprocessing": {
                "resize": resize,
            },
            "batch_size": 1,
            "root_folder": "",
        },
        "model": {},
        "process": {},
    }

    def __init__(self, method, num_workers):
        self.method = method
        self.config["data"]["num_workers"] = num_workers
        self.config["model"] = configs[method]["conf"]
        self.config["data"]["grayscale"] = configs[method]["gray"]
    def run(self, root_path, export_path):
        keys = configs[self.method]["keys"]
        self.config["data"]["root_folder"] = root_path
        dataset = get_dataset(self.config["data"]["name"])(self.config["data"])
        loader = dataset.get_data_loader("test")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(self.config["model"]["name"])(self.config["model"]).eval().to(device)

        export_features(loader, model, keys, export_path)


if __name__ == "__main__":

    data_root = Path("/home/zjt/code/matcher/imc-2021-data/phototourism")
    output_root = Path("../../data/imc")

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="sp")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    method_name = configs[args.method]["name"]
    export_root = Path(output_root / method_name / 'phototourism')
    export_root.mkdir(exist_ok=True, parents=True)
    scenes = [p.name for p in data_root.iterdir() if p.is_dir()]

    pipeline = ExtractPipeline(args.method, args.num_workers)
    for scene in scenes:
        print(f"extract features in {scene}.")
        scene_path = data_root / scene
        export_path = export_root / scene
        pipeline.run(scene_path, export_path)





