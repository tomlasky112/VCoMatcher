from pathlib import Path

import numpy as np
import torch

from src.datasets import BaseDataset
from src.utils.image import ImagePreprocessor, load_image
from ..geometry.wrappers import Camera, Pose
from ..settings import DATA_PATH, MEGA_DATA_PATH

def names_to_pair(name0, name1, separator="/"):
    name0_l = name0.split("/")
    name0_l.pop(1)
    name1_l = name1.split("/")
    name1_l.pop(1)
    name0 = '/'.join(name0_l)
    name1 = '/'.join(name1_l)

    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def parse_camera(calib_elems) -> Camera:
    assert len(calib_elems) == 9
    K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
    return Camera.from_calibration_matrix(K)

def parse_relative_pose(pose_elems) -> Pose:
    # assert len(calib_list) == 9
    R, t = pose_elems[:9], pose_elems[9:12]
    R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
    t = np.array([float(x) for x in t]).astype(np.float32)
    return Pose.from_Rt(R, t)

class ImageQuads(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "quads": None,
        "root": None,
        "preprocessing": ImagePreprocessor.default_conf,
        "extra_data": None,
    }

    def _init(self, conf):
        quad_f = (
            Path(conf.quads) if Path(conf.quads).exists() else MEGA_DATA_PATH / conf.quads
        )
        with open(str(quad_f), "r") as f:
            self.items = [line.rstrip() for line in f]
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split):
        return self

    def _read_view(self, name):
        path = MEGA_DATA_PATH / self.conf.root / name
        img = load_image(path)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        assert self.conf.extra_data == "relative_pose"

        line = self.items[idx]
        quad_data = line.split(" ")
        name0, name1, name2, name3 = quad_data[:4]

        target_view = self._read_view(name0)
        target_view["camera"] = parse_camera(quad_data[4:13]).scale(
                target_view["scales"])

        source_views, Tt2s = [], []
        for i in range(3):
            source_view = self._read_view(locals()[f'name{i+1}'])
            source_view["camera"] = parse_camera(quad_data[13 + i * 9:13 + (i + 1) * 9]).scale(
                source_view["scales"])
            source_views.append(source_view)
            Tt2s.append(parse_relative_pose(quad_data[40 + i * 12:40 + (i + 1) * 12]))

        names = [names_to_pair(name0, name1), names_to_pair(name0, name2), names_to_pair(name0, name3)]

        data = {
            "target_view": target_view,
            "source_views": source_views,
            "Tt2s": Tt2s,
            "idx": idx,
            "names": '|'.join(names)
        }
        return data

    def __len__(self):
        return len(self.items)