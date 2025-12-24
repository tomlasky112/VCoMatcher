"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.settings import DATA_PATH
from src.utils.image import ImagePreprocessor, load_image
from src.utils.tools import fork_rng
from src.visualization.viz2d import plot_image_grid
from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def read_homography(path):
    with open(path) as f:
        result = []
        for line in f.readlines():
            while "  " in line:  # Remove double spaces
                line = line.replace("  ", " ")
            line = line.replace(" \n", "").replace("\n", "")
            # Split and discard empty strings
            elements = list(filter(lambda s: s, line.split(" ")))
            if elements:
                result.append(elements)
        return np.array(result).astype(float)


class MultiviewHPatches(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "hpatches-sequences-release",
        "subset": None,
        "ignore_large_images": True,
        "grayscale": False,
    }

    # Large images that were ignored in previous papers
    ignored_scenes = (
        "i_contruction",
        "i_crownnight",
        "i_dc",
        "i_pencils",
        "i_whitebuilding",
        "v_artisans",
        "v_astronautis",
        "v_talent",
        "v_soldiers"
    )
    url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"

    def _init(self, conf):
        assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Downloading the HPatches dataset.")
            self.download()
        self.sequences = sorted([x.name for x in self.root.iterdir()])
        if not self.sequences:
            raise ValueError("No image found!")
        self.items = []  # (seq, q_idx, is_illu)
        for seq in self.sequences:
            if conf.ignore_large_images and seq in self.ignored_scenes:
                continue
            if conf.subset is not None and conf.subset != seq[0]:
                continue

            self.items.append((seq, seq[0] == "i"))

    def download(self):
        data_dir = self.root.parent
        data_dir.mkdir(exist_ok=True, parents=True)
        tar_path = data_dir / self.url.rsplit("/", 1)[-1]
        torch.hub.download_url_to_file(self.url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)
        tar_path.unlink()

    def get_dataset(self, split):
        assert split in ["val", "test"]
        return self

    def _read_image(self, seq: str, idx: int) -> dict:
        img = load_image(self.root / seq / f"{idx}.ppm", self.conf.grayscale)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        seq, is_illu = self.items[idx]

        target_view = self._read_image(seq, 1)

        source_views, Hs = [], []
        for i in range(2, 7):
            source_view = self._read_image(seq, i)
            if i == 2:
                shape = source_view["image"].shape
            assert shape == source_view["image"].shape, f"Shapes are not equal in {seq}!"

            source_views.append(source_view)

            H = read_homography(self.root / seq / f"H_1_{i}")
            H = source_view["transform"] @ H @ np.linalg.inv(target_view["transform"])
            Hs.append(H.astype(np.float32))

        return {
            "scene": seq,
            "idx": idx,
            "is_illu": is_illu,
            "target_view": target_view,
            "source_views": source_views,
            "Ht2s": Hs
        }

    def __len__(self):
        return len(self.items)

