import logging
from pathlib import Path

import torch

from src.datasets import BaseDataset
from src.utils.image import ImagePreprocessor, load_image


class IMCImageFolder(BaseDataset, torch.utils.data.Dataset):
    """
    Per scenes image folder dataset.
    We get item of a image once.
    """
    default_conf = {
        "glob": "*.jpg",
        "root_folder": "/",
        "preprocessing": ImagePreprocessor.default_conf,
        "grayscale": False,
    }

    def _init(self, conf):
        self.root = conf.root_folder / "set_100" / "images"

        self.images = []
        glob = conf.glob
        assert isinstance(glob, str)
        # image path
        self.images += list(self.root.glob("**/" + glob))

        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        logging.info(f"Found {len(self.images)} images in {self.root}")

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        path = self.images[idx]
        img = load_image(Path(path), grayscale=self.conf.grayscale)
        data = {"name": str(path), **self.preprocessor(img)}
        return data

    def __len__(self):
        return len(self.images)

