import argparse
import logging
import os
import shutil
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf

from src.geometry.wrappers import Camera, Pose
from src.models.cache_loader import CacheLoader
from src.settings import DATA_PATH, MEGA_DATA_PATH
from src.utils.image import ImagePreprocessor, load_image
from src.utils.tools import fork_rng
from src.visualization.viz2d import plot_heatmaps, plot_image_grid
from src.datasets.base_dataset import BaseDataset
from src.datasets.utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics
logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return [data[i] for i in selected]
    else:
        return data

class MultiviewMegaDepth(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "MegaDepth/",
        "depth_subpath": "depth_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",  # @TODO: intrinsics problem?
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": None,
        "val_file": None,
        # data sampling
        "sources_views_num": 3,
        "min_hardest_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 2,
        "i0_sample_num": 5,
        "save_items_to_txt": False,
        "clip_num": 10000,
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (MEGA_DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the MegaDepth dataset.")
            self.download()

    def download(self):
        data_dir = MEGA_DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "megadepth_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        for tar_name, out_name in (
            ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
            ("depth_undistorted.tar.gz", self.conf.depth_subpath),
            ("scene_info.tar.gz", self.conf.info_dir),
        ):
            tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(url_base + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=tmp_dir)
            tar_path.unlink()
            shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split):
        return _Dataset(self.conf, split)



class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = MEGA_DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        split_conf = conf[split + "_split"]  # txt file
        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)
            logger.info("dataset %s: loading features from cache", split)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}

        # load metadata
        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                info = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load scene info for scene %s at %s.", scene, path
                )
                continue
            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"]
            self.scenes.append(scene)

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        logger.info("Start sampling new items.")
        split = self.split

        # 2case: 1. train and num_pos   2. val and have val_file
        if split == "train":
            self.quadruples_all = {}
            for scene in self.scenes:
                path = self.info_dir / (scene + ".npz")
                assert path.exists(), path
                info = np.load(str(path), allow_pickle=True)

                # 1. get valid images from all images
                valid = (self.images[scene] != None) & (  # noqa: E711
                        self.depths[scene] != None  # noqa: E711
                )
                ind = np.where(valid)[0]
                mat = info["overlap_matrix"][valid][:, valid]

                # 2. filter images with resonable overlap
                good = (mat > self.conf.min_hardest_overlap) & (mat <= self.conf.max_overlap)

                # 3. get all possible quadruples
                quadruples_all = []
                pairs = np.stack(np.where(good), -1)

                # 3.1. get all good triplets of i1, i2, i3
                # remove duplicates of pairs
                pairs = np.sort(pairs, axis=1)
                pairs = np.unique(pairs, axis=0)

                # random shuffle and pick n/2 sample to reverse
                np.random.RandomState(0).shuffle(pairs)
                num_reverse = len(pairs) // 2
                indices = np.random.RandomState(0).choice(pairs.shape[0], num_reverse, replace=False)
                pairs[indices] = pairs[indices][:, ::-1]

                triples = []
                for i1, i2 in pairs:
                    if len(triples) > self.conf.clip_num:
                        break
                    # get all ind related to i1
                    i1_related = np.concatenate([pairs[pairs[:, 0] == i1, 1], pairs[pairs[:, 1] == i1, 0]])
                    np.random.RandomState(0).shuffle(i1_related)
                    for i3 in i1_related:
                        if good[i2, i3]:
                            triples.append((i1, i2, i3))
                            break
                # 3.2. triplets remove duplicates
                unique_triples = set(tuple(sorted(triple)) for triple in triples)

                seen = set()
                final_triples = []
                for triple in unique_triples:
                    if not any(element in seen for element in triple):
                        final_triples.append(triple)
                        seen.update(triple)

                # 3.3. get i0: simultaneously related with triplets
                for i1, i2, i3 in final_triples:
                    i1_related = np.concatenate([pairs[pairs[:, 0] == i1, 1], pairs[pairs[:, 1] == i1, 0]])
                    i2_related = np.concatenate([pairs[pairs[:, 0] == i2, 1], pairs[pairs[:, 1] == i2, 0]])
                    i3_related = np.concatenate([pairs[pairs[:, 0] == i3, 1], pairs[pairs[:, 1] == i3, 0]])
                    i0_array = np.intersect1d(np.intersect1d(i1_related, i2_related), i3_related)
                    np.random.RandomState(0).shuffle(i0_array)
                    count = 0
                    for i0 in i0_array:
                        if good[i0, i1] and good[i0, i2] and good[i0, i3]:
                            eastest_overlap = np.max([mat[i0, i1], mat[i0, i2], mat[i0, i3]])
                            quadruples_all.append((i0, i1, i2, i3, eastest_overlap))
                            count += 1
                            if count >= self.conf.i0_sample_num:
                                break
                self.quadruples_all[scene] = quadruples_all
        self.sample_every_epoch(seed)

    def sample_every_epoch(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        num_pos = num_per_scene

        # 2case: 1. train and num_pos   2. val and have val_file
        if split != "train" and self.conf[split + "_file"] is not None:
            # case 2
            assert num_pos is None
            quadruples_path = scene_lists_path / self.conf[split + "_file"]
            for line in quadruples_path.read_text().rstrip("\n").split("\n"):
                im0, im1, im2, im3 = line.split(" ")
                scene = im0.split("/")[0]
                assert im1.split("/")[0] == im2.split("/")[0] == im3.split("/")[0] == scene
                im0, im1, im2, im3 = [self.conf.image_subpath + im for im in [im0, im1, im2, im3]]
                assert im0 in self.images[scene]
                assert im1 in self.images[scene]
                assert im2 in self.images[scene]
                assert im3 in self.images[scene]
                i0 = np.where(self.images[scene] == im0)[0][0]
                i1 = np.where(self.images[scene] == im1)[0][0]
                i2 = np.where(self.images[scene] == im2)[0][0]
                i3 = np.where(self.images[scene] == im3)[0][0]
                self.items.append((scene, i0, i1, i2, i3, 1, 1, 1, 1))
        else:
            for scene in self.scenes:
                quadruples_all = self.quadruples_all[scene]

                path = self.info_dir / (scene + ".npz")
                assert path.exists(), path
                info = np.load(str(path), allow_pickle=True)

                # 1. get valid images from all images
                valid = (self.images[scene] != None) & (  # noqa: E711
                        self.depths[scene] != None  # noqa: E711
                )
                ind = np.where(valid)[0]
                mat = info["overlap_matrix"][valid][:, valid]
                # 4. devide results into bins
                num_bins = self.conf.num_overlap_bins
                assert num_bins == 3
                bin_width = [0.2, 0.2, self.conf.max_overlap - self.conf.min_hardest_overlap - 0.4]

                quadruples_devide = []
                qua_overlap_array = np.array([item[-1] for item in quadruples_all])

                bin_min = self.conf.min_hardest_overlap
                for i in range(num_bins):
                    bin_max = bin_min + bin_width[i]
                    index = ((qua_overlap_array >= bin_min) & (qua_overlap_array < bin_max))
                    index = np.where(index)[0]
                    quadruples_bin = [quadruples_all[ind] for ind in index]
                    quadruples_devide.append(quadruples_bin)
                    bin_min = bin_max

                # 5. sample from each bin
                quadruples = []
                box = num_pos
                for i in range(num_bins - 1, -1, -1):
                    quadruples_bin = quadruples_devide[i]
                    if box > 0:
                        if len(quadruples_bin) <= box:
                            quadruples.extend(quadruples_bin)
                            box -= len(quadruples_bin)
                        else:
                            quadruples.extend(sample_n(quadruples_bin, box, seed))
                            box = 0
                if box > 0:
                    logger.info("The scene %s has no reasonable data", scene)

                # 6. save results
                quadruples = [
                    (scene, ind[i0], ind[i1], ind[i2], ind[i3], mat[i0, i1], mat[i0, i2], mat[i0, i3], ho)
                    for i0, i1, i2, i3, ho in quadruples
                ]
                self.items.extend(quadruples)
        np.random.RandomState(seed).shuffle(self.items)

        # count items in different bins
        if split == "train":
            counter = [0] * 7
            for idx in range(len(self.items)):
                scene, i0, i1, i2, i3, o01, o02, o03, ho = self.items[idx]
                bin_width = (self.conf.max_overlap - self.conf.min_hardest_overlap) / 7  # bin_width = 0.1
                # count the number of items in each bin
                for i, o in enumerate([o01, o02, o03]):
                    bin_index = int((o - self.conf.min_hardest_overlap - 0.0001) // bin_width)
                    counter[bin_index] += 1
            counter_ = [0] * 3
            counter_[0] = counter[0] + counter[1]
            counter_[1] = counter[2] + counter[3]
            counter_[2] = counter[4] + counter[5] + counter[6]
            logger.info("The number of items in each bin: %s", counter_)








    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]

        # read pose data
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        # read image
        if self.conf.read_image:
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

        # read depth
        if self.conf.read_depth:
            depth_path = (
                self.root / self.conf.depth_subpath / scene / (path.stem + ".h5")
            )
            with h5py.File(str(depth_path), "r") as f:
                depth = f["/depth"].__array__().astype(np.float32, copy=False)
                depth = torch.Tensor(depth)[None]
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None

        # add random rotations
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = np.rot90(img, k=-k, axes=(-2, -1))
                if self.conf.read_depth:
                    depth = np.rot90(depth, k=-k, axes=(-2, -1)).copy()
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)

        name = path.name

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]
        K = scale_intrinsics(K, data["scales"])

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": depth,
            "camera": Camera.from_calibration_matrix(K).float(),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].copy()
                x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)


    def getitem(self, idx):
        scene, i0, i1, i2, i3, o01, o02, o03, ho = self.items[idx]
        target_view = self._read_view(scene, i0)

        source_views = []
        source_views_name = ''
        for i in range(self.conf.sources_views_num):
            source_views.append(self._read_view(scene, locals()[f'i{i+1}']))
            source_views_name += source_views[i]["name"] + " "
            source_views = [{k: v for k, v in sv.items() if k not in ["scene", "name"]} for sv in source_views]


        Tt2s = [source_view["T_w2cam"] @ target_view["T_w2cam"].inv() for source_view in source_views]
        Ts2t = [target_view["T_w2cam"] @ source_view["T_w2cam"].inv() for source_view in source_views]
        overlap = [o01, o02, o03]

        return {
            "scene": scene,
            "ind": idx,
            "target_view": target_view,
            "source_views": source_views,
            "source_views_name": source_views_name,
            "Tt2s": Tt2s,
            "Ts2t": Ts2t,
            "overlap": overlap,
            "eastest_overlap": ho
        }


    def __len__(self):
        return len(self.items)

