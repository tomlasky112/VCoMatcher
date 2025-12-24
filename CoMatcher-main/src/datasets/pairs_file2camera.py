from pathlib import Path

import numpy as np

from src import logger
from src.settings import MEGA_DATA_PATH

scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"


class IO:
    @staticmethod
    def extract_images(images_file_path):
        with open(images_file_path, 'r') as f:
            # remove /n from the end of each line
            pairs = f.readlines()
        pairs = [pair.strip().split(' ') for pair in pairs]
        return pairs

    @staticmethod
    def save_to_file(data, save_path):
        try:
            with open(save_path, 'w') as f:
                for item in data:
                    # write dict by order of "name0", "name1", "K0", "K1", "R", "t"
                    f.write(str(item["name0"]) + " " + str(item["name1"]) + " " + str(item["K0"])
                            + " " + str(item["K1"]) + " " + str(item["R"]) + " " + str(item["t"]) + "\n")

        except Exception as e:
            logger.error("Error saving data to file: %s", e)
            return False






class MegaDepth:
    def __init__(self):
        self.root = MEGA_DATA_PATH / "MegaDepth"
        assert self.root.exists(), self.root
        self.info_dir = self.root / "scene_info"
        assert self.info_dir.exists(), self.info_dir

        scenes = ['0015', '0022']
        # build data table

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}

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
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"]
            self.scenes.append(scene)


    def read_KT(self, scene, idx):
        K = self.intrinsics[scene][idx]
        T = self.poses[scene][idx]

        return {
            "T_w2cam": T[0].astype(np.float32),
            "K": K[0].astype(np.float32),
        }



    def get_camera_model(self, pairs):
        info = []

        for image_path0, image_path1 in pairs:
            # scene is same for both images
            assert image_path0.split('/')[0] == image_path1.split('/')[0]
            scene = image_path0.split('/')[0]
            assert scene in self.scenes

            # Update image path
            image_path0 = "Undistorted_SfM/" + image_path0
            image_path1 = "Undistorted_SfM/" + image_path1

            idx0 = np.where(self.images[scene] == image_path0)[0]
            idx1 = np.where(self.images[scene] == image_path1)[0]

            data0 = self.read_KT(scene, idx0)
            data1 = self.read_KT(scene, idx1)

            # ndarray 2 single element list
            K0 = ' '.join([str(i) for i in data0["K"].flatten().tolist()])
            K1 = ' '.join([str(i) for i in data1["K"].flatten().tolist()])

            T0to1 = data1["T_w2cam"] @ np.linalg.inv(data0["T_w2cam"])
            R = ' '.join([str(i) for i in T0to1[:3, :3].flatten().tolist()])
            t = ' '.join([str(i) for i in T0to1[:3, 3].flatten().tolist()])

            data = {
                "K0": K0,
                "K1": K1,
                "R": R,
                "t": t,
                "name0": image_path0.split("/")[1] + "/" + image_path0.split("/")[3],
                "name1": image_path1.split("/")[1] + "/" + image_path1.split("/")[3]
            }
            info.append(data)
        return info





if __name__ == '__main__':
    pairs_name = 'valid_pairs_hard.txt'
    pairs = IO.extract_images(scene_lists_path / pairs_name)

    mega_depth = MegaDepth()
    results = mega_depth.get_camera_model(pairs)

    # save to file
    save_name = 'pairs_calibrated_valid_pairs_hard.txt'
    IO.save_to_file(results, scene_lists_path / save_name)


