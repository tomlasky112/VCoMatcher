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
            quads = f.readlines()
        quads = [quad.strip().split(' ') for quad in quads]
        return quads

    @staticmethod
    def save_to_file(data, save_path):
        try:
            with open(save_path, 'w') as f:
                for item in data:
                    # write dict by order of "name", "K", "R", "t"
                    f.write(f"{item['name0']} {item['name1']} {item['name2']} {item['name3']}"
                            f" {item['K0']} {item['K1']} {item['K2']} {item['K3']}"
                            f" {item['R0to1']} {item['t0to1']} {item['R0to2']} {item['t0to2']} {item['R0to3']} {item['t0to3']}\n")
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

    def get_camera_model(self, quads):
        info = []

        for image_path0, image_path1, image_path2, image_path3 in quads:
            assert image_path0.split('/')[0] == image_path1.split('/')[0] == image_path2.split('/')[0] == image_path3.split('/')[0]
            scene = image_path0.split('/')[0]
            assert scene in self.scenes

            # Update image path
            image_path0 = "Undistorted_SfM/" + image_path0
            image_path1 = "Undistorted_SfM/" + image_path1
            image_path2 = "Undistorted_SfM/" + image_path2
            image_path3 = "Undistorted_SfM/" + image_path3

            idx0 = np.where(self.images[scene] == image_path0)[0]
            idx1 = np.where(self.images[scene] == image_path1)[0]
            idx2 = np.where(self.images[scene] == image_path2)[0]
            idx3 = np.where(self.images[scene] == image_path3)[0]

            data0 = self.read_KT(scene, idx0)
            data1 = self.read_KT(scene, idx1)
            data2 = self.read_KT(scene, idx2)
            data3 = self.read_KT(scene, idx3)

            K0 = ' '.join([str(i) for i in data0["K"].flatten().tolist()])
            K1 = ' '.join([str(i) for i in data1["K"].flatten().tolist()])
            K2 = ' '.join([str(i) for i in data2["K"].flatten().tolist()])
            K3 = ' '.join([str(i) for i in data3["K"].flatten().tolist()])

            T0to1 = data1["T_w2cam"] @ np.linalg.inv(data0["T_w2cam"])
            R0to1 = ' '.join([str(i) for i in T0to1[:3, :3].flatten().tolist()])
            t0to1 = ' '.join([str(i) for i in T0to1[:3, 3].flatten().tolist()])

            T0to2 = data2["T_w2cam"] @ np.linalg.inv(data0["T_w2cam"])
            R0to2 = ' '.join([str(i) for i in T0to2[:3, :3].flatten().tolist()])
            t0to2 = ' '.join([str(i) for i in T0to2[:3, 3].flatten().tolist()])

            T0to3 = data3["T_w2cam"] @ np.linalg.inv(data0["T_w2cam"])
            R0to3 = ' '.join([str(i) for i in T0to3[:3, :3].flatten().tolist()])
            t0to3 = ' '.join([str(i) for i in T0to3[:3, 3].flatten().tolist()])

            data = {
                "name0": "/".join(image_path0.split("/")[1:4]),
                "name1": "/".join(image_path1.split("/")[1:4]),
                "name2": "/".join(image_path2.split("/")[1:4]),
                "name3": "/".join(image_path3.split("/")[1:4]),
                "K0": K0,
                "K1": K1,
                "K2": K2,
                "K3": K3,
                "R0to1": R0to1,
                "t0to1": t0to1,
                "R0to2": R0to2,
                "t0to2": t0to2,
                "R0to3": R0to3,
                "t0to3": t0to3,
            }
            info.append(data)

        return info





if __name__ == '__main__':
    quads_name = 'valid_quadruples.txt'
    quads = IO.extract_images(scene_lists_path / quads_name)

    mega_depth = MegaDepth()
    results = mega_depth.get_camera_model(quads)

    # save to file
    save_name = 'quads_calibrated.txt'
    IO.save_to_file(results, scene_lists_path / save_name)




