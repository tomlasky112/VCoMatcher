import shutil
from pathlib import Path
from src.settings import root, MEGA_DATA_PATH

scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"
DESTINATION_FOLDER = root / "data" / "megadepth" / "eval_images"

image_sub_path = MEGA_DATA_PATH / "MegaDepth" / "Undistorted_SfM"

def get_quadruples_list(file_name):
    path = scene_lists_path / file_name
    with open(path, 'r') as f:
        quadruples = f.readlines()
    return quadruples


if __name__ == '__main__':
    # get the quadruples list from file
    file_name = "valid_quadruples.txt"
    quadruples = get_quadruples_list(file_name)

    for idx, quadruple in enumerate(quadruples):
        quadruple = quadruple.strip().split()
        for i, image in enumerate(quadruple):
            image_path = image_sub_path / image
            # create destination folder if it does not exist
            destination_folder = DESTINATION_FOLDER / image.split('/')[0] / f"{idx}"
            destination_folder.mkdir(parents=True, exist_ok=True)
            # copy image to the destination folder
            shutil.copy(image_path, destination_folder)
            print(f"copied {image_path} to {destination_folder}")

            if i == 0:
                # create a txt named by image in destination folder
                txt_path = destination_folder / f"{image.split('/')[-1].split('.')[0]}.txt"
                with open(txt_path, 'w') as f:
                    pass
    print("done")

