from pathlib import Path

scene_lists_path = Path(__file__).parent.parent / "megadepth_scene_lists"

def get_scenes_list(file_name):
    path = scene_lists_path / file_name
    with open(path, 'r') as f:
        scenes = f.readlines()
    # remove '\n'
    return [scene.strip() for scene in scenes]


if __name__ == '__main__':
    old_train_scenes_path = 'train_scenes_clean.txt'
    val_scenes_path = 'valid_scenes.txt'
    test_scenes_path = 'test_scenes_clean.txt'

    old_train_scenes = get_scenes_list(old_train_scenes_path)
    val_scenes = get_scenes_list(val_scenes_path)
    test_scenes = get_scenes_list(test_scenes_path)

    train_scenes = list(set(old_train_scenes).union(set(val_scenes), set(test_scenes)))
    # sort
    train_scenes.sort()
    # write to file
    destination_path = scene_lists_path / 'train_scenes_final.txt.txt'
    with open(destination_path, 'w') as f:
        # add '\n'
        train_scenes = [scene + '\n' for scene in train_scenes]
        f.writelines(train_scenes)
