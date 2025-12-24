from pathlib import Path

scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"



def read_quadruples(quad_file_name):
    path = scene_lists_path / quad_file_name
    with open(path, 'r') as f:
        quadruples = f.readlines()
    # remove '\n' from each line
    quadruples = [quad.strip() for quad in quadruples]
    return quadruples

if __name__ == '__main__':
    quad_file_name = "valid_quadruples.txt"
    quadruples = read_quadruples(quad_file_name)

    # turn quadruples into pairs
    pairs = []
    for quad in quadruples:
        quad = quad.split()
        pairs.append((quad[0], quad[1]))
        pairs.append((quad[0], quad[2]))
        pairs.append((quad[0], quad[3]))

    # write pairs to file
    pair_file_name = "valid_pairs_hard.txt"
    with open(scene_lists_path / pair_file_name, 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"Pairs written to {pair_file_name}")
    print(f"Number of pairs: {len(pairs)}")


