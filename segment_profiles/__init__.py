# segment_profiles/__init__.py

import json, argparse, os
from pprint import pprint

from segment_profiles.lvlset_segmentation import run_segmentation as circ_segmnetation
from segment_profiles.lvlset_segmentation_flat import (
    run_segmentation as flat_segmentation,
)
from segment_profiles.gui_init import preprocess_images


def read_json(json_file: str):
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {json_file} not found. Generating a template file and launching the GUI."
        )


def write_json(data: dict, json_file: str) -> None:
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    return


def run_segmentation_from_json(json_file: str):

    try:
        data = read_json(json_file)
    except FileNotFoundError:
        segmentation_options = {
            "timestep": 3,
            "iter_inner": 10,
            "iter_outer": 400,
            "lmda": 2,
            "alfa": -2,
            "epsilon": 1.5,
            "sigma": 0.5,
            "tol_contours": 5,
            "tol_circle": 5,
            "tol_circle_normal": 0.05,
        }
        boxes, lines, angle, path, flat_top = preprocess_images(folder_path=".")

        data = {
            "folder_path": path,
            "boxes": boxes,
            "lines": lines,
            "angle": angle,
            "flat_top": flat_top,
            "segmentation_options": segmentation_options,
        }

    pprint(data)
    write_json(data, json_file)
    out_file = json_file.replace(".json", "_profiles.npz")

    if data["flat_top"]:
        contact_lines_data = flat_segmentation(**data)
    else:
        del data["flat_top"]
        contact_lines_data = circ_segmnetation(**data)

    contact_lines_data.save_data(out_file)
    return


def main():
    parser = argparse.ArgumentParser(
        description="Program to segment the profiles of a capilary bridge."
    )

    parser.add_argument(
        "path",
        nargs="?",  # Makes the argument optional
        default=None,  # Default to None when the argument is not provided
        help="Path to an existing valid json file. If not provided, the program will generate one such file at the current directory and ask for user input.",
    )

    args = parser.parse_args()

    # If path is None, set it to the current directory
    path = args.path or os.path.join(os.getcwd(), "segmentation_options.json")

    run_segmentation_from_json(path)
    print("DONE")


if __name__ == "__main__":
    main()
