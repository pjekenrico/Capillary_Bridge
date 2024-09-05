# segment_profiles/__init__.py

import json, argparse, os
from pprint import pprint

from segment_profiles.lvlset_segmentation import run_segmentation
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
            "timestep": 2,  # time step
            "iter_inner": 10,  # number of iterations of the level set evolution in the inner loop
            "iter_outer": 400,  # number of iterations of the level set evolution in the outer loop
            "lmda": 4,  # coefficient of the weighted length term L(phi)
            "alfa": -1,  # coefficient of the weighted area term A(phi)
            "epsilon": 2,  # parameter that specifies the width of the DiracDelta function
            "sigma": 1.5,  # smoothing the image (Gaussian kernel)
            "tol_contours": 0.5,  # tolerance of the gradient descent method
            "tol_circle": 2,  # tolerance of the gradient descent method
            "tol_circle_normal": 0.1,  # tolerance of the gradient descent method
        }
        boxes, lines, angle, path = preprocess_images(folder_path=".")

        data = {
            "folder_path": path,
            "boxes": boxes,
            "lines": lines,
            "angle": angle,
            "segmentation_options": segmentation_options,
        }

    pprint(data)
    write_json(data, json_file)
    run_segmentation(**data)


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
