import os, re, glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

matplotlib.use("Qt5Agg")
import numpy as np
from skimage import measure
from skimage.color import rgb2gray

from segment_profiles.interface_check import (
    match_line_with_circle_and_normal,
    preprocess_circle,
)
from segment_profiles.tools import (
    rotate_and_crop_img,
    most_common_image_format,
    load_image,
)
from segment_profiles.find_lsf import find_lsf


class Dataframe(object):

    def __init__(
        self,
        folder_path: str = None,
        boxes: list[tuple] = None,
        lines: list = None,
        angle: float = None,
        segmentation_options: dict = None,
        data_path: str = None,
    ):
        self.metadata = {  # metadata
            "folder_path": folder_path,
            "boxes": boxes,
            "lines": lines,
            "angle": angle,
            "segmentation_options": segmentation_options,
        }

        self.contact_lines = []
        self.indices = []
        self.circle_positions = []  # [[x,y,r], ... ]

        if data_path:
            self.load_data(data_path)

        return

    def add_data(
        self, contact_lines: list[np.ndarray], idx: int, circle_positions: list[float]
    ):
        self.contact_lines.append(contact_lines)
        self.indices.append(idx)
        self.circle_positions.append(circle_positions)
        return

    def save_data(self, filename: str):
        np.savez_compressed(
            filename,
            metadata=self.metadata,
            contact_lines=np.array(self.contact_lines, dtype=object),
            indices=self.indices,
            circle_positions=self.circle_positions,
        )
        return

    def load_data(self, filename: str):
        data = np.load(filename, allow_pickle=True)
        self.metadata = data["metadata"].item()
        self.contact_lines = data["contact_lines"]
        self.indices = data["indices"]
        self.circle_positions = data["circle_positions"]
        return


def get_file_paths(folder_path: str, extension: str = ".tiff"):
    # Load all .tiff files in the folder
    image_files = glob.glob(os.path.join(folder_path, "*" + extension))
    return sorted(image_files)


def get_image_numbers(image_files: list[str]) -> np.ndarray:
    # Extract image numbers from filenames
    image_numbers = []

    for f in image_files:
        # Get the filename without the path
        basename = os.path.basename(f)

        # Use regex to extract the numeric part (digits after the last underscore, before the extension)
        match = re.search(r"_(\d+)\.[a-zA-Z0-9]+$", basename)

        if match:
            # Convert the extracted number to an integer
            image_number = int(match.group(1))
            image_numbers.append(image_number)
        else:
            raise ValueError(f"Filename format is incorrect: {basename}")

    return np.array(image_numbers)


def run_segmentation(
    folder_path: str,
    boxes: list[tuple],
    lines: list,
    angle: float,
    segmentation_options: dict,
) -> Dataframe:

    data_frame = Dataframe(folder_path, boxes, lines, angle, segmentation_options)

    plt.ion()
    _, ax = plt.subplots()

    extension = most_common_image_format(folder_path)
    paths = get_file_paths(folder_path, extension)
    idx_to_analyze = get_image_numbers(paths)

    # idx_to_analyze = segment_indices(numbers)
    x, yc, r = preprocess_circle(folder_path, idx_to_analyze, extension, angle, lines)

    for idx, xc in zip(idx_to_analyze, x):
        print(f"\nidx= {idx}")
        orig_img = load_image(folder_path, idx, extension)
        orig_img = rotate_and_crop_img(orig_img, angle)

        try:
            img = rgb2gray(orig_img)
        except:
            img = orig_img
        if img is None:
            continue

        if not "c0" in locals():
            # initialize LSF as binary step function
            c0 = 3
            initial_lsf = c0 * np.ones(img.shape)
            # generate the initial region R0 as two rectangles
            for box in boxes:
                initial_lsf[box[1] : box[-1], box[0] : box[2]] = -c0
            initial_lsf = initial_lsf[lines[0] : lines[-1]]
        else:
            initial_lsf = 2 * c0 / np.pi * np.arctan(phi)

        img = np.interp(img, [np.min(img), np.max(img)], [0, 255])
        old_contours = np.inf * np.ones_like(boxes)
        converged = False

        for phi in find_lsf(
            img=img[lines[0] : lines[-1]],
            initial_lsf=initial_lsf,
            **segmentation_options,
        ):
            ax.cla()
            contours = measure.find_contours(phi, 0)
            ax.imshow(orig_img, interpolation="quadric", cmap=plt.get_cmap("gray"))
            ax.autoscale(False, axis="x")
            ax.autoscale(False, axis="y")

            if len(contours) < 2:
                converged = True
                print("Reached merging point!")
                break

            if len(contours) == len(boxes):
                converged = True
                for cont, old_cont in zip(contours, old_contours):
                    if not cont.shape == old_cont.shape:
                        converged = False
                        break

                    err = np.linalg.norm(cont - old_cont)
                    if err > segmentation_options["tol_contours"]:
                        converged = False
                        break
                    print("Converged with norms:", err)
            old_contours = contours
            profiles = []

            # Reorder contours to have the left one first
            for contour in sorted(contours, key=lambda x: np.min(x[:, 1])):
                profile_idx = match_line_with_circle_and_normal(
                    contour,
                    orig_img[lines[0] : lines[-1]],
                    epsilon=segmentation_options["tol_circle"],
                    normal_tolerance=segmentation_options["tol_circle_normal"],
                    circle_data=(xc, yc, r),
                )
                profile = contour[profile_idx]
                profiles.append(profile)
                ax.plot(profile[:, 1], profile[:, 0] + lines[0], linewidth=1)

            # Update the left image
            ax.add_patch(
                Circle(
                    (yc, xc + lines[0]),
                    r,
                    facecolor="none",
                    edgecolor=(0, 0.8, 0.8),
                    linewidth=1,
                )
            )
            plt.pause(0.001)

            if converged:
                # Reorder profiles to have the left one first
                profiles = sorted(profiles, key=lambda x: np.min(x[:, 1]))
                data_frame.add_data(profiles, idx, [xc, yc, r])
                break

    return data_frame
