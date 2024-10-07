import os, re, glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

matplotlib.use("Qt5Agg")
import numpy as np
from skimage import measure
from skimage.color import rgb2gray

from segment_profiles.interface_check import (
    find_circle,
    match_line_with_circle_and_normal,
)
from segment_profiles.tools import rotate_and_crop_img, most_common_image_format
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

    def add_data(self, contact_lines, idx, circle_positions):
        self.contact_lines.append(contact_lines)
        self.indices.append(idx)
        self.circle_positions.append(circle_positions)
        return

    def save_data(self, filename):
        np.savez_compressed(
            filename,
            metadata=self.metadata,
            contact_lines=np.array(self.contact_lines, dtype=object),
            indices=self.indices,
            circle_positions=self.circle_positions,
        )
        return

    def load_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.metadata = data["metadata"].item()
        self.contact_lines = data["contact_lines"]
        self.indices = data["indices"]
        self.circle_positions = data["circle_positions"]
        return


def get_file_paths(folder_path, extension=".tiff"):
    # Load all .tiff files in the folder
    image_files = glob.glob(os.path.join(folder_path, "*" + extension))
    return sorted(image_files)


def get_image_numbers(image_files):
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


def segment_indices(numbers):
    reversed_numbers = np.flip(numbers)
    idx_to_analyze = [numbers[0]]
    idx = 0
    step = 1
    while idx < len(reversed_numbers):
        idx_to_analyze.append(reversed_numbers[int(idx)])
        idx += step
        if idx >= 49:
            step = 100
        if idx >= 499:
            step = 200
    idx_to_analyze = np.sort(idx_to_analyze)
    if idx_to_analyze[0] == idx_to_analyze[1]:
        idx_to_analyze = idx_to_analyze[1:]

    return idx_to_analyze


def load_image(folder_path, image_number, extension=".tiff"):
    # Format the number with leading zeros (e.g., 00001)
    number_str = f"{image_number:05d}"
    image_files = glob.glob(os.path.join(folder_path, "*" + extension))

    # Find the file that matches the pattern imageseriesname_00001.<extension>
    matching_files = [f for f in image_files if f"_{number_str}" in os.path.basename(f)]

    if not matching_files:
        number_str = f"{image_number:04d}"

        # Find the file that matches the pattern imageseriesname_00001.<extension>
        matching_files = [
            f for f in image_files if f"_{number_str}" in os.path.basename(f)
        ]

    if matching_files:
        # Use the first matching file (in case there are multiple)
        image_path = os.path.join(folder_path, matching_files[0])

        # Check if the file exists and load the image
        if os.path.exists(image_path):
            return plt.imread(image_path)

    return None


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
    numbers = get_image_numbers(paths)

    idx_to_analyze = segment_indices(numbers)

    for idx in idx_to_analyze:
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
            c0 = 2
            initial_lsf = c0 * np.ones(img.shape)
            # generate the initial region R0 as two rectangles
            for box in boxes:
                initial_lsf[box[1] : box[-1], box[0] : box[2]] = -c0
            initial_lsf = initial_lsf[lines[0] : lines[-1]]
        else:
            initial_lsf = phi

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
                )

                profile = contour[profile_idx]
                profiles.append(profile)
                ax.plot(profile[:, 1], profile[:, 0] + lines[0], linewidth=2)

            xc, yc, r = find_circle(orig_img)
            # Update the left image
            ax.add_patch(
                Circle(
                    (yc, xc), r, facecolor="none", edgecolor=(0, 0.8, 0.8), linewidth=2
                )
            )
            plt.pause(0.001)

            if converged:
                # Reorder profiles to have the left one first
                profiles = sorted(profiles, key=lambda x: np.min(x[:, 1]))
                data_frame.add_data(profiles, idx, [xc, yc, r])
                break

    return data_frame
