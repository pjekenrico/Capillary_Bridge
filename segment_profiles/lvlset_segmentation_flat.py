import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")
import numpy as np
from skimage import measure
from skimage.color import rgb2gray

from segment_profiles.interface_check import preprocess_line
from segment_profiles.tools import (
    rotate_and_crop_img,
    most_common_image_format,
    load_image,
    clip_and_normalize,
)
from segment_profiles.find_lsf import find_lsf, gaussian_filter
from segment_profiles.lvlset_segmentation import get_file_paths, get_image_numbers


class Dataframe(object):

    def __init__(
        self,
        folder_path: str = None,
        boxes: list[tuple] = None,
        lines: list = None,
        angle: float = None,
        flat_top: bool = False,
        segmentation_options: dict = None,
        data_path: str = None,
    ):
        self.metadata = {  # metadata
            "folder_path": folder_path,
            "boxes": boxes,
            "lines": lines,
            "angle": angle,
            "flat_top": flat_top,
            "segmentation_options": segmentation_options,
        }

        self.contact_lines = []
        self.indices = []
        self.plate_height = []  # [h1, h2, ... ]

        if data_path:
            self.load_data(data_path)

        return

    def add_data(self, contact_lines: list[np.ndarray], idx: int, plate_height: float):
        self.contact_lines.append(contact_lines)
        self.indices.append(idx)
        self.plate_height.append(plate_height)
        return

    def save_data(self, filename: str):
        np.savez_compressed(
            filename,
            metadata=self.metadata,
            contact_lines=np.array(self.contact_lines, dtype=object),
            indices=self.indices,
            plate_height=self.plate_height,
        )
        return

    def load_data(self, filename: str):
        data = np.load(filename, allow_pickle=True)
        self.metadata = data["metadata"].item()
        self.contact_lines = data["contact_lines"]
        self.indices = data["indices"]
        self.plate_height = data["plate_height"]
        return


def run_segmentation(
    folder_path: str,
    boxes: list[tuple],
    lines: list,
    angle: float,
    segmentation_options: dict,
    flat_top: bool = False,
) -> Dataframe:

    data_frame = Dataframe(
        folder_path, boxes, lines, angle, flat_top, segmentation_options
    )

    plt.ion()
    _, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

    extension = most_common_image_format(folder_path)
    paths = get_file_paths(folder_path, extension)
    idx_to_analyze = get_image_numbers(paths)

    # idx_to_analyze = segment_indices(numbers)
    x = preprocess_line(folder_path, idx_to_analyze, extension, angle, lines)

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

        # Apply thresholding
        img = clip_and_normalize(img, 0, 255)

        if not "c0" in locals():
            # initialize LSF as binary step function
            c0 = 3
            initial_lsf = c0 * np.ones(img.shape)
            # generate the initial region R0 as two rectangles
            for box in boxes:
                initial_lsf[box[1] : box[-1], box[0] : box[2]] = -c0
            initial_lsf = initial_lsf[lines[0] : lines[-1]]
        else:
            x_init = np.mean(contours[0][:, 0]) / 2
            x_init += np.mean(contours[1][:, 0]) / 2
            x_init = int(np.mean(x_init)) + lines[0]
            initial_lsf = c0 * np.ones(img.shape)
            # generate the initial region R0 as two rectangles
            for box in boxes:
                initial_lsf[x_init - 1 : x_init + 2, box[0] : box[2]] = -c0
            initial_lsf = initial_lsf[lines[0] : lines[-1]]

        old_contours = np.inf * np.ones_like(boxes)
        converged = False

        for phi in find_lsf(
            img=img[lines[0] : lines[-1]],
            initial_lsf=initial_lsf,
            **segmentation_options,
        ):

            ax[0].cla()
            contours = measure.find_contours(phi, 0)
            ax[0].imshow(
                orig_img[lines[0] : lines[-1]],
                interpolation="quadric",
                cmap=plt.get_cmap("gray"),
            )
            ax[0].autoscale(False, axis="x")
            ax[0].autoscale(False, axis="y")
            ax[0].set_title(f"idx= {idx} / {idx_to_analyze[-1]}")
            ax[0].set_ylabel(f"Original image")

            ax[1].cla()
            contours = measure.find_contours(phi, 0)
            proc_img = gaussian_filter(
                img[lines[0] : lines[-1]], segmentation_options["sigma"]
            )
            ax[1].imshow(proc_img, interpolation="quadric", cmap=plt.get_cmap("gray"))
            ax[1].autoscale(False, axis="x")
            ax[1].autoscale(False, axis="y")
            ax[1].set_ylabel(f"Processed image")

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
                profile_idx = np.where(
                    contour[:, 0] > xc + segmentation_options["tol_circle"]
                )[0]
                profile = contour[profile_idx]
                profiles.append(profile)
                ax[0].plot(profile[:, 1], profile[:, 0], linewidth=1)
                ax[1].plot(profile[:, 1], profile[:, 0], linewidth=1)

            ax[0].axhline(xc, color=(0, 0.8, 0.8), linewidth=1)
            ax[1].axhline(xc, color=(0, 0.8, 0.8), linewidth=1)
            plt.pause(0.001)

            if converged:
                # Reorder profiles to have the left one first
                profiles = sorted(profiles, key=lambda x: np.min(x[:, 1]))
                data_frame.add_data(profiles, idx, xc - lines[0])
                break

    return data_frame
