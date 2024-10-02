import os
import numpy as np
from skimage import transform
from collections import Counter


def rotate_and_crop_img(image: np.ndarray, angle: float | int) -> np.ndarray:
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Perform the rotation using scikit-image (mode='constant' ensures padding with black if necessary)
    rotated = transform.rotate(image, angle, resize=True, mode="constant", cval=0)

    # Calculate the absolute value of cosine and sine of the rotation angle in radians
    angle_rad = np.deg2rad(angle)
    cos_theta = np.abs(np.cos(angle_rad))
    sin_theta = np.abs(np.sin(angle_rad))

    # Compute the dimensions of the bounding box for the rotated image
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    # Compute the valid dimensions to avoid black borders
    valid_w = int(w * cos_theta - h * sin_theta)
    valid_h = int(h * cos_theta - w * sin_theta)

    # Calculate the top-left coordinates to crop the rotated image
    x_start = (new_w - valid_w) // 2
    y_start = (new_h - valid_h) // 2

    # Crop the valid region from the rotated image
    cropped_rotated_image = rotated[
        y_start : y_start + valid_h, x_start : x_start + valid_w
    ]

    return cropped_rotated_image


def most_common_image_format(folder_path):
    # Define a set of image extensions
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }

    # List all files in the folder and filter by image extensions
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        return ".tif"

    # Extract extensions and count occurrences
    extensions = [os.path.splitext(f)[1].lower() for f in image_files]
    most_common_ext = Counter(extensions).most_common(1)[0][0]

    return most_common_ext
