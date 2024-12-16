import os, glob
import numpy as np
from skimage import transform
from collections import Counter
import matplotlib.pyplot as plt

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


def load_image(
    folder_path: str, image_number: int, extension: str = ".tiff"
) -> np.ndarray | None:
    # Format the number with leading zeros (e.g., 00001)
    number_str = f"{image_number:05d}"
    image_files = glob.glob(os.path.join(folder_path, "*" + extension))

    # Find the file that matches the pattern imageseriesname_00001.<extension>
    matching_files = [f for f in image_files if f"_{number_str}" in os.path.basename(f)]

    if not matching_files:
        number_str = f"{image_number:04d}"

        # Find the file that matches the pattern imageseriesname_0001.<extension>
        matching_files = [
            f for f in image_files if f"_{number_str}" in os.path.basename(f)
        ]

    if not matching_files:
        number_str = f"{image_number:03d}"

        # Find the file that matches the pattern imageseriesname_0001.<extension>
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


def clip_and_normalize(image, min_val, max_val):
    """
    Clips the pixel values of an image to a specified range and normalizes it to 0-255.
    Automatically handles images in [0, 1] float format by converting them to uint8.

    Parameters:
    - image (np.ndarray): Input image (grayscale or single channel).
    - min_val (int): Minimum pixel value threshold.
    - max_val (int): Maximum pixel value threshold.

    Returns:
    - np.ndarray: The processed image with values normalized to 0-255.
    """
    # Check if the image is in float format [0, 1] and convert to uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = (image * 255).astype(np.uint8)

    # Clip values outside the range
    clipped_image = np.clip(image, min_val, max_val)
    
    # Avoid division by zero
    if max_val > min_val:
        # Normalize to 0-255 range
        normalized_image = (
            ((clipped_image - min_val) / (max_val - min_val)) * 255
        ).astype(np.uint8)
    else:
        # If range is invalid, return a zeroed image
        normalized_image = np.zeros_like(image, dtype=np.uint8)

    return normalized_image