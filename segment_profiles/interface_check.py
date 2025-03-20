import numpy as np
from scipy import signal, stats

from segment_profiles.tools import load_image, rotate_and_crop_img


def fit_circle_2d(x, y, w=[]):
    """FIT CIRCLE 2D
    - Find center [xc, yc] and radius r of circle fitting to set of 2D points
    - Optionally specify weights for points
    - Optionally specify known radius r_known

    - Implicit circle function:
    (x-xc)^2 + (y-yc)^2 = r^2
    (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
    c[0]*x + c[1]*y + c[2] = x^2+y^2

    - Solution by method of least squares:
    A*c = b, c' = argmin(||A*c - b||^2)
    A = [x y 1], b = [x^2+y^2]
    """

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r


def find_circle(img: np.ndarray):

    scharr = np.array(
        [[-3j, 0 - 10j, -3j], [0, 0 + 0j, 0], [3j, 0 + 10j, 3j]]
    )  # Gx + j*Gy

    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)
    grad = signal.convolve2d(img, scharr, boundary="symm", mode="same")
    grad = np.absolute(grad)

    grad *= grad > np.max(grad) * 0.7
    grad[grad > 0] = 1

    intensity = np.sum(grad, axis=1)

    criterium = intensity[1:] - intensity[:-1]
    criterium = np.insert(criterium, 0, criterium[0])
    criterium = (intensity * (criterium < 0)) > 0

    filter_length = 10
    win = [1 for k in range(filter_length + 1)]
    win += [0 for k in range(filter_length)]
    win = np.array(win, dtype=float)
    win /= np.sum(win)

    criterium = signal.convolve(criterium, win, mode="same")
    criterium = criterium > 0

    init_bmp = 0
    finish_bmp = 0
    for k in range(1, len(criterium)):
        if ~init_bmp and criterium[k - 1] == 0 and criterium[k] == 1:
            init_bmp = k
        elif ~finish_bmp and criterium[k - 1] == 1 and criterium[k] == 0:
            finish_bmp = k
            break

    grad[finish_bmp:, :] = 0

    idx = np.where(grad)

    # Generate weights that are large at the left and right edges and smaller in the middle, eg with a normal distribution
    std = np.std(idx[1])
    mu = img.shape[1] / 2
    w = 1 - np.exp(-0.5 * ((idx[1] - mu) / std) ** 2)

    xc, yc, r = fit_circle_2d(idx[0], idx[1], w=w)

    return xc, yc, r


def remove_true_islands(arr, max_size):
    """
    Removes islands of True values beyond a certain size.

    Parameters:
    arr (list of bool): The input 1D boolean array.
    max_size (int): The maximum allowed size for True islands.

    Returns:
    list of bool: The modified boolean array with large True islands removed.
    """
    import itertools

    result = arr[:]

    # Identify islands of True values
    start = None
    for i, (key, group) in enumerate(
        itertools.groupby(enumerate(arr), key=lambda x: x[1])
    ):
        if key:  # True island
            indices = [idx for idx, val in group]
            if len(indices) < max_size:
                for idx in indices:
                    result[idx] = False  # Remove the island

    return result


def match_line_with_circle_and_normal(
    interface_points: np.ndarray,
    img: np.ndarray,
    epsilon: float = 2.0,
    normal_tolerance: float = 0.05,
    tol_bath: int = 0,
    circle_data: list = None,
):
    """
    Identify which parts of the interface line match well with the circle, including normal direction checks using numpy.

    Parameters:
    - interface_points: List of tuples [(x1, y1), (x2, y2), ...] representing the interface line.
    - circle_center: Tuple (xc, yc) representing the center of the circle.
    - radius: The radius of the circle.
    - epsilon: The tolerance level for distance comparison. Default is 0.01.
    - normal_tolerance: Tolerance level for normal alignment (dot product). Default is 0.1.

    Returns:
    - matches: List of booleans [True, False, ...] indicating whether each point matches with the circle considering both distance and normal alignment.
    """

    if circle_data is None:
        xc, yc, r = find_circle(img)
    else:
        xc, yc, r = circle_data
    circle_center = np.array([xc, yc])

    # Compute the circle normals and their normalized forms
    circle_normals = interface_points - circle_center
    circle_normals_magnitude = np.linalg.norm(circle_normals, axis=1, keepdims=True)
    circle_normals_direction = circle_normals / circle_normals_magnitude

    # Compute the tangent vectors
    diffs = interface_points[2:] - interface_points[:-2]
    diffs = np.vstack([interface_points[1] - interface_points[0], diffs])
    diffs = np.vstack([diffs, interface_points[-1] - interface_points[-2]])

    # Calculate the interface normals (rotate by 90 degrees)
    interface_normals = np.empty_like(diffs)
    interface_normals[:, 0] = -diffs[:, 1]
    interface_normals[:, 1] = diffs[:, 0]

    # Normalize the interface normals
    interface_normals_magnitude = np.linalg.norm(
        interface_normals, axis=1, keepdims=True
    )
    interface_normals_direction = interface_normals / interface_normals_magnitude

    # Calculate the dot products
    dot_products = np.einsum(
        "ij,ij->i", circle_normals_direction, interface_normals_direction
    )

    # Check distance match
    distance_to_center = circle_normals_magnitude.flatten()
    distance_match = np.abs(distance_to_center - r) <= epsilon

    # Check normal alignment
    normal_match = np.abs(dot_products) >= (1 - normal_tolerance)

    # Final match condition
    matches = np.logical_not(distance_match & normal_match)

    # Remove matches with x > len(img) - tol_bath
    matches = np.logical_and(matches, interface_points[:, 0] < len(img) - tol_bath)

    # Remove true islands (imperfections in the profile at the circle)
    matches = remove_true_islands(matches, max_size=10)

    return matches


def preprocess_circle(
    folder_path: str,
    idx_to_analyze: np.ndarray[int],
    extension: str,
    angle: float,
    lines: np.ndarray,
):
    """
    Preprocess the circle data and save it to a file.

    Parameters:
    - folder_path: The path to the folder containing the images.
    - idx_to_analyze: The indices of the images to analyze.
    - extension: The extension of the images.
    - angle: The angle of the images.

    Returns:
    - None
    """

    # Load the images
    images = []
    for idx in idx_to_analyze:
        image = load_image(folder_path, idx, extension)
        image = rotate_and_crop_img(image, angle)
        images.append(image[lines[0] : lines[1]])
    images = np.array(images)

    # Find the circle centers
    circle_centers = []
    for image in images:
        xc, yc, r = find_circle(image)
        circle_centers.append([xc, yc, r])

    # Save the circle data
    circle_data = np.array(circle_centers)

    # np.savez_compressed("circle_data.npz", circle_data=circle_data)

    x = circle_data[:, 2] - np.mean(circle_data[:, 2]) + circle_data[:, 0]
    y = np.median(circle_data[:, 1])
    r = np.mean(circle_data[:, 2])
    stat = stats.linregress(idx_to_analyze, x)
    x = stat.slope * idx_to_analyze + stat.intercept

    return x, y, r


def compute_curvature(x, y):
    """
    Compute the curvature and curvature direction at each point of a 2D curve.

    Parameters:
    x (array-like): x-coordinates of the curve points.
    y (array-like): y-coordinates of the curve points.

    Returns:
    curvature (np.ndarray): Curvature at each point.
    curvature_direction (np.ndarray): Curvature direction as a unit vector at each point.
    """

    # Convert input to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # First derivatives using central difference
    dx = (np.roll(x, -1) - np.roll(x, 1)) / 2
    dy = (np.roll(y, -1) - np.roll(y, 1)) / 2

    # Second derivatives using central difference
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)

    # Compute curvature
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)

    # Compute the curvature direction as a unit normal vector
    norm = np.sqrt(dx**2 + dy**2)
    curvature_direction = np.column_stack((-dy / norm, dx / norm))

    return curvature, curvature_direction


def find_line(img: np.ndarray) -> float:

    # Define the Scharr operator for a gradients in the height axis
    scharr = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    grad = signal.convolve2d(img, scharr, boundary="symm", mode="same")
    xc = np.argmax(np.mean(grad, axis=1))
    return xc


def preprocess_line(
    folder_path: str,
    idx_to_analyze: np.ndarray[int],
    extension: str,
    angle: float,
    lines: np.ndarray,
):
    """
    Preprocess the circle data and save it to a file.

    Parameters:
    - folder_path: The path to the folder containing the images.
    - idx_to_analyze: The indices of the images to analyze.
    - extension: The extension of the images.
    - angle: The angle of the images.

    Returns:
    - None
    """

    # Load the images and find the line heights
    plate_height = []
    for idx in idx_to_analyze:
        image = load_image(folder_path, idx, extension)
        image = rotate_and_crop_img(image, angle)
        plate_height.append(find_line(image[lines[0] : lines[-1]]))

    return np.array(plate_height)
