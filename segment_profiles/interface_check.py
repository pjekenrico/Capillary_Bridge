import numpy as np
from scipy import signal
from scipy import optimize


def optimize_circle_position(
    pts, xc, yc, r, initial_lr=0.001, num_iterations=1000, decay=0.99
):

    # Distance error function (vectorized)
    def dist_err(x_ctr):
        distances = np.sqrt((pts[0] - x_ctr) ** 2 + (pts[1] - yc) ** 2)
        return np.sum((distances - r) ** 2)

    # Gradient of the distance error function (vectorized)
    def grad_dist_err(x_ctr):
        distances = np.sqrt((pts[0] - x_ctr) ** 2 + (pts[1] - yc) ** 2)
        nonzero_distances = np.where(distances != 0, distances, np.inf)
        gradient = 2 * np.sum((distances - r) * (x_ctr - pts[0]) / nonzero_distances)
        return gradient

    x_ctr = xc
    learning_rate = initial_lr

    for i in range(num_iterations):
        grad = grad_dist_err(x_ctr)  # Calculate gradient w.r.t. x_ctr
        x_ctr = x_ctr - learning_rate * grad  # Update x_ctr

        # Decay learning rate every iteration
        learning_rate *= decay

        # Optionally, break if gradient change is very small
        if np.abs(learning_rate * grad) < 1e-6:
            print(f"Converged after {i+1} iterations.")
            break

        print(
            f"Iteration {i+1}, x_ctr: {x_ctr}, Learning Rate: {learning_rate:.6f}, Gradient: {grad:.6f}"
        )

    return x_ctr


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

    # r = 1880
    # xc = optimize_circle_position(idx, xc, yc, r)

    return xc, yc, r


def match_line_with_circle_and_normal(
    interface_points, img, epsilon=2, normal_tolerance=0.1
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

    xc, yc, r = find_circle(img)
    circle_center = np.array([xc, yc])

    # Convert inputs to numpy arrays
    interface_points = np.array(interface_points)

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

    return matches
