import numpy as np
from scipy.ndimage import laplace, gaussian_filter


def drlse_edge(
    phi_0, g, lmda, mu, alfa, epsilon, timestep, iters
):  # Updated Level Set Function
    """

    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iters: number of iterations
    """
    phi = phi_0.copy()
    [vy, vx] = np.gradient(g)
    for k in range(iters):
        # phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)
        dist_reg_term = dist_reg_p2(phi)
        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g  # balloon/pressure force
        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature
        phi += timestep * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)
    return phi


def dist_reg_p2(phi):
    """
    compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = s > 1
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (
        s - 1
    )  # compute first order derivative of the double-well potential p2 in equation (16)
    dps = ((ps != 0) * ps + (ps == 0)) / (
        (s != 0) * s + (s == 0)
    )  # compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode="nearest")


def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (0.5 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def neumann_bound_cond(f):
    """
    Make a function satisfy Neumann boundary condition
    """
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g


def find_lsf(
    img: np.ndarray,
    initial_lsf: np.ndarray,
    timestep=1,
    iter_inner=10,
    iter_outer=30,
    lmda=5,
    alfa=-3,
    epsilon=1.5,
    sigma=0.8,
    **kwargs,
):
    """
    :param img: Input image as a grey scale uint8 array (0-255)
    :param initial_lsf: Array as same size as the img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run drlse before showing the output
    :param iter_outer: How many iterations to run the iter_inner
    :param lmda: coefficient of the weighted length term L(phi)
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernal
    """
    if len(img.shape) != 2:
        raise Exception("Input image should be a gray scale one")

    if len(img.shape) != len(initial_lsf.shape):
        raise Exception("Input image and the initial LSF should be in the same shape")

    if np.max(img) <= 1:
        raise Exception("Please make sure the image data is in the range [0, 255]")

    # parameters
    mu = 0.1 / timestep  # coefficient of the distance regularization term R(phi)

    img = np.array(img, dtype="float32")
    img_smooth = gaussian_filter(img, sigma)  # smooth image by Gaussian convolution
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1 + f)  # edge indicator function.

    # initialize LSF as binary step function
    phi = initial_lsf.copy()

    # start level set evolution
    for n in range(iter_outer):
        phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner)
        yield phi

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine)
    yield phi
