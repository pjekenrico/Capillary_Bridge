import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
import matplotlib.pyplot as plt


class Bridge(object):

    def __init__(
        self,
        profile,
        times,
        circle_positions,
        fps=5000,
        res=0.046937338652898376,
        N=100,
    ):
        self.profile = profile
        self.times = (times - times[0]) / fps

        circle_positions = np.array(circle_positions).T
        self.xc, self.yc, self.r, self.apex = self.check_circle(*circle_positions)

        re_interp = np.zeros((len(times), 2, 2, N))

        for i in range(len(times)):
            for d in range(2):
                re_interp[i, d] = self.refit_profile(*(profile[i][d].T), N)

        # self.xc *= res
        # self.yc *= res
        # self.r *= res
        # re_interp *= res
        # self.apex *= res
        self.R, self.H = self.fit_R_of_ts(re_interp)

        return

    def fit_R_of_ts(
        self, profiles: np.ndarray
    ) -> list[RectBivariateSpline, RectBivariateSpline]:

        s = np.linspace(0, 1, len(profiles[0, 0, 0]))
        R_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 1], kx=3, ky=3, s=0),
            RectBivariateSpline(self.times, s, profiles[:, 1, 1], kx=3, ky=3, s=0),
        ]

        h_min = np.max(profiles[:, :, 0])
        profiles[:, :, 0] -= h_min
        profiles[:, :, 0] *= -1
        self.apex -= h_min
        self.apex *= -1

        H_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 0], kx=3, ky=3, s=0),
            RectBivariateSpline(self.times, s, profiles[:, 1, 0], kx=3, ky=3, s=0),
        ]

        return R_of_ts, H_of_ts

    def refit_profile(self, h, R, N):
        s = np.linspace(0, 1, len(h))
        xs = UnivariateSpline(s, h, k=3, s=0)
        ys = UnivariateSpline(s, np.abs(R - self.yc), k=3, s=0)
        return xs(np.linspace(0, 1, N)), ys(np.linspace(0, 1, N))

    def check_circle(self, xc, yc, r) -> list[np.ndarray, float, float, np.ndarray]:

        y = np.mean(yc)
        sigma_y = np.std(yc)
        if sigma_y > 1:
            print(
                "Warning: sigma_y > 1. This may indicate a problem with the circle positions. Or that the angle of the image in the segmentation is not well chosen. In practical terms it means that the ball makes a diagonal movement and does not descend in a centered way."
            )
        else:
            print(
                f"Checking circle horizontal position... OK\nHorizontal position: {y} +- {sigma_y} pxl"
            )

        apex = xc + r
        r = np.mean(r)
        sigma_r = np.std(r)
        if sigma_r > 1:
            print(
                "Warning: sigma_r > 1. This may indicate issues with the circle recognition. Proceed with caution."
            )
        else:
            print(
                f"Checking circle radius... OK\nEstimated radius: {r} +- {sigma_r} pxl"
            )

        return xc, y, r, apex
