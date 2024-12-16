import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt


class Bridge(object):

    def __init__(
        self,
        profile: list[np.ndarray],
        times: np.ndarray,
        circle_positions: np.ndarray,
        line_positions: np.ndarray = None,
        fps: int = 5000,
        res: float = 0.046937338652898376,
        N: int = 100,
    ):
        self.times = (times - times[0]) / fps

        if line_positions is None:
            circle_positions = np.array(circle_positions).T
            self.xc, self.yc, self.r, apex = self.check_circle(*circle_positions)
        else:
            self.xc = line_positions
            x_max_left = np.max([np.max(profile[i][0][0]) for i in range(len(profile))])
            x_min_right = np.min([np.min(profile[i][1][0]) for i in range(len(profile))])
            
            self.yc = np.mean(x_max_left + x_min_right) / 2
            self.r = None
            apex = None

        re_interp = np.zeros((len(times), 2, 2, N))

        for i in range(len(times)):
            for d in range(2):
                re_interp[i, d] = self.refit_profile(*(profile[i][d].T), N)

        self.xc *= res
        self.yc *= res
        self.r *= res
        re_interp *= res
        apex *= res

        # Profiles, where self.R[0] and self.R[0] are the left and right radia as a function of time and position s and self.H[0] and self.H[1] are the heights
        self.R, self.H, apex, self.dR, self.dH = self.fit_R_of_ts(re_interp, apex)

        # Apex height as a function of time
        self.apex = UnivariateSpline(self.times, apex, k=2)

        self.contact_radius, self.contact_height, self.contact_angle = (
            self.compute_contact_data(ds=0.02)
        )

        self.neck_height, self.neck_radius = self.compute_neck_data()

        self.volume = self.compute_volume()

        self.curvature = self.compute_curvature()

        return

    def compute_contact_data(self):
        # Contact angles, s = 1
        contact_radius = [
            UnivariateSpline(
                self.times, np.squeeze(self.R[0](self.times, 1)), k=2, s=0
            ),
            UnivariateSpline(
                self.times, np.squeeze(self.R[1](self.times, 1)), k=2, s=0
            ),
        ]

        contact_height = [
            UnivariateSpline(
                self.times, np.squeeze(self.H[0](self.times, 1)), k=2, s=0
            ),
            UnivariateSpline(
                self.times, np.squeeze(self.H[1](self.times, 1)), k=2, s=0
            ),
        ]

        contact_angle = self.compute_contact_angles(ds=0.02)

        return contact_radius, contact_height, contact_angle

    def compute_neck_data(self, N=100):

        s = np.linspace(0, 1, N)

        neck_idx = [np.argmin(rad(self.times, s), axis=1) for rad in self.R]

        height = [
            UnivariateSpline(
                self.times, np.squeeze(rad(self.times, s[neck_idx[k]], grid=False)), s=0
            )
            for k, rad in enumerate(self.H)
        ]
        radius = [
            UnivariateSpline(
                self.times, np.squeeze(rad(self.times, s[neck_idx[k]], grid=False)), s=0
            )
            for k, rad in enumerate(self.R)
        ]
        return height, radius

    def compute_volume(self):
        volumes = np.zeros((2, len(self.times)))
        for i, t in enumerate(self.times):
            for j in range(2):

                # Base integral
                func = lambda s: np.pi * self.H[j](t, s, dy=1) * self.R[j](t, s) ** 2
                volumes[j, i] = quad(func, 0, 1)[0]

                if self.r is not None:
                # Negative part
                    neg_vol = (
                        np.pi
                        / 3
                        * (self.contact_height[j](t) - self.apex(t)) ** 2
                        * (3 * self.r - self.contact_height[j](t) + self.apex(t))
                    )
                    volumes[j, i] -= neg_vol

        volume_functions = [
            UnivariateSpline(self.times, vol, k=2, s=0) for vol in volumes
        ]

        return volume_functions

    def compute_contact_angles(self, ds=0.02):

        contact_radia = np.array([c(self.times, np.array([1 - ds, 1])) for c in self.R])

        contact_heights = np.array([c(self.times, [1 - ds, 1]) for c in self.H])

        contact_angles = np.arctan(
            (contact_heights[:, :, 1] - contact_heights[:, :, 0])
            / (contact_radia[:, :, 1] - contact_radia[:, :, 0])
        )

        contact_angles = np.rad2deg(contact_angles)
        contact_angles = np.abs(contact_angles)
        contact_angles = [
            UnivariateSpline(self.times, angles, k=2, s=0) for angles in contact_angles
        ]

        return contact_angles

    def plot_profiles_at_time(self, t, N=100):
        s = np.linspace(0, 1, N)

        plt.figure()
        plt.plot(
            np.squeeze(self.R[0](t, s)),
            np.squeeze(self.H[0](t, s)),
            label="Left profile",
        )
        plt.plot(
            np.squeeze(self.R[1](t, s)),
            np.squeeze(self.H[1](t, s)),
            label="Right profile",
        )
        plt.plot(0, self.apex(t), "o", label="Apex")
        plt.plot(self.neck_radius[0](t), self.neck_height[0](t), "o", label="Left neck")
        plt.plot(
            self.neck_radius[1](t), self.neck_height[1](t), "o", label="Right neck"
        )
        plt.xlabel("Radius [mm]")
        plt.ylabel("Height [mm]")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.show()

    def fit_R_of_ts(
        self, profiles: np.ndarray, apex: np.ndarray
    ) -> list[RectBivariateSpline, RectBivariateSpline]:

        s = np.linspace(0, 1, len(profiles[0, 0, 0]))
        R_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 1], kx=3, ky=3, s=0),
            RectBivariateSpline(
                self.times, s, profiles[:, 1, 1, ::-1], kx=3, ky=3, s=0
            ),
        ]

        h_min = np.max(profiles[:, :, 0])
        profiles[:, :, 0] -= h_min
        profiles[:, :, 0] *= -1
        apex -= h_min
        apex *= -1

        H_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 0], kx=3, ky=3, s=0),
            RectBivariateSpline(
                self.times, s, profiles[:, 1, 0, ::-1], kx=3, ky=3, s=0
            ),
        ]

        dR_dt_of_ts = [r.partial_derivative(dx=1, dy=0) for r in R_of_ts]
        dH_dt_of_ts = [h.partial_derivative(dx=1, dy=0) for h in H_of_ts]

        return R_of_ts, H_of_ts, apex, dR_dt_of_ts, dH_dt_of_ts

    def refit_profile(self, h, R, N=100):
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

    def compute_curvature(self):

        # Reinterpolate the derivatives to have a smoother curvature
        t = self.times
        s = np.linspace(0, 1, 100)

        dx = [
            RectBivariateSpline(t, s, self.R[0](t, s, dy=1)),
            RectBivariateSpline(t, s, self.R[1](t, s, dy=1)),
        ]

        ddx = [
            RectBivariateSpline(t, s, dx[0](t, s, dy=1)),
            RectBivariateSpline(t, s, dx[1](t, s, dy=1)),
        ]

        dy = [
            RectBivariateSpline(t, s, self.H[0](t, s, dy=1)),
            RectBivariateSpline(t, s, self.H[1](t, s, dy=1)),
        ]

        ddy = [
            RectBivariateSpline(t, s, dy[0](t, s, dy=1)),
            RectBivariateSpline(t, s, dy[1](t, s, dy=1)),
        ]

        curvature = []
        for i in range(2):
            curvature.append(
                lambda t, s: np.abs(
                    (dx[i](t, s) * ddy[i](t, s) - dy[i](t, s) * ddx[i](t, s))
                    / ((dx[i](t, s) ** 2 + dy[i](t, s) ** 2) ** 1.5)
                )
            )

        return curvature
