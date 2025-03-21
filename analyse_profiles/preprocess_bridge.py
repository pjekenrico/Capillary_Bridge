import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d, PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


def rolling_average(data, window_size):
    """
    Compute the rolling average of a 1D array.

    Parameters:
        data (array-like): The input 1D array.
        window_size (int): The size of the moving window.

    Returns:
        np.ndarray: The rolling average array. The result has NaN for edges where the window doesn't fully fit.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if window_size > len(data):
        raise ValueError("Window size must not exceed the length of the data.")

    # Use np.convolve to calculate the moving average
    kernel = np.ones(window_size) / window_size

    # pad edges with boundary
    data = np.pad(data, (window_size // 2, window_size // 2), mode="edge")

    return np.convolve(data, kernel, mode="valid")

class Bridge(object):

    def __init__(
        self,
        profile: list[np.ndarray],
        times: np.ndarray,
        circle_positions: np.ndarray,
        line_positions: np.ndarray = None,
        bath_height: float = None,
        break_index: int = None,
        fps: int = 5000,
        res: float = 0.046937338652898376, 
        N: int = 100,
    ):
        self.times = (times - times[0]) / fps
        self.bath_height = bath_height
        self.break_time = (break_index - times[0]) / fps

        if line_positions is None:
            circle_positions = np.array(circle_positions).T
            self.xc, self.yc, self.r, self.apex = self.check_circle(*circle_positions)
        else:
            self.xc = line_positions
            x_max_left = np.max([np.max(profile[i][0][0]) for i in range(len(profile))])
            x_min_right = np.min(
                [np.min(profile[i][1][0]) for i in range(len(profile))]
            )

            self.yc = np.mean(x_max_left + x_min_right) / 2
            self.r = None
            self.apex = None

        re_interp = np.zeros((len(times), 2, 2, N))

        for i in range(len(times)):
            for d in range(2):
                re_interp[i, d] = self.refit_profile(*(profile[i][d].T), N)

        self.xc *= res
        self.yc *= res
        self.r *= res
        re_interp *= res
        self.apex *= res
        self.bath_height *= res

        # Profiles, where self.R[0] and self.R[0] are the left and right radia as a function of time and position s and self.H[0] and self.H[1] are the heights
        self.R, self.H, self.dR, self.dH = self.fit_R_of_ts(re_interp)

        # Apex and xc height as a function of time
        self.apex = UnivariateSpline(self.times, self.apex, k=1)
        self.xc = UnivariateSpline(self.times, self.xc, k=1)

        self.contact_radius, self.contact_height, self.contact_angle = (
            self.compute_contact_data()
        )

        self.neck_height, self.neck_radius = self.compute_neck_data()

        self.volume = self.compute_volume()

        self.curvature = self.compute_curvature()

        return

    def compute_contact_data(self):

        cont_rad_left = np.squeeze(self.R[0](self.times, 1))
        cont_rad_left[cont_rad_left < cont_rad_left[-1]] = cont_rad_left[-1]

        cont_rad_right = np.squeeze(self.R[1](self.times, 1))
        cont_rad_right[cont_rad_right < cont_rad_right[-1]] = cont_rad_right[-1]

        cont_height_left = np.squeeze(self.H[0](self.times, 1))
        cont_height_left[cont_height_left < cont_height_left[-1]] = cont_height_left[-1]

        cont_height_right = np.squeeze(self.H[1](self.times, 1))
        cont_height_right[cont_height_right < cont_height_right[-1]] = (
            cont_height_right[-1]
        )

        contact_radius = [
            UnivariateSpline(self.times, cont_rad_left, k=2, s=0),
            UnivariateSpline(self.times, cont_rad_right, k=2, s=0),
        ]

        contact_height = [
            UnivariateSpline(self.times, cont_height_left, k=2, s=0),
            UnivariateSpline(self.times, cont_height_right, k=2, s=0),
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
        
        def remove_outliers_robust(data, times, factor=0):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                mad = np.mean(np.abs(data - median))
            threshold = factor * mad
            valid_indices = np.where(np.abs(data - median) <= threshold)[0]
            if len(valid_indices) < 5:
                return data
            return interp1d(times[valid_indices], data[valid_indices], 
                            kind='linear', fill_value='extrapolate')(times)
        
        def remove_sharp_changes_robust(data, times, factor=0):
            dt = np.diff(times)
            if np.any(dt <= 0):
                raise ValueError("Times must be strictly increasing.")
            d_data = np.diff(data) / dt
            median_deriv = np.median(d_data)
            mad_deriv = np.median(np.abs(d_data - median_deriv))
            if mad_deriv == 0:
                mad_deriv = np.mean(np.abs(d_data - median_deriv))
            threshold = factor * mad_deriv
            valid_deriv_indices = np.where(np.abs(d_data - median_deriv) <= threshold)[0]
            valid_mask = np.zeros_like(data, dtype=bool)
            for i in valid_deriv_indices:
                valid_mask[i] = True
                valid_mask[i+1] = True
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 10:
                return data
            return interp1d(times[valid_indices], data[valid_indices], 
                            kind='linear', fill_value='extrapolate')(times)
        
        def apply_adaptive_smoothing(data, window_ratio=0.6):
            min_window = 5
            window_length = max(min_window, int(len(data) * window_ratio))
            if window_length % 2 == 0:
                window_length += 1
            polyorder = min(3, window_length - 1)
            try:
                return savgol_filter(data, window_length, polyorder)
            except:
                return data
        
        def very_smooth(data):
            return np.polyval(np.polyfit(np.arange(len(data)), data, 3), np.arange(len(data)))
        
        def enforce_monotonic_trend(times, data):
            ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
            return ir.fit_transform(times.reshape(-1, 1), data)
        
        processed_contact_angles = []
        for angles in contact_angles:
            cleaned = remove_outliers_robust(angles, self.times)
            cleaned = remove_sharp_changes_robust(cleaned, self.times)
            smoothed = apply_adaptive_smoothing(cleaned)
            over_smoothed = very_smooth(smoothed)
            monotonic = enforce_monotonic_trend(self.times, over_smoothed)
            interpolator = PchipInterpolator(self.times, monotonic)
            processed_contact_angles.append(interpolator)
        
        return processed_contact_angles

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
        plt.xlim(0, np.max([np.max(self.R[0](t, s)), np.max(self.R[1](t, s))]))
        plt.axis("equal")
        ax = plt.gca()
        ax.add_patch(
            Circle(
                (0, self.xc(t)),
                self.r,
                facecolor="none",
                edgecolor=(0, 0.8, 0.8),
                linewidth=1,
            )
        )
        plt.grid()
        plt.legend()
        plt.show()

    def fit_R_of_ts(
        self, profiles: np.ndarray
    ) -> list[RectBivariateSpline, RectBivariateSpline]:

        s = np.linspace(0, 1, len(profiles[0, 0, 0]))
        R_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 1], kx=2, ky=2, s=0),
            RectBivariateSpline(
                self.times, s, profiles[:, 1, 1, ::-1], kx=2, ky=2, s=0
            ),
        ]

        h_min = self.bath_height
        # h_min = np.max(profiles[:, :, 0])

        profiles[:, :, 0] -= h_min
        profiles[:, :, 0] *= -1
        self.apex -= h_min
        self.apex *= -1
        self.xc -= h_min
        self.xc *= -1

        H_of_ts = [
            RectBivariateSpline(self.times, s, profiles[:, 0, 0], kx=2, ky=3, s=0),
            RectBivariateSpline(
                self.times, s, profiles[:, 1, 0, ::-1], kx=2, ky=3, s=0
            ),
        ]

        dR_dt_of_ts = [r.partial_derivative(dx=1, dy=0) for r in R_of_ts]
        dH_dt_of_ts = [h.partial_derivative(dx=1, dy=0) for h in H_of_ts]

        return R_of_ts, H_of_ts, dR_dt_of_ts, dH_dt_of_ts

    def refit_profile(self, h, R, N=30):
        s = np.linspace(0, 1, len(h))
        print(len(h))
        xs = UnivariateSpline(s, h, k=2, s=0)
        if np.mean(R) < self.yc:
            ys = UnivariateSpline(s,  self.yc-R, k=2, s=0)
        else:
            ys = UnivariateSpline(s, R - self.yc, k=2, s=0)
            
        return xs(np.linspace(0, 1, N)), ys(np.linspace(0, 1, N))

    def check_circle(self, xc:np.ndarray, yc:np.ndarray, r:np.ndarray) -> list[np.ndarray, float, float, np.ndarray]:
        y = np.mean(yc)
        xc = xc.astype(float)
        apex = xc + r
        r = np.mean(r)
        return xc, y, r, apex

    def compute_curvature(self, N=40):
        s = np.linspace(0, 1, N)
        t = np.array(self.times)
        
        # Evaluate interpolators over the grid of t and s
        R_left = self.R[0](t, s)
        R_right = self.R[1](t, s)
        data_R = (R_left + R_right) / 2
        
        H_left = self.H[0](t, s)
        H_right = self.H[1](t, s)
        data_H = (H_left + H_right) / 2
        
        smoothing = 0.1
        # Smooth the averaged data
        R_smooth = RectBivariateSpline(t, s, data_R, s=smoothing)
        H_smooth = RectBivariateSpline(t, s, data_H, s=smoothing)
        
        # Evaluate the smoothed functions on the original grid
        R = R_smooth(t, s)
        H = H_smooth(t, s)
        
        # Compute derivatives with respect to s (axis=1)
        dRds = np.gradient(R, s, axis=1)
        dHds = np.gradient(H, s, axis=1)
        
        dRdH = dRds / dHds
        
        ddRddH = np.gradient(dRdH, s, axis=1) / dHds
        
        curvature = ddRddH / (1 + dRdH**2)**1.5 - 1/(np.sqrt(1 + dRdH**2) * R)
        
        return np.squeeze(curvature)
