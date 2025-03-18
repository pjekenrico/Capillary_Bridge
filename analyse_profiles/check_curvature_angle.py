import os
import matplotlib.pyplot as plt
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def set_nice_grid(ax):
    ax.grid(True)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.minorticks_on()


def compute_contact_angles(t, R, H, ds=0.02):

    contact_radia = np.array([c(t, np.array([1 - ds, 1])) for c in R])

    contact_heights = np.array([c(t, [1 - ds, 1]) for c in H])

    contact_angles = np.arctan(
        (contact_heights[:, :, 1] - contact_heights[:, :, 0])
        / (contact_radia[:, :, 1] - contact_radia[:, :, 0])
    )

    contact_angles = np.rad2deg(contact_angles)
    contact_angles = np.abs(contact_angles)
    contact_angles = [
        UnivariateSpline(t, angles, k=2, s=0) for angles in contact_angles
    ]

    return contact_angles

def numerical_curvature(R, H):
    
    # Compute the derivatives with numerical gradients
    dRdH = np.gradient(R) / np.gradient(H)
    ddRddH = np.gradient(dRdH) / np.gradient(H)
    
    curvature = ddRddH / (1 + dRdH ** 2) ** 1.5 - 1 / np.sqrt(1 + dRdH ** 2) / R
    
    return np.squeeze(curvature)

def smooth_RH(t, s, R, H, smoothing=0.2):
    data = (R[0](t, s) + R[1](t, s)) / 2
    R_smooth = RectBivariateSpline(t, s, data, s=smoothing)

    data = (H[0](t, s) + H[1](t, s)) / 2
    H_smooth = RectBivariateSpline(t, s, data, s=smoothing)
    return R_smooth, H_smooth


def main():

    path = os.path.join("series", "NWG10_V2500_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    T = np.linspace(np.min(bridge.times), np.max(bridge.times), 100)
    s = np.linspace(0, 1, 40)

    # second plot
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    R, H = smooth_RH(bridge.times, s, bridge.R, bridge.H, smoothing=None)

    plt.xlabel("Contact Radius [mm]")
    plt.ylabel("Contact Height [mm]")
    set_nice_grid(ax)

    # Normalize curvature
    norm = Normalize(vmin=-1, vmax=1)
    min_r = np.min([np.min(R(t, s)) for t in T])
    max_r = np.max([np.max(R(t, s)) for t in T])
    min_h = np.min([np.min(H(t, s)) for t in T])
    max_h = np.max([np.max(H(t, s)) for t in T])

    for t in T:
        # Extract radius, height, and curvature for current time
        r = np.squeeze(R(t, s))
        h = np.squeeze(H(t, s))
        c = numerical_curvature(r, h)

        # Define segments for LineCollection
        points = np.array([r, h]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection and add it
        lc = LineCollection(segments, cmap="viridis", norm=norm)
        lc.set_array(c)  # Assign curvature values for coloring
        lc.set_linewidth(2)
        ax.add_collection(lc)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Curvature [1/mm]")

    plt.xlim([min_r, max_r])
    plt.ylim([min_h, max_h])
    plt.tight_layout()

    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
