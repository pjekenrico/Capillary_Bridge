import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path of the 'main' directory
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add 'main' to the Python path
sys.path.append(main_dir)

from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def smooth_RH(t, s, R, H, smoothing=0.2):
    data = (R[0](t, s) + R[1](t, s)) / 2
    R_smooth = RectBivariateSpline(t, s, data, s=smoothing)

    data = (H[0](t, s) + H[1](t, s)) / 2
    H_smooth = RectBivariateSpline(t, s, data, s=smoothing)
    return R_smooth, H_smooth


def main():
    path = os.path.join("series", "NWG50_V50_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    s = np.linspace(0, 1, 40)
    t = bridge.times
    
    bridge.plot_profiles_at_time(t[-1])
    
    # t = np.linspace(np.min(t), np.max(t), 100)

    # # Smooth the radius and height functions
    # R_smooth, H_smooth = smooth_RH(bridge.times, s, bridge.R, bridge.H, smoothing=0)

    # # Extract smoothed radius and height at t0
    # r = np.squeeze(R_smooth(t[-1], s))
    # h = np.squeeze(H_smooth(t[-1], s))
    
    # # Circle position at t0
    # xc = bridge.xc(t[-1])
    # yc = bridge.yc
    # rc = 81
    # r_s = bridge.r
    # apex = bridge.apex(t[-1])
    # apex_xc = apex + r_s
    

    # # Plot the smoothed profile
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(r, h, color="black", linewidth=2)  # Simple black line
    
    # # Plot the circle
    # circle = plt.Circle((0, xc), r_s, color="red", fill=False)
    # ax.add_artist(circle)
    # ax.plot(0, xc, 'bo')
    # ax.plot(0, apex, 'ro')
    # ax.plot(0, apex_xc, 'go')
    

    # plt.xlabel("Contact Radius [mm]")
    # plt.ylabel("Contact Height [mm]")

    # # plt.xlim([np.min(r), np.max(r)])
    # # plt.ylim([np.min(h), np.max(h)])
    # plt.tight_layout()

    # plt.show()
    # print("Done!")

if __name__ == "__main__":
    main()

