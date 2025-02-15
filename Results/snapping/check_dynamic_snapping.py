import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import matplotlib.colors as mcolors

# Enable LaTeX rendering in plots
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# Get the absolute path of the 'main' directory
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge

def set_nice_grid(ax):
    ax.grid(True)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.minorticks_on()

def extrapolate_to_zero(time, radius):
    """ Perform linear regression on the last few points and extrapolate to R = 0. """
    num_points = 3  # Use last 3 points for extrapolation
    slope, intercept, _, _, _ = linregress(time[-num_points:], radius[-num_points:])

    if slope == 0:
        return None, None

    t_zero = -intercept / slope
    t = np.linspace(time[-1], t_zero, 100)
    r = slope * t + intercept

    return t, r

def main():
    datasets = {
        "SHWG50_V500_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 500 \mu m/s$",
        "SHWG50_V1000_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 1000 \mu m/s$",
        "SHWG50_V2500_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 2500 \mu m/s$",
        "SHWG10_V500_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 500 \mu m/s$",
        "SHWG10_V1000_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 1000 \mu m/s$",
        "SHWG10_V2500_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 2500 \mu m/s$",
        "SHW_V100_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 100 \mu m/s$",
        "SHW_V1000_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 1000 \mu m/s$",
        "SHW_V2500_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 2500 \mu m/s$",
    }
    colors_WG50 = ["#6681fa", "#3c6ad4", "#1253ad"]
    colors_WG10 = ["#f2517c", "#bc5090", "#8a508f"]
    colors_W = ["#ffd380", "#ffa600", "#ff8531"]
    colors = colors_WG50 + colors_WG10 +colors_W 
    markers = ["o", "d", "*", "o", "d", "*", "o", "d", "*"]
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    plt.subplots_adjust(right=0.5) 

    csv_filename = "contact_radius_data.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "tc-t", "R/R0"])

        for i, (filename, velocity_label) in enumerate(datasets.items()):
            path = os.path.join("series", filename)
            data = Dataframe(data_path=path)
            bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

            t = np.linspace(np.min(bridge.times), np.max(bridge.times), 1000)
            c1, c2 = bridge.contact_radius[0](t), bridge.contact_radius[1](t)
            mean_c = (c1 + c2) / 2
            extended_time, extended_radius = extrapolate_to_zero(t, mean_c)
            t = np.concatenate((t, extended_time))
            mean_c = np.concatenate((mean_c, extended_radius))
            smooth_c = UnivariateSpline(t, mean_c, s=0.1)
            smooth_c_values = smooth_c(t) / smooth_c(t[0])
            time = t[-1] - t + 1e-10

            for j in range(len(time)):
                writer.writerow([filename, time[j], smooth_c_values[j]])

            color_rgb = np.array(mcolors.to_rgb(colors[i]))
            lightened_color = 0.3 * color_rgb + 0.7 * np.array([1, 1, 1])
            ax.plot(time, smooth_c_values, color='gray', alpha=0.2, linewidth=1.5, linestyle="--")
            ax.scatter(time[::25], smooth_c_values[::25], s=35, marker=markers[i], 
                       edgecolors=colors[i], facecolors=[lightened_color], alpha=0.8, 
                       label=f"{velocity_label}")
    
    # ax.plot(time, (time) ** (1/ 3), "--", color="black", alpha=0.8, label=r"$t^{1/3}$")
    # ax.plot(time, (time) ** (1 / 5), "-", color="black", alpha=0.8, label=r"$t^{2/5}$")
    
    # # Reference slopes
    # x_slope = np.array([1e-5, 10])  # Pick appropriate x-range
    # y_slope_23 = 3.6 * (x_slope ** (2 / 3))  # Scaled for visibility
    # # y_slope_35 = 1.6 * (x_slope ** (3 / 5))  # Scaled for visibility
    # y_slope_25 = 0.95 * (x_slope ** (1 / 5))  # Scaled for visibility
    # y_sope_1_20 =55 * (x_slope ** (1))  # Scaled for visibility
    # # y_sope_1_2 = 10 * x_slope  # Scaled for visibility
    # # y_sope_1_3 = 7 * (x_slope ** 1)  # Scaled for visibility

    # ax.plot(x_slope, y_slope_23, "-", color="black", alpha=0.8, label=r"$t^{2/3}$")
    # # ax.plot(x_slope, y_slope_35, "-.", color="black", alpha=0.8, label=r"$t^{3/5}$")
    # ax.plot(x_slope, y_slope_25, "--", color="black", alpha=0.8, label=r"$t^{1/5}$")
    # ax.plot(x_slope, y_sope_1_20, ":", color="black", alpha=0.8, label=r"$t$")
    # # ax.plot(x_slope, y_sope_1_2, ":", color="black", alpha=0.8)
    # # ax.plot(x_slope, y_sope_1_3, ":", color="black", alpha=0.8)
    
    ax.set_ylabel(r"$R^* = R/R_0$")
    ax.set_xlabel(r"$t_c - t$")
    set_nice_grid(ax)
    # log 
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 3)
    ax.set_ylim(1e-4, 3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.suptitle(r"\textbf{Contact Radius}", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Data saved to {csv_filename}")
    print("Done!")

if __name__ == "__main__":
    main()
