import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator, interp1d
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

# ======== PATH CONFIGURATION ========
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)
series_dir = os.path.join(main_dir, 'series')

from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge

# ======== PHYSICAL PARAMETERS ========
W_density = 1000
WG10_density = 1023
WG50_density = 1129
g = 9.81
W_surface_tension = 0.072
WG10_surface_tension = 0.071
WG50_surface_tension = 0.067
W_viscosity = 1e-3
WG10_viscosity = 1.5e-3
WG50_viscosity = 6e-3

# ======== PLOT CONFIGURATION ========
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

def set_nice_grid(ax):
    ax.grid(True)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.minorticks_on()

def capillary_length(sigma, rho, g):
    return np.sqrt(sigma / (rho * g)) * 1e3

def time_gamma(sigma, rho, g):
    return (sigma / (rho * g**3)) ** (1/4)

def main():
    datasets = {
        "NWG50_V50_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 50 \mu m/s$",
        "NWG50_V500_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 500 \mu m/s$",
        "NWG50_V1000_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 1000 \mu m/s$",
        "NWG50_V2500_profiles.npz": "$\eta_{Gly 50\%} = 6\eta_{W}$ \, \, \,$v = 2500 \mu m/s$",
        "NWG10_V50_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 50 \mu m/s$",
        "NWG10_V500_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 500 \mu m/s$",
        "NWG10_V1000_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 1000 \mu m/s$",
        "NWG10_V2500_profiles.npz": "$\eta_{Gly 10\%} = 1.5\eta_{W}$ \, $v = 2500 \mu m/s$",
        "NW_V50_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 50 \mu m/s$",
        "NW_V500_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 500 \mu m/s$",
        "NW_V1000_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 1000 \mu m/s$",
        "NW_V2500_profiles.npz": "$\eta_{W} = 1 mPa.s$ \, \, \, $v = 2500 \mu m/s$",
    }
    
    # ======== COLOR PALETTES ========
    colors_WG50 = ["#90a9ff", "#6681fa", "#3c6ad4", "#1253ad"]
    colors_WG10 = ["#ff7fa3", "#f2517c", "#bc5090", "#8a508f"]
    colors_W = ["#ffe5b3", "#ffd380", "#ffa600", "#ff8531"]

    # ======== REFERENCE DATA STORAGE ========
    reference_data = {}

    # ======== PLOT INITIALIZATION ========
    _, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4))
    plt.subplots_adjust(right=0.5)

    # ======== PROCESS NPZ DATASETS ========
    for i, (filename, label) in enumerate(datasets.items()):
        # Load NPZ data
        data_path = os.path.join(series_dir, filename)
        data = Dataframe(data_path=data_path)
        
        # Process bridge data
        bridge = Bridge(
            data.contact_lines,
            data.indices,
            data.circle_positions,
            bath_height=data.metadata["bath_height"] - data.metadata["lines"][0],
            break_index=data.metadata["break_index"]
        )
        
        # Calculate parameters
        sigma = WG50_surface_tension if "WG50" in filename else WG10_surface_tension if "WG10" in filename else W_surface_tension
        rho = WG50_density if "WG50" in filename else WG10_density if "WG10" in filename else W_density
        l_c = capillary_length(sigma, rho, g)
        t_gamma = time_gamma(sigma, rho, g)
        
        # Store reference data
        material = "WG50" if "WG50" in filename else "WG10" if "WG10" in filename else "W"
        velocity = int(''.join(filter(str.isdigit, filename.split('_V')[1].split('_')[0])))
        reference_data[(material, velocity)]   = {
            'l_c': l_c,
            't_gamma': t_gamma,
            'color': [colors_WG50, colors_WG10, colors_W][i//4][i%4]
        }
        
        # Time
        t = np.linspace(np.min(bridge.times), np.max(bridge.times), 1000)
              
        # Raw data
        theta1_raw = bridge.contact_angle[0](t)
        theta2_raw = bridge.contact_angle[1](t)
        
        # Mean raw data
        theta_raw = 0.5 * (theta1_raw + theta2_raw)
        
        # Time normalization
        time = (t[-1] - t) / t_gamma  # Time normalization
        
        # # Velocity data
        # contact_line_velocity1 = bridge.dR[0](t,1)
        # contact_line_velocity2 = bridge.dR[1](t,1)
        # # Flatten velocities data or reshape to match theta data
        # contact_line_velocity1_arr = np.array([contact_line_velocity1])
        # contact_line_velocity2_arr = np.array([contact_line_velocity2])
        # # Flatten 
        # contact_line_velocity1 = contact_line_velocity1_arr.flatten()
        # contact_line_velocity2 = contact_line_velocity2_arr.flatten()
        # # Mean velocity
        # contact_line_velocity = np.abs(0.5 * (contact_line_velocity1 + contact_line_velocity2))
        
        # Plot original data and smoothed data
        ax[0].plot(time, theta1_raw, label=label, color=reference_data[(material, velocity)]['color'])
        ax[1].plot(time, theta2_raw, label=label, color=reference_data[(material, velocity)]['color'])
        ax[0].set(xlabel=r"$(t_c - t)/t_{\gamma}$", 
          ylabel=r"$\theta [\deg]$",
          xscale='log')
        ax[1].set(xlabel=r"$(t_c - t)/t_{\gamma}$", 
          ylabel=r"$\theta [\deg]$",
            xscale='log')
        ax[1].legend(loc='upper right', bbox_to_anchor=(1.5, 1), ncol=1)
        set_nice_grid(ax[0])
        set_nice_grid(ax[1])
        
    # ======== MODEL SLOPES ========
    # velocity = theta ** (1/3)
    # x = np.linspace(0, 1, 1000)
    # y = x ** (1/3)
    # ax[0].plot(x, y * 180 / np.pi, linestyle='--', color='black')
    # ax[1].plot(x, y * 180 / np.pi, linestyle='--', color='black')
    # ax[0].text(0.5, 30, r"$\theta \propto v^{1/3}$", fontsize=12)
    # ax[1].text(0.5, 30, r"$\theta \propto v^{1/3}$", fontsize=12)
    
    # ======== PLOT FORMATTING ========    
    plt.tight_layout()
    plt.show()

    print("Done!")

if __name__ == "__main__":
    main()
