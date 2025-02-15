import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

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

def remove_sharp_changes(data, times, factor=1):
    """ Removes data points where the rate of change is too high. """
    dt = np.diff(times)
    d_angle = np.diff(data) / dt  # Compute derivative

    threshold = factor * np.std(d_angle)  # Set threshold based on standard deviation
    valid_indices = np.where(np.abs(d_angle) < threshold)[0]  # Keep points below threshold
    valid_indices = np.append(valid_indices, valid_indices[-1] + 1)  # Include last point

    # Interpolate over the removed points
    interp_func = UnivariateSpline(times[valid_indices], data[valid_indices], k=2, s=0)
    return interp_func(times)

def remove_outliers(data, times, factor=10):
    """ Removes outliers from data. """
    mean = np.mean(data)
    std = np.std(data)

    # Keep only points within factor standard deviations from the mean
    valid_indices = np.where(np.abs(data - mean) < factor * std)[0]

    # Interpolate over the removed points
    interp_func = UnivariateSpline(times[valid_indices], data[valid_indices], k=2, s=0)
    return interp_func(times)

def fit_two_polynomes_concat(data, times, degree1=1, degree2=2):
    """ Fit two polynomes to the data. """
    n = len(data)
    p1 = np.polyfit(times[:n//2], data[:n//2], degree1)
    p2 = np.polyfit(times[n//2:], data[n//2:], degree2)
    return np.concatenate([np.polyval(p1, times[:n//2]), np.polyval(p2, times[n//2:])])

def fit_polynome(data, times, degree=3):
    """ Fit a polynome to the data. """
    p = np.polyfit(times, data, degree)
    return np.polyval(p, times)

def compute_contact_angles(bridge):
    """ Compute and filter contact angles from bridge data. """
    t = bridge.times
    theta1_raw = bridge.contact_angle[0](t)
    theta2_raw = bridge.contact_angle[1](t)
    
    # Remove outliers
    theta1_filtered = remove_outliers(theta1_raw, t)
    theta2_filtered = remove_outliers(theta2_raw, t)

    # Remove sharp peaks and dips
    # theta1_filtered = remove_sharp_changes(theta1_raw, t)
    # theta2_filtered = remove_sharp_changes(theta2_raw, t)
    
    # Fit polynome to data
    # theta1_filtered = fit_polynome(theta1_filtered, t)
    # theta2_filtered = fit_polynome(theta2_filtered, t)
    
    # Fit two polynomes to data
    # theta1_filtered = fit_two_polynomes_concat(theta1_filtered, t)
    # theta2_filtered = fit_two_polynomes_concat(theta2_filtered, t)

    # Apply smoothing after filtering
    theta1_smooth = UnivariateSpline(t, theta1_filtered, s=1)
    theta2_smooth = UnivariateSpline(t, theta2_filtered, s=1)

    return t, theta1_smooth, theta2_smooth

def main():
    path = os.path.join("series", "SHW_V50_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    t, theta1_smooth, theta2_smooth = compute_contact_angles(bridge)

    _, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4))

    # Plot original data
    ax[0].set_ylabel("Contact Angle [deg]")
    set_nice_grid(ax[0])
    ax[0].set_xlim([np.min(t), np.max(t)])
    ax[0].plot(t, bridge.contact_angle[0](t), label="Raw Theta 1", alpha=0.5)
    ax[0].plot(t, bridge.contact_angle[1](t), label="Raw Theta 2", alpha=0.5)
    ax[0].legend()

    # Plot smoothed data
    ax[1].set_ylabel("Smoothed Contact Angle [deg]")
    set_nice_grid(ax[1])
    mean_theta = (theta1_smooth(t) + theta2_smooth(t)) / 2  # Compute mean contact angle
    
    ax[1].plot(t, theta1_smooth(t), label="Smoothed Theta 1")
    ax[1].plot(t, theta2_smooth(t), label="Smoothed Theta 2")
    ax[1].plot(t, mean_theta, label="Mean Theta", linestyle="--", color="black")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    print("Done!")

if __name__ == "__main__":
    main()
