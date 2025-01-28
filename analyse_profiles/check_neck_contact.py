import os
import matplotlib.pyplot as plt
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge
from scipy.interpolate import UnivariateSpline


def set_nice_grid(ax):
    ax.grid(True)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.minorticks_on()


def main():

    path = os.path.join("series", "NWG10_V2500_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    t = bridge.times
    t = np.linspace(np.min(t), np.max(t), 100)
    _, ax = plt.subplots(2, 3, sharex=True, figsize=(12, 8))
    
    c1 = bridge.contact_radius[0]
    c2 = bridge.contact_radius[1]

    ax[0][0].set_ylabel("Contact Radius [mm]")
    set_nice_grid(ax[0][0])
    ax[0][0].set_xlim([np.min(t), np.max(t)])
    ax[0][0].plot(t, c1(t))
    ax[0][0].plot(t, c2(t))

    ax[0][1].set_ylabel("Smoothed Contact Radius [mm]")
    set_nice_grid(ax[0][1])
    d1 = UnivariateSpline(t, c1(t), s=0.05)
    d2 = UnivariateSpline(t, c2(t), s=0.25)
    ax[0][1].plot(t, d1(t))
    ax[0][1].plot(t, d2(t))
    
    ax[0][2].set_ylabel("Contact Velocity [mm/s]")
    set_nice_grid(ax[0][2])
    ax[0][2].plot(t, d1.derivative()(t))
    ax[0][2].plot(t, d2.derivative()(t))
    
    n1 = bridge.neck_radius[0]
    n2 = bridge.neck_radius[1]

    ax[1][0].set_ylabel("Neck Radius [mm]")
    ax[1][0].set_xlabel("Time")
    set_nice_grid(ax[1][0])
    ax[1][0].plot(t, n1(t))
    ax[1][0].plot(t, n2(t))
    
    dn1 = UnivariateSpline(t, n1(t), s=0.05)
    dn2 = UnivariateSpline(t, n2(t), s=0.05)

    ax[1][1].set_ylabel("Smoothed Neck Radius [mm]")
    ax[1][1].set_xlabel("Time")
    set_nice_grid(ax[1][1])
    ax[1][1].plot(t, dn1(t))
    ax[1][1].plot(t, dn2(t))
    
        
    ax[1][2].set_ylabel("Neck Velocity [mm/s]")
    set_nice_grid(ax[1][2])
    ax[1][2].plot(t, dn1.derivative()(t))
    ax[1][2].plot(t, dn2.derivative()(t))

    plt.tight_layout()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
