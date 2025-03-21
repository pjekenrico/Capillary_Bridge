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

# Enable LaTeX rendering in plots
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# Constants
g = 9.81  # m/s^2

# Dictionary mapping filenames to their corresponding properties
fluid_properties = {
    "WG50": {"density": 1129, "surface_tension": 0.067, "viscosity": 6e-3},
    "WG10": {"density": 1023, "surface_tension": 0.071, "viscosity": 1.5e-3},
    "W":    {"density": 1000, "surface_tension": 0.071, "viscosity": 1e-3}
}

# Function to extract fluid properties from filename
def get_fluid_properties(filename):
    if "WG50" in filename:
        return fluid_properties["WG50"]
    elif "WG10" in filename:
        return fluid_properties["WG10"]
    else:
        return fluid_properties["W"]

def set_nice_grid(ax):
    ax.grid(True)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax.minorticks_on()

def main():
    filename = "TLWG50_V1000_profiles.npz"
    path = os.path.join("series", filename)

    properties = get_fluid_properties(filename)
    density = properties["density"]
    surface_tension = properties["surface_tension"]
    viscosity = properties["viscosity"] 

    data = Dataframe(data_path=path) 
    bridge = Bridge(
        data.contact_lines,
        data.indices,
        data.circle_positions,
        bath_height=data.metadata["bath_height"] - data.metadata["lines"][0],
        break_index=data.metadata["break_index"]
    )
    s = np.linspace(0, 1, 40)
    t = bridge.times
    
    bridge.plot_profiles_at_time(t[-1])
    plt.show()

if __name__ == "__main__":
    main()

