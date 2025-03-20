import os 
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" if using Qt5

# Get the absolute path of the 'main' directory
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add 'main' to the Python path
sys.path.append(main_dir)

from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge
from scipy.interpolate import RectBivariateSpline
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import mplcursors

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

labels = {
    "viscous_pressure": "Viscous Pressure [Pa]",
    "inertia_pressure": "Inertia Pressure [Pa]",
    "gravity_pressure": "Gravity Pressure [Pa]",
    "laplace_pressure": "Laplace Pressure [Pa]",
    "elastic_pressure": "Elastic Pressure [Pa]",
    "total_pressure": "Total Pressure [Pa]",
    "local_reynolds": "Reynolds Number [-]",
    "velocity_z": "Axial Velocity [mm/s]",
    "dr": "Radial Velocity [mm/s]",
    "residence_time": "Residence Time [s]"
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

def numerical_curvature(R, H, s):
    dRdH = np.gradient(R) / np.gradient(H)
    ddRddH = np.gradient(dRdH) / np.gradient(H)
    curvature = ddRddH / (1 + dRdH ** 2) ** 1.5 - 1 / np.sqrt(1 + dRdH ** 2) / R
    return np.squeeze(curvature)

def smooth_RH(t, s, R, H, smoothing=0):
    data_R = (R[0](t, s) + R[1](t, s)) / 2
    data_H = (H[0](t, s) + H[1](t, s)) / 2
    R_smooth = RectBivariateSpline(t, s, data_R, s=smoothing)
    H_smooth = RectBivariateSpline(t, s, data_H, s=smoothing)
    return R_smooth, H_smooth

def smooth_dR(t, s, dR, smoothing=0):
    data = (dR[0](t, s) + dR[1](t, s)) / 2 
    dR_smooth = RectBivariateSpline(t, s, data, s=smoothing)
    return dR_smooth

def smooth_dH(t, s, dH, smoothing=0):
    data = (dH[0](t, s) + dH[1](t, s)) / 2
    dH_smooth = RectBivariateSpline(t, s, data, s=smoothing)
    return dH_smooth

def velocity_z(t, s, R, H, dR):
    dz = np.gradient(H, s)
    integrand = 2 * R * dR * dz
    integral = np.cumsum(integrand)
    velocity = integral / (R**2)
    velocity[np.abs(velocity) < 1e-6] = 1e-6
    return velocity

def laplace_pressure(surface_tension, R, H, s):
    curvature = numerical_curvature(R, H, s)
    return surface_tension * curvature * 1000

def inertia_pressure(density, t, s, R, H, dR):
    velocity = velocity_z(t, s, R, H, dR)
    return 0.5 * density * (velocity * 0.001) ** 2

def gravity_pressure(density, g, H):
    return density * g * (H * 0.001)

def elastic_pressure(surface_tension, R, H, s):
    lateral_surface = 2 * np.pi * R * H
    dz = H / len(s)
    integral = sum(2 * np.pi * R * H * dz)
    return surface_tension * lateral_surface / integral * 1000

def viscous_pressure(viscosity, R, H, dR):
    return 6 * viscosity * dR / R

def local_reynolds(density, viscosity, t, s, R, H, dR):
    velocity = velocity_z(t, s, R, H, dR)
    characteristic_length = R
    return (density * np.abs(velocity) * 0.001 * characteristic_length * 0.001) / viscosity

def residence_time(t, s, R, H, dR):
    velocity = velocity_z(t, s, R, H, dR)
    dz = np.gradient(H, s)
    residence = np.zeros_like(s)
    
    # Integrate from top (s=1) to bottom (s=0)
    for i in range(len(s)-2, -1, -1):
        if np.abs(velocity[i]) > 1e-6:
            residence[i] = residence[i+1] + dz[i] / np.abs(velocity[i])
        else:
            residence[i] = residence[i+1]
    residence = np.flip(residence)
    # residence = np.cumsum(residence)
    # normalise residence time
    # residence = residence / np.max(residence)
    
    return residence


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

    T_1 = np.linspace(np.max(bridge.times)-0.03, np.max(bridge.times), 80)
    T_2 = np.linspace(np.min(bridge.times), np.max(bridge.times)-0.03, 650)
    T = np.concatenate((T_1, T_2))
    # T = bridge.times
    s = np.linspace(0, 1, 20)

    R, H = smooth_RH(bridge.times, s, bridge.R, bridge.H, smoothing=None)
    dR = smooth_dR(bridge.times, s, bridge.dR, smoothing=None)
    dH = smooth_dH(bridge.times, s, bridge.dH, smoothing=None)

    cases = [
        {
            'name': 'viscous_pressure',
            'func': lambda t, s, r, h, dr, dh: viscous_pressure(viscosity, r, h, dr),
            'label': labels['viscous_pressure'],
            'title': "Evolution of Viscous Pressure in the Liquid Bridge  "
        },
        {
            'name': 'inertia_pressure',
            'func': lambda t, s, r, h, dr, dh: inertia_pressure(density, t, s, r, h, dr),
            'label': labels['inertia_pressure'],
            'title': "Evolution of Inertial Pressure in the Liquid Bridge  "
        },
        {
            'name': 'gravity_pressure',
            'func': lambda t, s, r, h, dr, dh: gravity_pressure(density, g, h),
            'label': labels['gravity_pressure'],
            'title': "Evolution of Hydrostatic Pressure in the Liquid Bridge  "
        },
        {
            'name': 'laplace_pressure',
            'func': lambda t, s, r, h, dr, dh: laplace_pressure(surface_tension, r, h, s),
            'label': labels['laplace_pressure'],
            'title': "Evolution of Laplace Pressure in the Liquid Bridge  "
        },
        {
            'name': 'elastic_pressure',
            'func': lambda t, s, r, h, dr, dh: elastic_pressure(surface_tension, r, h, s),
            'label': labels['elastic_pressure'],
            'title': "Evolution of Elastic Pressure in the Liquid Bridge  "
        },
        {
            'name': 'total_pressure',
            'func': lambda t, s, r, h, dr, dh: (-laplace_pressure(surface_tension, r, h, s) 
                                           + gravity_pressure(density, g, h) 
                                           + inertia_pressure(density, t, s, r, h, dr)),
            'label': labels['total_pressure'],
            'title': "Evolution of Total Pressure in the Liquid Bridge  "
        },
        {
            'name': 'local_reynolds',
            'func': lambda t, s, r, h, dr, dh: local_reynolds(density, viscosity, t, s, r, h, dr),
            'label': labels['local_reynolds'],
            'title': "Evolution of Local Reynolds Number in the Liquid Bridge  "
        },
        {
            'name': 'velocity_z',
            'func': lambda t, s, r, h, dr, dh: velocity_z(t, s, r, h, dr),
            'label': labels['velocity_z'],
            'title': "Evolution of Axial Velocity in the Liquid Bridge  "
        },
        {
            'name': 'dr',
            'func': lambda t, s, r, h, dr, dh: dr,
            'label': labels['dr'],
            'title': "Evolution of Radial Velocity in the Liquid Bridge  "
        },
        {
            'name': 'residence_time',
            'func': lambda t, s, r, h, dr, dh: residence_time(t, s, r, h, dr),
            'label': labels['residence_time'],
            'title': "Evolution of Residence Time in the Liquid Bridge  "
        }
    ]

    figures = []

    for case in cases:
        all_c = []
        for t in T:
            r = np.squeeze(R(t, s))
            h = np.squeeze(H(t, s))
            dr_val = np.squeeze(dR(t, s))
            dh_val = np.squeeze(dH(t, s))
            c = case['func'](t, s, r, h, dr_val, dh_val)
            all_c.append(c)
        all_c_flat = np.concatenate(all_c)
        c_min = np.min(all_c_flat)
        c_max = np.max(all_c_flat)

        fig = plt.figure(figsize=(10, 7), dpi=200)
        figures.append((fig, case['name']))  # Store figures to save later
        ax = plt.gca()
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.set_xlabel("Contact Radius [mm]")
        ax.set_ylabel("Contact Height [mm]")
        ax.set_title(f"{case['title']} ({filename.replace('_profiles.npz', '')
})")

        norm = Normalize(vmin=c_min, vmax=c_max)
        cmap = cm.get_cmap("viridis")

        for t in T:
            r = np.squeeze(R(t, s))
            h = np.squeeze(H(t, s))
            dr_val = np.squeeze(dR(t, s))
            dh_val = np.squeeze(dH(t, s))
            c = case['func'](t, s, r, h, dr_val, dh_val)

            points = np.array([r, h]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            colors = cmap(norm(c[:-1]))
            colors[:, 3] = 0.8

            lc = LineCollection(segments, colors=colors, norm=norm)
            lc.set_array(c)
            lc.set_linewidth(2)
            ax.add_collection(lc)

            if t == T_1[0] or t == T_2[0]:
                time_remaining = np.max(bridge.times) - t
                ax.text(r[0]+1, h[0]-0.2, f"{time_remaining:.3f} s", 
                        fontsize=12, color="black", ha="right", va="bottom")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label(case['label'])

        plt.xlim([-1, 20])
        plt.ylim([-0.5, 6])
        plt.tight_layout()
        
        cursor = mplcursors.cursor(hover=True)
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target
            min_dist = float('inf')
            closest_value = None
            for lc in ax.collections:
                if isinstance(lc, LineCollection):
                    segments = lc.get_segments()
                    colors = lc.get_array()
                    for i, segment in enumerate(segments):
                        dist = np.min(np.linalg.norm(segment - np.array([x, y]), axis=1))
                        if dist < min_dist:
                            min_dist = dist
                            closest_value = colors[i]
            if closest_value is not None:
                sel.annotation.set_text(f"x={x:.2f}, y={y:.2f}\nValue: {closest_value:.2f}")
            else:
                sel.annotation.set_text(f"x={x:.2f}, y={y:.2f}\nNo data")

    # Show all figures at once
    # plt.show()

    # Save figures **after** viewing them
    for fig, name in figures:
        # save in a folder called "figures"
        os.makedirs("figures", exist_ok=True)
        fig.savefig(f"figures/{filename}_{name}.png")
        plt.close(fig)
    print("Done!")
  

if __name__ == "__main__":
    main()
    
