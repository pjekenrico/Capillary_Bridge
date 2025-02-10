import os
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge

def main():

    path = os.path.join("series", "NWG10_V500_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    T = np.linspace(np.min(bridge.times), np.max(bridge.times), 100)
    s = np.linspace(0, 1, 40)

    bridge.plot_profiles_at_time(T[0])

if __name__ == "__main__":
    main()
