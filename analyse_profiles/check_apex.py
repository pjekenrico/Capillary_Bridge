import os
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge

import matplotlib.pyplot as plt

def main():

    path = os.path.join("series", "NWG10_V2500_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    T = np.linspace(np.min(bridge.times), np.max(bridge.times), 100)
    s = np.linspace(0, 1, 40)

    # bridge.plot_profiles_at_time(T[0])
    
    vol = 0.5*(bridge.volume[0](T) + bridge.volume[1](T))
    plt.plot(T, vol)

    plt.show()
    
if __name__ == "__main__":
    main()
