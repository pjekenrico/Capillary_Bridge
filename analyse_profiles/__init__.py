import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge


def main():

    path = os.path.join("npz_files", "WG10TLV50_profiles.npz")

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    t = bridge.times
    plt.figure()
    R = bridge.contact_radius[0](t)
    H = bridge.contact_height[0](t)
    plt.plot(R,H)
    
    R = bridge.contact_radius[1](t)
    H = bridge.contact_height[1](t)
    plt.plot(R, H)
    
    path = os.path.join("npz_files", "segmentation_options_profiles.npz")
    data = Dataframe(data_path=path)
    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)

    t = bridge.times
    # plot the contact radius as a function of the contact height for all times
    R = bridge.contact_radius[0](t)
    H = bridge.contact_height[0](t)
    plt.plot(R,H)
    
    R = bridge.contact_radius[1](t)
    H = bridge.contact_height[1](t)
    plt.plot(R, H)
    
    plt.ylim(1,6)

    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
