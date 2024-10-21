import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from segment_profiles.lvlset_segmentation import Dataframe
from analyse_profiles.preprocess_bridge import Bridge


def main():
    parser = argparse.ArgumentParser(
        description="Program to segment the profiles of a capilary bridge."
    )

    parser.add_argument(
        "path",
        nargs="?",  # Makes the argument optional
        default=None,  # Default to None when the argument is not provided
        help="Path to an existing valid json file. If not provided, the program will generate one such file at the current directory and ask for user input.",
    )

    args = parser.parse_args()

    # Get a list of npz files in the current directory
    npz_files = [f for f in os.listdir() if f.endswith(".npz")]

    path = args.path or os.path.join(os.getcwd(), npz_files[0])

    # Reads and extracts data from npz
    data = Dataframe(data_path=path)

    bridge = Bridge(data.contact_lines, data.indices, data.circle_positions)
    # The important attributes of the bridge object are:
    # - bridge.times: Times of the profiles
    # - bridge.R: Radii of the profiles (as a list of callable functions R(t,s))
    # - bridge.H: Heights of the profiles (as a list of callable functions H(t,s))
    # - bridge.apex: Apex height as a function of time
    # - bridge.contact_radius: Contact radius as a function of time
    # - bridge.contact_height: Contact height as a function of time
    # - bridge.contact_angle: Contact angle as a function of time
    # - bridge.neck_height: Neck height as a function of time
    # - bridge.neck_radius: Neck radius as a function of time
    # - bridge.volume: Volume of the bridge as a function of time

    # bridge.plot_profiles_at_time(bridge.times[6])

    s = np.linspace(0, 1, 100)
    t = np.linspace(bridge.times[0], bridge.times[-1], 100)

    fig, ax = plt.subplots(5, figsize=(10, 10))

    ax[0].plot(t, bridge.volume[0](t), label="Volume")
    ax[0].plot(t, bridge.volume[1](t), label="Volume")
    ax[1].plot(t, bridge.contact_radius[0](t), label="Contact radius")
    ax[1].plot(t, bridge.contact_radius[1](t), label="Contact radius")
    # ax[2].plot(t, np.squeeze(bridge.dR[0](t, 1)), label="Contact radius velocity")
    # ax[2].plot(t, np.squeeze(bridge.dR[1](t, 1)), label="Contact radius velocity")
    ax[2].plot(
        t,
        np.squeeze(bridge.contact_radius[0].derivative(1)(t)),
        label="Contact radius velocity",
    )
    ax[2].plot(
        t,
        np.squeeze(bridge.contact_radius[1].derivative(1)(t)),
        label="Contact radius velocity",
    )
    ax[3].plot(t, bridge.contact_height[0](t), label="Contact Height")
    ax[3].plot(t, bridge.contact_height[1](t), label="Contact Height")
    ax[3].set_ylim(0, 6)
    ax[4].plot(t, bridge.contact_angle[0](t), label="Contact Angle")
    ax[4].plot(t, bridge.contact_angle[1](t), label="Contact Angle")
    ax[4].set_ylim(0, 80)
    plt.tight_layout()
    plt.legend()

    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
