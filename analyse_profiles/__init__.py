import argparse, os
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

    bridge.plot_profiles_at_time(0.2)


if __name__ == "__main__":
    main()
