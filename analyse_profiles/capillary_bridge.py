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
    
    Bridge(data.contact_lines, data.indices, data.circle_positions)
    print("DONE")


if __name__ == "__main__":
    main()
