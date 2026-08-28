import argparse

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DAT_FILE = "/Users/gabrielkern/Documents/OpenFOAM/SupersonicUAV/postProcessing/forceCoeffs1/0/coefficient.dat"


def parse_header(path) -> list[str]:
    """Grab the last '#' comment line in an OpenFOAM .dat file - that's the column header."""
    header_line = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header_line = line
            else:
                break
    return header_line.lstrip("#").split()


def plot_coefficients(path) -> None:
    columns = parse_header(path)
    data = np.loadtxt(path, comments="#")

    time = data[:, 0]
    for col_name, col_data in zip(columns[1:], data[:, 1:].T):
        plt.plot(time, col_data, label=col_name)

    plt.xlabel("Time [s]")
    plt.ylabel("Coefficient value")
    plt.title(f"Force/moment coefficient convergence\n{path}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot OpenFOAM forceCoeffs .dat output over time")
    parser.add_argument("dat_file", nargs="?", default=DEFAULT_DAT_FILE)
    args = parser.parse_args()

    plot_coefficients(args.dat_file)
