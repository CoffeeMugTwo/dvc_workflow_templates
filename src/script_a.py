import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True
    )
    args = parser.parse_args()

    data = np.arange(0, 2*np.pi, 0.01)
    df = pd.DataFrame({"x": data})
    df.to_csv(
        path_or_buf=args.output,
        sep=";",
        index=False,
        header=True
    )
    return


if __name__ == "__main__":
    main()
