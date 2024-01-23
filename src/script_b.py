import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True
    )
    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--test",
        action="store_true"
    )
    args = parser.parse_args()

    df = pd.read_csv(
        filepath_or_buffer=args.input,
        sep=";",
        index_col=None,
        header=0
    )

    if args.test:
        df = df[:10]

    df["y"] = np.sin(df["x"])

    df.to_csv(
        path_or_buf=args.output,
        sep=";",
        index=False,
        header=True
    )

if __name__ == "__main__":
    main()
