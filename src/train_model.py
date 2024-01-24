import argparse


from dvclive import Live
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


MAX_DEGREE = 10


def train_model(x, y, degree):
    fit_params, residuals, _, _, _ = np.polyfit(x, y, degree, full=True)
    return fit_params, residuals


def make_plot(x_test, y_test, fit_params, degree):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        x_test,
        y_test,
        marker="o",
        linewidth=0,
        color="C0",
        label="Test data"
    )

    p = np.poly1d(fit_params)
    ax.plot(
        x_test,
        p(x_test),
        marker="x",
        linewidth=0,
        color="C1",
        label=f"Fit degree: {degree}"
    )
    ax.legend()
    return fig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to training data",
        required=True
    )

    parser.add_argument(
        "--output",
        help="Output path",
        required=True
    )

    args = parser.parse_args()

    training_data = pd.read_csv(
        args.input,
        sep=";",
        header=0,
        index_col=None
    )

    x_train, x_test, y_train, y_test = train_test_split(training_data["x"], training_data["y"])

    model_path = f"{args.output}/model.txt"
    plot_path = f"{args.output}/plot.pdf"

    with Live() as live:

        live.log_param("max_degree", MAX_DEGREE)

        for degree in range(1, MAX_DEGREE):

            live.log_param("degree", degree)
            
            # Train model
            model, residuals = train_model(x_train, y_train, degree)
            np.savetxt(model_path, model)

            # Evaluate model
            plot = make_plot(x_test, y_test, model, degree)
            plot.savefig(plot_path)

            # Log metrics
            print(residuals)
            live.log_metric("residual", residuals[0])

            live.next_step()

            live.log_artifact(
                path=model_path,
                type="model",
                name=f"poly_d{degree}"
            )

            live.log_artifact(
                path=plot_path,
                type="plot",
                name=f"plot_d{degree}"
            )

if __name__ == "__main__":
    main()
