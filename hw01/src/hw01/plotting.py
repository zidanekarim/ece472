import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import LinearModel, NNXLinearModel, NNXRegressionModel, NNXBasisExpansion

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)
plt.rcParams["svg.fonttype"] = "none"


def compare_linear_models(a: LinearModel, b: LinearModel):
    """Prints a comparison of two linear models."""
    log.info("Comparing models", true=a, estimated=b)
    print("w,    w_hat")
    for w_a, w_b in zip(a.weights, b.weights):
        print(f"{w_a:0.2f}, {w_b:0.2f}")

    print(f"{a.bias:0.2f}, {b.bias:0.2f}")


def plot_fit(
    model: NNXRegressionModel,  # changed to Regression model here
    data: Data,
    settings: PlottingSettings,
):
    """Plots the linear fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Regression Fit")
    ax.set_xlabel("x")
    ax.set_ylim(np.amax(data.y) * -1.5, np.amax(data.y) * 1.5)  # added negative y-lim
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 100)
    xs = xs[:, np.newaxis]

    true_sin_y = np.sin(2 * np.pi * xs)
    # true sin plot
    ax.plot(
        xs, true_sin_y, color="red", linestyle="--", linewidth=1.5, label="True Sin"
    )

    # noisy sin plot (same as true, but with sigma noise added)
    ax.plot(
        np.squeeze(data.x),
        data.y,
        "o",
        color="green",
        markeredgecolor="black",
        label="Noisy Sin",
    )

    # regression plot
    ax.plot(
        xs,
        np.squeeze(model(jnp.asarray(xs[:, np.newaxis]))),
        "-",
        color="blue",
        label="Regression",
    )

    plt.tight_layout()
    plt.legend()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = settings.output_dir / "fit.svg"
    pdf_path = settings.output_dir / "fit.pdf"
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))


def plot_basis_functions(
    model: NNXBasisExpansion,  # probably? also i had to ask someone what plotting a basis expansion meant and I learned it was just phi's
    settings: PlottingSettings,
):
    print("Learned mus:\n", np.sort(np.array(model.mu.value)))
    print("Learned (log) sigmas:\n", np.array(model.sig.value))
    """Plots M Gaussian Basis Functions and saves it to a file."""
    log.info("Plotting bases")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Bases for fit")
    ax.set_xlabel("x")
    ax.set_ylim(0, 1.0)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(-2, 2, 1000)
    xs = xs[:, np.newaxis]
    ax.plot(xs, np.squeeze(model(jnp.asarray(xs))), "-")

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = settings.output_dir / "bases.svg"
    pdf_path = settings.output_dir / "fit.pdf"
    plt.savefig(svg_path)
    # plt.savefig(pdf_path)
    plt.close(fig)
    log.info("Saved plot", path=str(svg_path))
