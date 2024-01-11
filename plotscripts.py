import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    FLUCTUATION_FILE,
    FLUCTUATION_FILE2,
    LOSS_FILE,
    LOSS_FILE2,
    OPTIMIZATION_FILE,
    PLOT_FILE,
    json_load,
)

RED = "#f54242"
YELLOW = "#edd51a"
GREEN = "#2db350"
PURPLE = "#587ADB"
BLUE = "#4287f5"


def plot_matrix(matrix, path, title="", save=True, width=0.5):
    """
    Visualize hamiltonian in a heatmap form.

    Args:
        matrix (np.ndarray): target matrix to be represented in heatmap form.
        title (str): figure title.
        save (bool): if ``True``, the figure is saved as `./plots/matrix_title.pdf`.
        width (float): ratio of the LaTeX manuscript which will be occupied by
            the figure. This argument is useful to standardize the image and font sizes.
    """
    fig, ax = plt.subplots(figsize=(10 * width, 10 * width))
    ax.set_title(title)
    try:
        im = ax.imshow(np.absolute(matrix), cmap="inferno")
    except TypeError:
        im = ax.imshow(np.absolute(matrix.get()), cmap="inferno")
    fig.colorbar(im, ax=ax)
    if save:
        plt.savefig(f"{path}/matrix_{title}.pdf", bbox_inches="tight")


def plot_loss(loss_history, path, title="", save=True, width=0.5):
    """
    Plot loss function history.

    Args:
        loss_history (list or np.ndarray): loss function history.
        title (str): figure title.
        save (bool): if ``True``, the figure is saved as `./plots/matrix_title.pdf`.
        width (float): ratio of the LaTeX manuscript which will be occupied by
            the figure. This argument is useful to standardize the image and font sizes.
    """
    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))
    plt.title(title)
    plt.plot(loss_history, lw=1.5, color=BLUE)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if save:
        plt.savefig(f"{path}/loss_{title}.pdf", bbox_inches="tight")


def plot_results(folder: pathlib.Path, energy_dbi: Optional[Tuple] = None):
    """Plots the energy and the energy fluctuations."""
    data = json_load(folder / OPTIMIZATION_FILE)
    energy_file = LOSS_FILE + ".npy"
    fluctuation_file = FLUCTUATION_FILE + ".npy"
    energy = np.load(folder / energy_file)
    print("FFFF", energy)
    errors = np.load(folder / fluctuation_file)
    epochs = range(len(energy))
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("VQE Training", fontsize=20)
    ax[0].plot(epochs, energy, color="navy", label="VQE training")
    ax[0].fill_between(
        epochs, energy - errors, energy + errors, color="royalblue", alpha=0.5
    )
    ax[0].axhline(
        y=data["true_ground_energy"], color="r", linestyle="-", label="True value"
    )
    if energy_dbi is not None:
        ax[0].axhline(y=energy_dbi[0], color="orange", linestyle="dashed", label="DBI")
        energy_file = LOSS_FILE2 + ".npy"
        fluctuation_file = FLUCTUATION_FILE2 + ".npy"

        energy2 = np.load(folder / energy_file)
        errors2 = np.load(folder / fluctuation_file)
        epochs2 = range(len(energy) - 1, len(energy) + len(energy2) - 1)
        ax[0].plot(epochs2, energy2, color="darkgreen", label="VQE training 2")
        ax[0].fill_between(
            epochs2, energy2 - errors2, energy2 + errors2, color="green", alpha=0.5
        )
        print(energy2)

    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Energy")
    ax[0].legend()
    ax[0].grid(True, which="major")
    ax[1].plot(epochs, np.abs(energy / data["true_ground_energy"]))
    if energy_dbi is not None:
        ax[1].axhline(
            y=energy_dbi[0] / data["true_ground_energy"],
            linestyle="dashed",
            color="orange",
            label="DBI",
        )

    ax[1].set_yscale("log")
    ax[1].axhline(y=1, color="r")
    ax[1].grid(True)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Energy ratio with true value")
    plt.savefig(folder / PLOT_FILE)
