import json
import os.path
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from qibo.backends import GlobalBackend

from utils import (
    DBI_ENERGIES,
    DBI_FLUCTUATIONS,
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

LINE_STYLES = ["--", "-", "-.", ":"]


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


def plot_loss(
    path,
    title="",
    save=True,
    width=0.5,
):
    """
    Plot loss with confidence belt.
    """
    fluctuations_vqe = dict(np.load(path / f"{FLUCTUATION_FILE + '.npz'}"))
    loss_vqe = dict(np.load(path / f"{LOSS_FILE + '.npz'}"))
    # loss_history = np.load(path / LOSS_FILE)
    config = json.loads((path / OPTIMIZATION_FILE).read_text())
    target_energy = config["true_ground_energy"]
    dbi_energies = dict(np.load(path / f"{DBI_ENERGIES + '.npz'}"))
    dbi_fluctuations = dict(np.load(path / f"{DBI_FLUCTUATIONS + '.npz'}"))
    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))
    plt.title(title)

    for i in range(config["nboost"]):
        start = (
            0
            if str(i - 1) not in loss_vqe
            else sum(
                len(loss_vqe[str(j)]) + len(dbi_energies[str(j)])
                for j in range(config["nboost"])
                if j < i
            )
            - 2 * i
        )
        plt.plot(
            np.arange(start, len(loss_vqe[str(i)]) + start),
            loss_vqe[str(i)],
            color=BLUE,
            lw=1.5,
            label="VQE",
        )
        plt.plot(
            np.arange(
                len(loss_vqe[str(i)]) + start - 1,
                len(dbi_energies[str(i)]) + len(loss_vqe[str(i)]) + start - 1,
            ),
            dbi_energies[str(i)],
            color=GREEN,
            lw=1.5,
            label="DBI",
        )
        plt.fill_between(
            np.arange(start, len(loss_vqe[str(i)]) + start),
            loss_vqe[str(i)] - fluctuations_vqe[str(i)],
            loss_vqe[str(i)] + fluctuations_vqe[str(i)],
            color=BLUE,
            alpha=0.4,
        )
        plt.fill_between(
            np.arange(
                len(loss_vqe[str(i)]) + start - 1,
                len(dbi_energies[str(i)]) + len(loss_vqe[str(i)]) + start - 1,
            ),
            dbi_energies[str(i)] - dbi_fluctuations[str(i)],
            dbi_energies[str(i)] + dbi_fluctuations[str(i)],
            color=GREEN,
            alpha=0.4,
        )

    max_length = (
        sum(len(l) for l in list(dbi_energies.values()))
        + sum(len(l) for l in list(loss_vqe.values()))
        - 2 * config["nboost"]
    )
    plt.hlines(target_energy, 1, max_length, color="red", lw=1, label="Target energy")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save:
        plt.savefig(f"{path}/loss_{title}.pdf", bbox_inches="tight")


def plot_results(folder: pathlib.Path):
    """Plots the energy and the energy fluctuations."""
    data = json_load(folder / OPTIMIZATION_FILE)
    # energy_file = LOSS_FILE + ".npy"
    # fluctuation_file = FLUCTUATION_FILE + ".npy"
    energy = np.load(folder / energy_file)
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
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Energy")
    ax[0].legend()
    ax[0].grid(True, which="major")
    ax[1].plot(epochs, np.abs(energy / data["true_ground_energy"]))

    if os.path.isfile(folder / DBI_RESULTS):
        energy_dbi = json_load(folder / DBI_RESULTS)
        ax[0].axhline(
            y=GlobalBackend().to_numpy(energy_dbi["energy"]),
            color="orange",
            linestyle="dashed",
            label="DBI",
        )
        energy_file = LOSS_FILE2 + ".npy"
        fluctuation_file = FLUCTUATION_FILE2 + ".npy"

        energy2 = np.load(folder / energy_file)
        errors2 = np.load(folder / fluctuation_file)
        epochs2 = range(len(energy) - 1, len(energy) + len(energy2) - 1)
        ax[0].plot(epochs2, energy2, color="darkgreen", label="VQE training 2")
        ax[0].fill_between(
            epochs2, energy2 - errors2, energy2 + errors2, color="green", alpha=0.5
        )

        ax[1].axhline(
            y=GlobalBackend().to_numpy(
                energy_dbi["energy"] / data["true_ground_energy"]
            ),
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
