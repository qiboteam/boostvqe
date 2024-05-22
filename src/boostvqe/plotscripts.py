import os
import json

import matplotlib.pyplot as plt
import numpy as np

from boostvqe.utils import (
    DBI_ENERGIES,
    DBI_FLUCTUATIONS,
    FLUCTUATION_FILE,
    GRADS_FILE,
    LOSS_FILE,
    OPTIMIZATION_FILE,
)

RED = "#F05F51"
YELLOW = "#edd51a"
GREEN = "#2db350"
PURPLE = "#587ADB"
BLUE = "#5D51F0"

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
            color=RED,
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
            color=RED,
            alpha=0.4,
        )

    max_length = (
        sum(len(l) for l in list(dbi_energies.values()))
        + sum(len(l) for l in list(loss_vqe.values()))
        - 2 * config["nboost"]
        + 1
    )
    plt.hlines(
        target_energy,
        0,
        max_length,
        color="black",
        lw=1,
        label="Target energy",
        ls="-",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save:
        plt.savefig(f"{path}/loss_{title}.pdf", bbox_inches="tight")


def plot_gradients(
    path,
    title="",
    save=True,
    width=0.5,
):
    """
    Plot gradients magnitude during the training.
    Each value is the average over the parameters of the absolute value of the
    derivative of the loss function with respect to the parameter.
    """
    grads = dict(np.load(path / f"{GRADS_FILE + '.npz'}"))
    config = json.loads((path / OPTIMIZATION_FILE).read_text())
    ave_grads = []
    dbi_steps = config["dbi_steps"]
    iterations = []
    for epoch in grads:
        len_iterations = len(iterations)
        iterations.extend(
            [
                i + int(epoch) * (dbi_steps - 1) + len_iterations
                for i in range(len(grads[epoch]))
            ]
        )
        for grads_list in grads[epoch]:
            ave_grads.append(np.mean(np.abs(grads_list)))
    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))
    plt.title(title)
    plt.plot(
        iterations,
        ave_grads,
        color=BLUE,
        lw=1.5,
        label=r"$\langle |\partial_{\theta_i}\text{L}| \rangle_i$",
    )

    boost_x = 0
    for b in range(config["nboost"] - 1):
        boost_x += len(grads[str(b)])
        label = None
        if b == 0:
            label = "Step after DBI"
        plt.plot(
            (boost_x + b * (dbi_steps - 1) - 1, boost_x + (b + 1) * (dbi_steps - 1)),
            (ave_grads[boost_x - 1], ave_grads[boost_x]),
            color=RED,
            lw=1.6,
            alpha=1,
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Gradients magnitude")
    plt.legend()
    if save:
        plt.savefig(f"{path}/grads_{title}.pdf", bbox_inches="tight")


def plot_nruns_result(
    path,
    training_specs,
    title="",
    save=True,
    width=0.5,
):
    
    losses = []

    for f in os.listdir(path):
        if training_specs in f:
            losses.append(np.load(path + "/" + f))
    
    losses = np.array(losses)
    means = np.mean(losses, axis=0)
    stds = np.std(losses, axis=0)

    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))
    plt.plot(means, color=BLUE)
    plt.fill_between(means - stds, means + stds, alpha=0.3, color=BLUE)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\langle L \rangle$")
    if save:
        plt.savefig("runs.png")