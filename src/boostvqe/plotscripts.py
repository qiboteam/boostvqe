import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

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
    error_bars=False,
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
        if len(dbi_energies[str(i)]) > 0:
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
            if error_bars:
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

        if error_bars:
            plt.fill_between(
                np.arange(start, len(loss_vqe[str(i)]) + start),
                loss_vqe[str(i)] - fluctuations_vqe[str(i)],
                loss_vqe[str(i)] + fluctuations_vqe[str(i)],
                color=BLUE,
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
    plt.title(
        rf'$N_{{\rm qubits}}={config["nqubits"]}, \, N_{{\rm layers}}={config["nlayers"]}, \, \mathrm{{{config["hamiltonian"]}}}$'
    )
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


def plot_loss_nruns(
    path,
    training_specs,
    title="",
    save=True,
    width=0.5,
):
    """
    Plot loss with confidence belt.
    """

    losses_dbi = []
    losses_vqe = []

    for i, f in enumerate(os.listdir(path)):
        this_path = path + "/" + f + "/"
        if i == 0:
            with open(this_path + OPTIMIZATION_FILE) as file:
                config = json.load(file)
            target_energy = config["true_ground_energy"]
        # accumulating dictionaries with results for each boost
        if training_specs in f:
            losses_vqe.append(dict(np.load(this_path + f"{LOSS_FILE + '.npz'}")))
            losses_dbi.append(dict(np.load(this_path + f"{DBI_ENERGIES + '.npz'}")))

    loss_vqe, dbi_energies, stds_vqe, stds_dbi = {}, {}, {}, {}

    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))
    plt.title(title)

    for i in range(config["nboost"]):
        this_vqe_losses, this_dbi_losses = [], []
        for d in range(len(loss_vqe)):
            this_vqe_losses.append(losses_vqe[d][str(i)])
            this_dbi_losses.append(losses_dbi[d][str(i)])

        loss_vqe.update({str(i): np.mean(np.asarray(this_vqe_losses), axis=0)})
        dbi_energies.update({str(i): np.mean(np.asarray(this_dbi_losses), axis=0)})
        stds_vqe.update({str(i): np.std(np.asarray(this_vqe_losses), axis=0)})
        stds_dbi.update({str(i): np.std(np.asarray(this_dbi_losses), axis=0)})

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
            loss_vqe[str(i)] - stds_vqe[str(i)],
            loss_vqe[str(i)] + stds_vqe[str(i)],
            color=BLUE,
            alpha=0.4,
        )
        plt.fill_between(
            np.arange(
                len(loss_vqe[str(i)]) + start - 1,
                len(dbi_energies[str(i)]) + len(loss_vqe[str(i)]) + start - 1,
            ),
            dbi_energies[str(i)] - stds_vqe[str(i)],
            dbi_energies[str(i)] + stds_dbi[str(i)],
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


def plot_lr_analysis(
    path,
    training_specs,
    zoom_coord,
    lr_list,
    title="",
    save=True,
    width=0.5,
):
    losses_vqe, losses_dbi, grads, fluct_dbi = [], [], [], []

    for i, lr in enumerate(lr_list):
        this_path = path + "/" + training_specs + f"_{lr}decay/"
        if i == 0:
            with open(this_path + OPTIMIZATION_FILE) as file:
                config = json.load(file)
            target_energy = config["true_ground_energy"]
            boost_frequency = config["boost_frequency"]
            dbi_steps = config["dbi_steps"]
        losses_vqe.append(dict(np.load(this_path + f"{LOSS_FILE + '.npz'}")))
        losses_dbi.append(dict(np.load(this_path + f"{DBI_ENERGIES + '.npz'}")))
        fluct_dbi.append(dict(np.load(this_path + f"{DBI_FLUCTUATIONS + '.npz'}")))
        grads.append(dict(np.load(this_path + f"{GRADS_FILE + '.npz'}")))

    colors = sns.color_palette("Spectral", n_colors=len(losses_vqe)).as_hex()

    _, ax = plt.subplots(figsize=(10 * width, 10 * width * 6 / 8))
    x1, x2, y1, y2 = zoom_coord[0], zoom_coord[1], zoom_coord[2], zoom_coord[3]
    axins = zoomed_inset_axes(
        ax,
        zoom=4,
        loc=1,
        bbox_to_anchor=(-0.015, -0.015, 1, 1),
        bbox_transform=ax.transAxes,
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    for b in range(config["nboost"] - 1):
        start = (b + 1) * boost_frequency + 1
        ax.plot(
            np.arange(start, start + dbi_steps + 1),
            losses_dbi[0][str(b)],
            color="black",
            lw=1.5,
            label="DBI",
        )
        axins.plot(
            np.arange(start, start + dbi_steps + 1),
            losses_dbi[0][str(b)],
            color="black",
            lw=1.5,
        )

    for i in range(len(losses_vqe)):
        for b in range(config["nboost"]):
            if b == 0:
                if i != 0:
                    continue
                else:
                    ax.plot(
                        np.arange(
                            b * boost_frequency + 1, (b + 1) * boost_frequency + 2
                        ),
                        losses_vqe[i][str(b)],
                        color=BLUE,
                        lw=1.5,
                    )
                    axins.plot(
                        np.arange(
                            b * boost_frequency + 1, (b + 1) * boost_frequency + 2
                        ),
                        losses_vqe[i][str(b)],
                        color=BLUE,
                        lw=1.5,
                    )
            else:
                ax.plot(
                    np.arange(
                        b * boost_frequency + dbi_steps * (b + 1) - 1,
                        (b + 1) * boost_frequency + dbi_steps * (b + 1),
                    ),
                    losses_vqe[i][str(b)],
                    color=colors[i],
                    label=rf"$f=${lr_list[i]}",
                    alpha=1,
                    lw=1.5,
                )
                axins.plot(
                    np.arange(
                        b * boost_frequency + dbi_steps * (b + 1) - 1,
                        (b + 1) * boost_frequency + dbi_steps * (b + 1),
                    ),
                    losses_vqe[i][str(b)],
                    color=colors[i],
                    alpha=1,
                    lw=1.5,
                )

    max_length = boost_frequency * len(losses_vqe[0])
    ax.hlines(
        target_energy,
        0,
        max_length,
        color="black",
        lw=1.5,
        label="Target energy",
        ls="--",
    )
    axins.hlines(
        target_energy,
        0,
        max_length,
        color="black",
        lw=1.5,
        label="Target energy",
        ls="--",
    )

    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    ax.legend(ncol=1, loc=2, framealpha=0.7)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Loss")

    if save:
        plt.savefig(f"lr_benchmark.pdf", bbox_inches="tight")

    plt.figure(figsize=(10 * width, 10 * width * 6 / 8))

    for i in range(len(grads)):
        ave_grads = []
        dbi_steps = config["dbi_steps"]
        iterations = []

        for epoch in range(len(grads[i])):
            len_iterations = len(iterations)
            iterations.extend(
                [
                    i + int(epoch) * (dbi_steps - 1) + len_iterations
                    for i in range(len(grads[i][str(epoch)]))
                ]
            )
            for grads_list in grads[i][str(epoch)]:
                ave_grads.append(np.mean(np.abs(grads_list)))

        plt.title(title)
        boost_x = 0
        for b in range(config["nboost"] - 1):
            boost_x += len(grads[i][str(b)])
            if i == 0:
                label = "DBI"
            else:
                label = None
            plt.plot(
                (
                    boost_x + b * (dbi_steps - 1) - 1,
                    boost_x + (b + 1) * (dbi_steps - 1),
                ),
                (ave_grads[boost_x - 1], ave_grads[boost_x]),
                color="black",
                label=label,
                lw=1.6,
                alpha=1,
            )
        for b in range(config["nboost"]):
            if b == 0:
                if i == 0:
                    plt.plot(
                        iterations[
                            b * boost_frequency + 1 : (b + 1) * boost_frequency + 2
                        ],
                        ave_grads[
                            b * boost_frequency + 1 : (b + 1) * boost_frequency + 2
                        ],
                        color=BLUE,
                        lw=1.5,
                    )
                else:
                    continue
            else:
                plt.plot(
                    iterations[b * boost_frequency + 1 : (b + 1) * boost_frequency + 2],
                    ave_grads[b * boost_frequency + 1 : (b + 1) * boost_frequency + 2],
                    color=colors[i],
                    lw=1.5,
                    label=rf"$f=${lr_list[i]}",
                )

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Gradients magnitude")
    plt.legend(loc=3)
    if save:
        plt.savefig(f"lr_grads_{title}.pdf", bbox_inches="tight")

