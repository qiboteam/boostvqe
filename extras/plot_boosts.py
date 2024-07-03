import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from qibo import hamiltonians, set_backend

from boostvqe.ansatze import build_circuit

set_backend("numpy")


def calculate_vqe_gates(nqubits, nlayers):
    """Calculate total number of CZ gates of a given VQE circuit."""
    c = build_circuit(nqubits, nlayers)
    ncz = len(c.gates_of_type("cz"))
    return ncz


def load_nfval_given_opt(boostpath, gci_steps, optimizer):
    """Load the number of function evaluations given optimizer and results."""
    filepath = f"{boostpath}/boosting_data.json"
    nfvals = []
    with open(filepath) as file:
        res_dict = json.load(file)
    for s in range(gci_steps):
        if optimizer == "sgd":
            raise ValueError("nfvals calculation not supported for SGD")
        elif optimizer == "cma":
            nfvals.append(res_dict[str(s + 1)]["cma_extras"]["evaluations"])
        else:
            nfvals.append(res_dict[str(s + 1)][f"{optimizer}_extras"]["nfev"])
    return nfvals


def load_all_results_given_path(path):
    """Load all dictionaries given path to a collection of VQE trainings."""
    # we collect results here
    results = []
    for vqedir in os.listdir(path):
        vqepath = os.path.join(path, vqedir)
        if os.path.isdir(vqepath):  # Check if it's a directory
            for boostdir in os.listdir(vqepath):
                if "cma" in boostdir and any(
                    epoch in boostdir for epoch in studied_epochs_str
                ):
                    filepath = os.path.join(vqepath, boostdir, "boosting_data.json")
                    if os.path.isfile(filepath):  # Check if file exists
                        with open(filepath) as file:
                            results.append(json.load(file))
    return results


def load_jumps_given_boosting(boostpath, gci_steps=3):
    """Load GCI energies collected during a boosting process."""
    filepath = f"{boostpath}/boosting_data.json"
    energies = []
    with open(filepath) as file:
        results = json.load(file)
    for s in range(gci_steps):
        energies.append(results[str(s + 1)]["gci_loss"])
    return energies


def load_cgates_given_boosting(boostpath, optimizer, gci_steps=3):
    """Load the two qubit gates required to perform up to `gci_steps` rotations."""
    filepath = f"{boostpath}/boosting_data.json"
    cgates, accuracies, nfvals = [], [], []
    with open(filepath) as file:
        results = json.load(file)
    nfvals = load_nfval_given_opt(boostpath, gci_steps, optimizer)
    for s in range(gci_steps):
        cgates.append(results[str(s + 1)]["nmb_cz"] + results[str(s + 1)]["nmb_cnot"])
        accuracies.append(
            abs(results[str(s + 1)]["gci_loss"] - results[str(s + 1)]["target_energy"])
        )
    return np.array(cgates), np.array(accuracies), np.array(nfvals)


def plot_jumps(path, nqubits, nlayers, seed, rotation_type, optimizer, epochs, title):
    vqepath = f"{path}sgd_{nqubits}q_{nlayers}l_{seed}"
    # load losses
    losses = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers}l_{seed}/energies.npz"))["0"]
    losses_1Lmore = dict(
        np.load(f"{path}sgd_{nqubits}q_{nlayers+1}l_{seed}/energies.npz")
    )["0"]
    losses_2Lmore = dict(
        np.load(f"{path}sgd_{nqubits}q_{nlayers+2}l_{seed}/energies.npz")
    )["0"]

    # load boosts executed at given epochs
    energies = []
    for e in epochs:
        energies.append(
            load_jumps_given_boosting(f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s")
        )
    energies = np.array(energies)

    nlosses = len(losses)
    delta = nlosses / 50

    _, (a0, a1) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [6, 2]})
    a0.plot(losses, color="royalblue", lw=1, alpha=1, label="VQE 7 layers")
    a0.plot(
        losses_1Lmore,
        color="royalblue",
        lw=1,
        alpha=0.7,
        label="VQE 8 layers",
    )
    a0.plot(
        losses_2Lmore,
        color="royalblue",
        lw=1,
        alpha=0.5,
        label="VQE 9 layers",
    )
    a0.hlines(
        true_ground_energy,
        0,
        len(losses),
        lw=1,
        color="black",
        ls="--",
        alpha=0.4,
    )
    a0.hlines(
        true_second_energy,
        0,
        len(losses),
        lw=1,
        color="black",
        ls="--",
        alpha=0.4,
    )

    for i in [2, 1, 0]:
        energy = energies[:, i]
        a0.vlines(epochs, energy, losses[epochs], color=steps_colors[i], ls="-", lw=1)

        a0.hlines(
            energy,
            epochs - delta,
            epochs + delta,
            color=steps_colors[i],
            lw=1,
            label=f"{i+1} GCI steps",
        )
        a0.scatter(
            epochs,
            energy,
            color=steps_colors[i],
            marker="s",
            s=2,
        )
    for i in range(energies.shape[1]):
        a1.plot(
            epochs,
            1 - (energies[:, i] / true_ground_energy),
            color=steps_colors[i],
            marker="s",
            markersize=2,
            lw=1,
        )
    a1.plot(
        1 - (np.array(losses) / true_ground_energy),
        color="royalblue",
        lw=1,
        alpha=1,
        label="VQE loss",
    )
    a0.annotate(
        "Ground state $E_0$", (200, true_ground_energy + 0.02), c="grey", size=8
    )
    a0.annotate(
        "First excited state", (200, true_second_energy + 0.02), c="grey", size=8
    )
    a1.set_ylim(2e-5, 1e-1)
    a0.set_ylim(-15.4, -14.1)
    a0.legend(loc=3, bbox_to_anchor=(0.6, 0.4))
    a1.set_xlabel("VQE training epochs")
    a1.set_yscale("log")
    a1.set_ylabel(r"$ 1-\frac{E}{E_0} $")
    a1.set_yticks([1e-2, 1e-3, 1e-4])
    yticks = np.round(
        np.concatenate(([true_ground_energy, true_second_energy], energies[0, 0:3:2])),
        2,
    )
    a0.set_yticks(yticks)
    a0.set_ylabel(r"Energy expectation $E$")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=600)


def scatterplot_acc_vs_gates(
    path, nqubits, nlayers, seed, rotation_type, optimizer, epochs, title
):
    vqepath = f"{path}sgd_{nqubits}q_{nlayers}l_{seed}"
    # load losses
    losses = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers}l_{seed}/energies.npz"))["0"]

    # load boosts executed at given epochs
    cgates, accuracies, vqe_cg, vqe_cg_1Lmore, vqe_cg_2Lmore = [], [], [], [], []

    colors = ["royalblue"]
    colors.extend(list(sns.color_palette("Reds", n_colors=3).as_hex()))
    markers = ["o", "s", "s", "s"]

    for j, e in enumerate(epochs):
        # for each epoch load 2q gates, accuracies and gci nfval for each of 3 steps
        gci_cg, acc, gci_nfval = load_cgates_given_boosting(
            f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s",
            optimizer,
            3,
        )

        nparams = len(build_circuit(nqubits, nlayers).get_parameters())
        nparams_1Lmore = len(build_circuit(nqubits, nlayers + 1).get_parameters())
        nparams_2Lmore = len(build_circuit(nqubits, nlayers + 2).get_parameters())

        vqe_cg.append(2 * nparams * e * calculate_vqe_gates(nqubits, nlayers))
        vqe_cg_1Lmore.append(
            2 * nparams_1Lmore * e * calculate_vqe_gates(nqubits, nlayers + 1)
        )
        vqe_cg_2Lmore.append(
            2 * nparams_2Lmore * e * calculate_vqe_gates(nqubits, nlayers + 2)
        )

        gci_cg = gci_cg * gci_nfval + vqe_cg[-1]

        cgates.append(gci_cg)
        accuracies.append(acc)

    vqe_acc = abs(1 - np.array(losses[epochs]) / true_ground_energy)
    accuracies = 1 - (np.array(accuracies) + true_ground_energy) / true_ground_energy
    print(vqe_acc)

    cgates = np.hstack((np.array(vqe_cg).reshape(-1, 1), np.array(cgates)))
    accuracies = np.hstack((np.array(vqe_acc).reshape(-1, 1), np.array(accuracies)))

    print(np.array(cgates).shape)

    plt.figure(figsize=(6, 6 * 6 / 8))

    # looping over algorithms (dim = 4)
    for j in range(np.array(cgates).shape[1]):
        for i in range(np.array(cgates).shape[0]):
            if j == 0:
                if i != np.array(cgates).shape[0] - 1:
                    plt.annotate(
                        "",
                        xy=(cgates[i][j], accuracies[i][j]),
                        xytext=(cgates[i + 1][j], accuracies[i + 1][j]),
                        arrowprops=dict(arrowstyle="<-", color=colors[0]),
                    )
            if j != np.array(cgates).shape[1] - 1:
                plt.annotate(
                    "",
                    xy=(cgates[i][j], accuracies[i][j]),
                    xytext=(cgates[i][j + 1], accuracies[i][j + 1]),
                    arrowprops=dict(arrowstyle="<-", color=colors[j + 1]),
                )
            plt.scatter(
                cgates[i][j],
                accuracies[i][j],
                color=colors[j],
                marker=markers[j],
                s=40,
            )

    plt.title(f"{nqubits} qubits, {nlayers} layers, XXZ")
    plt.ylabel(r"$ 1-\frac{E}{E_0} $")
    plt.xlabel("# 2q gates")
    plt.yscale("log")
    plt.savefig(f"{title}.png", dpi=600)
    plt.savefig(f"{title}.pdf")


path = "vqe_data/compile_targets_light/"
hamiltonian = hamiltonians.XXZ(10, delta=0.5)
eigenvalues = hamiltonian.eigenvalues()
eigenvalues = np.sort(eigenvalues)
true_ground_energy = float(eigenvalues[0])
true_second_energy = float(eigenvalues[1])
studied_epochs = np.arange(1000, 5001, 1000)
steps_colors = ["#f99f1e", "#f05426", "#8c1734"]
epoch_shapes = ["o", "s", "P", "d", "v"]
dbi_steps = 3
nq = 10
nl = 7
opt = "Powell"
seed = 13

studied_epochs_str = [str(epoch) for epoch in studied_epochs]

plot_jumps(
    path=path,
    nqubits=nq,
    nlayers=nl,
    seed=seed,
    rotation_type="group_commutator_third_order_reduced",
    optimizer=opt,
    epochs=studied_epochs,
    title=f"{nq}Q{nl}L{seed}S_{opt}_jumps",
)

scatterplot_acc_vs_gates(
    path=path,
    nqubits=nq,
    nlayers=nl,
    seed=seed,
    rotation_type="group_commutator_third_order_reduced",
    optimizer=opt,
    epochs=studied_epochs,
    title=f"{nq}Q{nl}L{seed}S_{opt}_scatter",
)
