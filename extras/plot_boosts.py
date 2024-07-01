import os
import json 

import numpy as np
import matplotlib.pyplot as plt

from boostvqe.ansatze import build_circuit

def calculate_vqe_gates(nqubits, nlayers):
    """Calculate total number of CZ gates of a given VQE circuit."""
    c = build_circuit(nqubits, nlayers)
    ncz = len(c.gates_of_type("cz"))
    return ncz

def load_nfval_given_opt(boostpath, gci_steps, optimizer):
    """Load the number of function evaluations given optimizer and results."""
    filepath = f"{boostpath}/boosting_data.json"
    nfvals = []    
    with open(filepath, 'r') as file:
        res_dict = json.load(file)  
    for s in range(gci_steps):
        if optimizer == "sgd":
            raise ValueError("nfvals calculation not supported for SGD")
        elif optimizer == "cma":
            nfvals.append(res_dict[str(s+1)]["cma_extras"]["evaluations"])
        else:
            nfvals.append(res_dict[str(s+1)][f"{optimizer}_extras"]["nfev"])
    return nfvals

def load_all_results_given_path(path):
    """Load all dictionaries given path to a collection of VQE trainings."""
    # we collect results here
    results = []
    for vqedir in os.listdir(path):
        vqepath = os.path.join(path, vqedir)
        if os.path.isdir(vqepath):  # Check if it's a directory
            for boostdir in os.listdir(vqepath):
                if (
                    "cma" in boostdir and 
                    any(epoch in boostdir for epoch in studied_epochs_str)
                ):
                    filepath = os.path.join(vqepath, boostdir, "boosting_data.json")
                    if os.path.isfile(filepath):  # Check if file exists
                        with open(filepath, 'r') as file:
                            results.append(json.load(file))
    return results


def load_jumps_given_boosting(boostpath, gci_steps=3):
    """Load GCI energies collected during a boosting process."""
    filepath = f"{boostpath}/boosting_data.json"
    energies = []    
    with open(filepath, 'r') as file:
        results = json.load(file)  
    for s in range(gci_steps):
        energies.append(results[str(s+1)]["gci_loss"])
    return energies

def load_cgates_given_boosting(boostpath, optimizer, gci_steps=3):
    """Load the two qubit gates required to perform up to `gci_steps` rotations."""
    filepath = f"{boostpath}/boosting_data.json"
    cgates, accuracies, nfvals = [], [], []    
    with open(filepath, 'r') as file:
        results = json.load(file)  
    for s in range(gci_steps):
        cgates.append(results[str(s+1)]["nmb_cz"] + results[str(s+1)]["nmb_cnot"])
        accuracies.append(abs(results[str(s+1)]["gci_loss"] - results[str(s+1)]["target_energy"]))
        nfvals.append(load_nfval_given_opt(boostpath, gci_steps, optimizer))
    return np.array(cgates), np.array(accuracies), np.array(nfvals)


def plot_jumps(path, nqubits, nlayers, seed, rotation_type, optimizer, epochs, title):

    vqepath = f"{path}sgd_{nqubits}q_{nlayers}l_{seed}"
    # load losses
    losses = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers}l_{seed}/energies.npz"))["0"]
    losses_1Lmore = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers+1}l_{seed}/energies.npz"))["0"]
    losses_2Lmore = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers+2}l_{seed}/energies.npz"))["0"]

    # load boosts executed at given epochs
    energies = []
    for e in epochs:
        energies.append(
            load_jumps_given_boosting(
                f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s"
            )
        )

    nlosses = len(losses)
    delta = nlosses/50

    plt.figure(figsize=(6, 6*6/8))
    plt.plot(losses, color="royalblue", lw=1, alpha=1, label="VQE loss")
    plt.plot(losses_1Lmore, color="royalblue", lw=1, alpha=0.8, label="VQE loss +1 layer", ls="--")
    plt.plot(losses_2Lmore, color="royalblue", lw=1, alpha=0.6, label="VQE loss +2 layer", ls="-.")

    plt.hlines(true_ground_energy, 0, len(losses), lw=1, color="black", ls="-", label="Target energy")
    for j, e in enumerate(epochs):
        for i, energy in enumerate(reversed(energies[j])):
            plt.vlines(epochs[j], energy, losses[epochs[j]], color=steps_colors[i], ls="-", lw=1)
            if j == 0:
                plt.hlines(energy, epochs[j]-delta, epochs[j]+delta, color=steps_colors[i], lw=1, label=f"{len(energies[j])-i} GCI steps")
            else:
                plt.hlines(energy, epochs[j]-delta, epochs[j]+delta, color=steps_colors[i], lw=1)
    plt.ylim(-15.4, -13)
    plt.legend(ncols=2)
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title(f"{nqubits} qubits, {nlayers} layers, XXZ")
    plt.savefig(f"{title}.png", dpi=600)



def scatterplot_acc_vs_gates(path, nqubits, nlayers, seed, rotation_type, optimizer, epochs, title):
    vqepath = f"{path}sgd_{nqubits}q_{nlayers}l_{seed}"
    # load losses
    losses = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers}l_{seed}/energies.npz"))["0"]
    losses_1Lmore = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers+1}l_{seed}/energies.npz"))["0"]
    losses_2Lmore = dict(np.load(f"{path}sgd_{nqubits}q_{nlayers+2}l_{seed}/energies.npz"))["0"]
    # load boosts executed at given epochs
    cgates, accuracies, gci_nfvals = [], [], []

    plt.figure(figsize=(6, 6 * 6/8))

    for j, e in enumerate(epochs):
        gci_cg, accuracies, gci_nfval = load_cgates_given_boosting(
            f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s",
            optimizer,
            3,
            )

        nparams = len(build_circuit(nqubits, nlayers).get_parameters())
        nparams_1Lmore = len(build_circuit(nqubits, nlayers+1).get_parameters())
        nparams_2Lmore = len(build_circuit(nqubits, nlayers+2).get_parameters())

        vqe_cg = 2 * nparams * e * calculate_vqe_gates(nqubits, nlayers)
        vqe_cg_1Lmore = 2 * nparams_1Lmore * e * calculate_vqe_gates(nqubits, nlayers+1)
        vqe_cg_2Lmore = 2 * nparams_2Lmore * e * calculate_vqe_gates(nqubits, nlayers+2)
        gci_cg = gci_cg * gci_nfval + vqe_cg
        gci_cg = gci_cg[0]

        cg_list = [vqe_cg, vqe_cg_1Lmore, vqe_cg_2Lmore, gci_cg[-1]]

        max_cg = max(cg_list)

        size = 50

        for i, gates in enumerate(gci_cg):
            if j == 0:
                if i == 0:
                    plt.scatter(
                        abs(losses[e] - true_ground_energy), 
                        vqe_cg, 
                        color="royalblue", 
                        marker=epoch_shapes[j],
                        label="VQE",
                        s=size,
                    )
                    plt.scatter(
                        abs(losses_1Lmore[e] - true_ground_energy), 
                        vqe_cg_1Lmore, 
                        color="#51B4F0",
                        marker=epoch_shapes[j],
                        label="VQE +1 layer",
                        s=size,
                    )
                    plt.scatter(
                        abs(losses_2Lmore[e] - true_ground_energy), 
                        vqe_cg_2Lmore, 
                        color="#51F0DF",
                        marker=epoch_shapes[j],
                        label="VQE +2 layers",
                        s=size,
                    )
                    plt.scatter(
                        accuracies[i], 
                        gates, 
                        color=steps_colors[i], 
                        marker=epoch_shapes[j], 
                        label=f"{i+1} GCI steps",
                        s=size,
                    )
                else:
                    plt.scatter(
                        accuracies[i], 
                        gates, 
                        color=steps_colors[i], 
                        marker=epoch_shapes[j], 
                        label=f"{i+1} GCI steps",
                        s=size,
                    )
                    
            else:
                plt.scatter(
                    accuracies[i], 
                    gates, 
                    color=steps_colors[i], 
                    marker=epoch_shapes[j],
                    s=size,
                )
                plt.scatter(
                    abs(losses[e] - true_ground_energy), 
                    vqe_cg, 
                    color="royalblue", 
                    marker=epoch_shapes[j],
                    s=size,
                )
                plt.scatter(
                    abs(losses_1Lmore[e] - true_ground_energy), 
                    vqe_cg_1Lmore, 
                    color="#51B4F0",
                    alpha=0.8, 
                    marker=epoch_shapes[j],
                    s=size,
                )
                plt.scatter(
                    abs(losses_2Lmore[e] - true_ground_energy), 
                    vqe_cg_2Lmore, 
                    color="#51F0DF",
                    alpha=0.6, 
                    marker=epoch_shapes[j],
                    s=size,
                )
        plt.hlines(max_cg, 0, 0.7, color="black", ls="-", lw=0.7)
    plt.title(f"{nqubits} qubits, {nlayers} layers, XXZ")
    plt.legend(ncols=2)
    plt.xlabel("Absolute error")
    plt.ylabel("# 2q gates")
    plt.savefig(f"{title}.png")
    plt.show()
        


path = "../results/moreonXXZ/compile_targets_light/"
true_ground_energy = -15.276131122065937
studied_epochs = [1000, 2000, 3000, 4000, 5000]
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
    title=f"{nq}Q{nl}L{seed}S_{opt}_jumps"
)

scatterplot_acc_vs_gates(
    path=path, 
    nqubits=nq,
    nlayers=nl,
    seed=seed,
    rotation_type="group_commutator_third_order_reduced",
    optimizer=opt,
    epochs=studied_epochs,
    title=f"{nq}Q{nl}L{seed}S_{opt}_scatter"    
)


