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

def load_cgates_given_boosting(boostpath, gci_steps=3):
    """Load the two qubit gates required to perform up to `gci_steps` rotations."""
    filepath = f"{boostpath}/boosting_data.json"
    cgates, accuracies = [], []    
    with open(filepath, 'r') as file:
        results = json.load(file)  
    for s in range(gci_steps):
        cgates.append(results[str(s+1)]["nmb_cz"] + results[str(s+1)]["nmb_cnot"])
        accuracies.append(abs(results[str(s+1)]["gci_loss"] - results[str(s+1)]["target_energy"]))
    return cgates, accuracies


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
    cgates, accuracies = []
    for e in epochs:
        cg, acc = load_jumps_given_boosting(
                f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s"
            )
        cgates.append(cg)
        accuracies.append(acc)



path = "../results/moreonXXZ/compile_targets_light/"
true_ground_energy = -15.276131122065937
studied_epochs = [1000, 2000, 3000, 4000, 5000]
steps_colors = ["#f99f1e", "#f05426", "#8c1734"]
dbi_steps = 3
nq = 10
nl = 4
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
    title=f"{nq}Q{nl}L{seed}S_{opt}"
)


