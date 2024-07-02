import os
import json 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    nfvals = load_nfval_given_opt(boostpath, gci_steps, optimizer)
    for s in range(gci_steps):
        cgates.append(results[str(s+1)]["nmb_cz"] + results[str(s+1)]["nmb_cnot"])
        accuracies.append(abs(results[str(s+1)]["gci_loss"] - results[str(s+1)]["target_energy"]))
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
    cgates, accuracies, vqe_cg, vqe_cg_1Lmore, vqe_cg_2Lmore = [], [], [], [], []

    colors = ["royalblue"]
    colors.extend(list(sns.color_palette("Reds", n_colors=3).as_hex()))

    labels = ["VQE training", "1 GCI step", "2 GCI steps", "3 GCI steps"]
    size = 6
    alpha = 1

    for j, e in enumerate(epochs):
        # for each epoch load 2q gates, accuracies and gci nfval for each of 3 steps
        gci_cg, acc, gci_nfval = load_cgates_given_boosting(
            f"{vqepath}/{rotation_type}_{optimizer}_{e}e_3s",
            optimizer,
            3,
            )

        nparams = len(build_circuit(nqubits, nlayers).get_parameters())
        nparams_1Lmore = len(build_circuit(nqubits, nlayers+1).get_parameters())
        nparams_2Lmore = len(build_circuit(nqubits, nlayers+2).get_parameters())

        vqe_cg.append(2 * nparams * e * calculate_vqe_gates(nqubits, nlayers))
        vqe_cg_1Lmore.append(2 * nparams_1Lmore * e * calculate_vqe_gates(nqubits, nlayers+1))
        vqe_cg_2Lmore.append(2 * nparams_2Lmore * e * calculate_vqe_gates(nqubits, nlayers+2))
        
        gci_cg = gci_cg * gci_nfval + vqe_cg[-1]

        cgates.append(gci_cg)
        accuracies.append(acc)
    
    vqe_acc = abs(np.array(losses[epochs]) - true_ground_energy)
    print(vqe_acc)
    
    min_cg = min(gci_cg)
    cgates = np.hstack((np.array(vqe_cg).reshape(-1,1), np.array(cgates)))
    accuracies = np.hstack((np.array(vqe_acc).reshape(-1,1), np.array(accuracies)))

    print(np.array(cgates).shape)

    plt.figure(figsize=(6, 6 * 6/8))


    # looping over algorithms (dim = 4)
    for j in range(np.array(cgates).shape[1]):
        plt.plot(
            np.array(cgates).T[j], 
            np.array(accuracies).T[j], 
            color=colors[j],
            lw=1,
            alpha=0.5,
            label=labels[j],
        )
        # looping over epochs
        for i in range(np.array(cgates).shape[0]):
            if (j != np.array(cgates).shape[1] - 1):
                plt.annotate(
                    '', 
                    xy=(cgates[i][j], accuracies[i][j]), 
                    xytext=(cgates[i][j+1], accuracies[i][j+1]), 
                    arrowprops=dict(arrowstyle='<-', color=colors[j+1])
                )  
            plt.scatter(
                cgates[i][j],
                accuracies[i][j],
                color=colors[j],
                s=40,
            ) 
   

    plt.title(f"{nqubits} qubits, {nlayers} layers, XXZ")
    # plt.legend(ncols=2)
    plt.ylabel("Absolute error")
    plt.xlabel("# 2q gates")
    plt.yscale("log")
    plt.legend(loc=3)
    plt.savefig(f"{title}.png", dpi=600)
    plt.savefig(f"{title}.pdf")
    plt.show()
        
    plt.figure(figsize=(6, 6 * 6/8))

    for i in range(3):
        cg = np.array(cgates).T[i]
        accs = np.array(accuracies).T[i]

        plt.plot(
            epochs,
            accs, 
            color=colors[i],
            label=f"{i+1} GCI steps"
        )
        for k, a in enumerate(accs):
            plt.scatter(
                epochs[k],
                a,
                color=colors[i],
                s=20 * cg[k] / (min_cg/2),
            )
    # plt.plot(
    #     vqe_cg,
    #     abs(losses[epochs] - true_ground_energy),
    #     color=colors[3],
    #     marker="o",
    #     markersize=size,
    #     label=f"VQE"
    # )
    # plt.plot(
    #     vqe_cg_1Lmore,
    #     abs(losses_1Lmore[epochs] - true_ground_energy),
    #     color=colors[4],
    #     marker="o",
    #     markersize=size,
    #     label=f"VQE +1L"
    # )
    # plt.plot(
    #     vqe_cg_2Lmore,
    #     abs(losses_2Lmore[epochs] - true_ground_energy),
    #     color=colors[5],
    #     marker="o",
    #     markersize=size,
    #     label=f"VQE +2L"
    # )       

    plt.title(f"{nqubits} qubits, {nlayers} layers, XXZ")
    plt.legend(ncols=2)
    plt.ylabel("Absolute error")
    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.savefig(f"{title}_epochs.png")
    plt.savefig(f"{title}_epochs.pdf")
    plt.show()

path = "../results/moreonXXZ/compile_targets/"
true_ground_energy = -15.276131122065937
studied_epochs = np.arange(250, 5001, 500)
steps_colors = ["#f99f1e", "#f05426", "#8c1734"]
epoch_shapes = ["o", "s", "P", "d", "v"] 
dbi_steps = 3
nq = 10
nl = 7
opt = "cma"
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


