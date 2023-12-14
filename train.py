import argparse
import os
import json
import pathlib
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import qibo
from qibo.models.variational import VQE
from qibo import hamiltonians

from ansatze import build_circuit

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.png"

def loss(params, circuit, hamiltonian):
            circuit.set_parameters(params)
            result = hamiltonian.backend.execute_circuit(circuit)
            final_state = result.state()
            return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(final_state)

def generate_path(args):
    return f"./results/{args.optimizer}_{args.nqubits}q_{args.nlayers}l"

def create_folder(args):
    path = args.output_folder
    if path is None:
        path = generate_path(args) 
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def json_dump(path, results, output_dict):
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    with open(f"{path}/{OPTIMIZATION_FILE}", "w") as file:
        json.dump(output_dict, file, indent=4)

def json_load(path):
    f = open(path)
    return json.load(f)

def plot_results(folder):
    data = json_load(folder / OPTIMIZATION_FILE)
    energy = np.array(data["energy_list"])
    errors = np.array(data["energy_fluctuation"]) 
    epochs = range(len(energy))
    fig, ax = plt.subplots(2, 1, figsize = (14, 10))
    fig.suptitle('VQE Training', fontsize=20)
    ax[0].plot(epochs, energy, color = "navy", label = "VQE training")
    ax[0].fill_between(epochs, energy - errors, energy + errors, color = "royalblue", alpha = 0.5)
    ax[0].axhline(y=data["true_ground_energy"], color='r', linestyle='-', label = "True value")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Energy")
    ax[0].legend()
    ax[0].grid(True, which='major')
    ax[1].plot(epochs, energy / data["true_ground_energy"])
    ax[1].set_yscale('log')
    ax[1].axhline(y = 1, color = 'r')
    ax[1].grid(True)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Energy ratio with true value")
    plt.savefig(folder / PLOT_FILE)

def main(args):
    """VQE training and DBI boosting."""
    # set backend and number of classical threads
    qibo.set_backend(backend=args.backend, platform=args.platform)
    qibo.set_threads(args.nthreads)
    
    # setup the results folder
    path = create_folder(args)
    # build hamiltonian and variational quantum circuit
    ham = hamiltonians.XXZ(nqubits=args.nqubits)
    circ = build_circuit(nqubits=args.nqubits, nlayers=args.nlayers)

    # just print the circuit
    print(circ.draw())
    nparams = len(circ.get_parameters())
    # initialize VQE
    params_history = []
    loss_list = []
    fluctuations = []
    vqe = VQE(circuit=circ, hamiltonian=ham)
    def update_loss(params, vqe = vqe, loss_list = loss_list, loss_fluctuation = fluctuations, params_history = params_history):
        energy, energy_fluctuation = loss(params, vqe.circuit, vqe.hamiltonian)
        loss_list.append(energy)
        loss_fluctuation.append(energy_fluctuation)
        params_history.append(params)

    # fix numpy seed to ensure replicability of the experiment
    np.random.seed(42)
    initial_parameters = np.random.randn(nparams)
    results = vqe.minimize(initial_parameters, method=args.optimizer, callback= update_loss, )
    opt_results = results[2]

    # save final results
    output_dict = {
        "nqubits": args.nqubits,
        "nlayers": args.nlayers,
        "optimizer": args.optimizer,
        "best_loss": float(opt_results.fun),
        "energy_list": loss_list,
        "energy_fluctuation": fluctuations,
        "true_ground_energy": min(ham.eigenvalues()),
        "success": opt_results.success,
        "message": opt_results.message,
        "backend": args.backend,
        "platform": args.platform
    }
    print(params_history)
    json_dump(path, params_history, output_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VQE training hyper-parameters.')
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--optimizer", default="Powell", type=str)
    parser.add_argument("--output_folder", default=None, type=Optional[str])
    parser.add_argument("--backend", default="qibojit", type=str)
    parser.add_argument("--platform", default="dummy", type=str)
    parser.add_argument("--nthreads", default=1, type=int)

    args = parser.parse_args()
    main(args)
    path = generate_path(args)
    plot_results(pathlib.Path(path))

