import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.png"


def loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(
        final_state
    )


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
    ax[1].plot(epochs, energy / data["true_ground_energy"])
    ax[1].set_yscale("log")
    ax[1].axhline(y=1, color="r")
    ax[1].grid(True)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Energy ratio with true value")
    plt.savefig(folder / PLOT_FILE)
