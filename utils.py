import argparse
import json
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qibo.models import Circuit, Hamiltonian

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.png"
ROOT_FOLDER = "results"


def loss(params: list, circuit: Circuit, hamiltonian: Hamiltonian):
    """
    Given a VQE `circuit` with parameters `params`, this function returns the
    expectation vaule of the Hamiltonian and its energy fluctuation.
    """
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(
        final_state
    )


def generate_path(optimizer: str, nqubits: int, nlayers: int):
    """Returns the path that contains the results."""
    return f"./{ROOT_FOLDER}/{optimizer}_{nqubits}q_{nlayers}l"


def create_folder(path: str):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    with open(f"{path}/{OPTIMIZATION_FILE}", "w") as file:
        json.dump(output_dict, file, indent=4)


def json_load(path: str):
    f = open(path)
    return json.load(f)


def plot_results(folder: pathlib.Path, energy_dbi: Optional[Tuple] = None):
    """Plots the energy and the energy fluctuations."""
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
    if energy is not None:
        ax[0].axhline(y=energy_dbi[0], color="orange", linestyle="dashed", label="DBI")

    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Energy")
    ax[0].legend()
    ax[0].grid(True, which="major")
    ax[1].plot(epochs, energy / data["true_ground_energy"])
    if energy is not None:
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
