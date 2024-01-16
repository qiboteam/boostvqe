import json
import logging
from pathlib import Path

import numpy as np
from qibo.models.variational import VQE

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.pdf"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations"
LOSS_FILE = "energies"
HAMILTONIAN_FILE = "hamiltonian_matrix.npy"
FLUCTUATION_FILE2 = "fluctuations2"
LOSS_FILE2 = "energies2"
SEED = 42
TOL = 1e-4
DBI_FILE = "dbi_matrix"
DBI_RESULTS = "dbi_output.json"

logging.basicConfig(level=logging.INFO)


def generate_path(args):
    if args.output_folder is None:
        output_folder = "results"
    else:
        output_folder = args.output_folder
    return f"./{output_folder}/{args.optimizer}_{args.nqubits}q_{args.nlayers}l"


def create_folder(path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    json_file = Path(f"{path}/{OPTIMIZATION_FILE}")
    dump_json(json_file, output_dict)


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=4))


def json_load(path: str):
    f = open(path)
    return json.load(f)


def loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(
        final_state
    )


def train_vqe(circ, ham, optimizer, initial_parameters, tol, tracker=0, nmessage=None):
    params_history = []
    loss_list = []
    fluctuations = []
    circ.set_parameters(initial_parameters)
    if tol is None:
        tol = TOL
    vqe = VQE(
        circuit=circ,
        hamiltonian=ham,
    )

    def callbacks(
        params,
        vqe=vqe,
        loss_list=loss_list,
        loss_fluctuation=fluctuations,
        params_history=params_history,
    ):
        """
        Callback function that updates the energy, the energy fluctuations and
        the parameters lists.
        """
        energy, energy_fluctuation = loss(params, vqe.circuit, vqe.hamiltonian)
        loss_list.append(float(energy))
        loss_fluctuation.append(float(energy_fluctuation))
        params_history.append(params)

        print("optimizing")

    callbacks(initial_parameters)

    # fix numpy seed to ensure replicability of the experiment
    logging.info("Minimize the energy")

    results = vqe.minimize(
        initial_parameters,
        method=optimizer,
        callback=callbacks,
        tol=tol,
    )
    return results, params_history, loss_list, fluctuations
