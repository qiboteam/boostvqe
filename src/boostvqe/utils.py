import json
import logging
from pathlib import Path

import numpy as np
from qibo import get_backend

from boostvqe.ansatze import VQE, compute_gradients

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.pdf"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations"
LOSS_FILE = "energies"
GRADS_FILE = "gradients"
HAMILTONIAN_FILE = "hamiltonian_matrix.npz"
SEED = 42
TOL = 1e-10
DBI_ENERGIES = "dbi_energies"
DBI_FLUCTUATIONS = "dbi_fluctuations"
DBI_STEPS = "dbi_steps"
DBI_D_MATRIX = "dbi_d_matrices"


logging.basicConfig(level=logging.INFO)


def generate_path(args) -> str:
    """Generate path according to job parameters"""
    if args.output_folder is None:
        output_folder = "results"
    else:
        output_folder = args.output_folder
    return f"./{output_folder}/{args.optimizer}_{args.nqubits}q_{args.nlayers}l_{args.seed}"


def create_folder(path: str) -> Path:
    """Create folder and returns path"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    """Dump"""
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    json_file = Path(f"{path}/{OPTIMIZATION_FILE}")
    dump_json(json_file, output_dict)


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=4))


def json_load(path: str):
    f = open(path)
    return json.load(f)


def callback_energy_fluctuations(params, circuit, hamiltonian):
    """Evaluate the energy fluctuations"""
    circ = circuit.copy(deep=True)
    circ.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circ)
    final_state = result.state()
    return hamiltonian.energy_fluctuation(final_state)


def train_vqe(
    circ,
    ham,
    optimizer,
    initial_parameters,
    tol,
    loss,
    niterations=None,
    nmessage=1,
    training_options=None,
):
    """Helper function which trains the VQE according to `circ` and `ham`."""
    params_history, loss_list, fluctuations, grads_history = (
        [],
        [],
        [],
        [],
    )

    if training_options is None:
        options = {}
    else:
        options = training_options

    circ.set_parameters(initial_parameters)

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
        grads_history=grads_history,
        loss=loss,
    ):
        """
        Callback function that updates the energy, the energy fluctuations and
        the parameters lists.
        """
        energy = loss(params, vqe.circuit, vqe.hamiltonian)
        loss_list.append(float(energy))
        loss_fluctuation.append(
            callback_energy_fluctuations(params, vqe.circuit, vqe.hamiltonian)
        )
        params_history.append(params)
        grads_history.append(
            compute_gradients(
                parameters=params, circuit=circ.copy(deep=True), hamiltonian=ham
            )
        )

        iteration_count = len(loss_list) - 1

        if niterations is not None and iteration_count % nmessage == 0:
            logging.info(f"Optimization iteration {iteration_count}/{niterations}")
            logging.info(f"Loss {energy:.5}")

    callbacks(initial_parameters)
    logging.info("Minimize the energy")

    results = vqe.minimize(
        initial_parameters,
        method=optimizer,
        callback=callbacks,
        tol=tol,
        loss_func=loss,
        options=options,
    )

    return (
        results,
        params_history,
        loss_list,
        grads_history,
        fluctuations,
        vqe,
    )


def rotate_h_with_vqe(hamiltonian, vqe):
    """Rotate `hamiltonian` using the unitary representing the `vqe`."""
    # inherit backend from hamiltonian and circuit from vqe
    backend = hamiltonian.backend
    circuit = vqe.circuit
    # create circuit matrix and compute the rotation
    matrix_circ = np.matrix(backend.to_numpy(circuit.fuse().unitary()))
    matrix_circ_dagger = backend.cast(matrix_circ.getH())
    matrix_circ = backend.cast(matrix_circ)
    new_hamiltonian = np.matmul(
        matrix_circ_dagger, np.matmul(hamiltonian.matrix, matrix_circ)
    )
    return new_hamiltonian


def apply_dbi_steps(dbi, nsteps, stepsize=0.01, optimize_step=False):
    """Apply `nsteps` of `dbi` to `hamiltonian`."""
    step = stepsize
    energies, fluctuations, hamiltonians, steps, d_matrix = [], [], [], [], []
    logging.info(f"Applying {nsteps} steps of DBI to the given hamiltonian.")
    operators = []
    for _ in range(nsteps):
        if optimize_step:
            # Change logging level to reduce verbosity
            logging.getLogger().setLevel(logging.WARNING)
            step = dbi.hyperopt_step(
                step_min=1e-4, step_max=1, max_evals=50, verbose=True
            )
            # Restore the original logging level
            logging.getLogger().setLevel(logging.INFO)
        operators.append(dbi(step=step, d=dbi.diagonal_h_matrix))
        steps.append(step)
        d_matrix.append(np.diag(dbi.diagonal_h_matrix))
        zero_state = np.transpose([dbi.h.backend.zero_state(dbi.h.nqubits)])

        energies.append(dbi.h.expectation(zero_state))
        fluctuations.append(dbi.energy_fluctuation(zero_state))
        hamiltonians.append(dbi.h.matrix)

        logging.info(f"DBI energies: {energies}")
    return hamiltonians, energies, fluctuations, steps, d_matrix, operators
