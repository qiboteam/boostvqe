import json
import logging
from pathlib import Path

import numpy as np
from qibo.models.variational import VQE

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.pdf"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations.npy"
LOSS_FILE = "energies.npy"
HAMILTONIAN_FILE = "hamiltonian_matrix.npy"
FLUCTUATION_FILE2 = "fluctuations2"
LOSS_FILE2 = "energies2"
SEED = 42
TOL = 1e-4
DBI_ENERGIES = "dbi_energies"
DBI_FLUCTUATIONS = "dbi_fluctuations"


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


def train_vqe(
    circ, ham, optimizer, initial_parameters, tol, niterations=None, nmessage=1
):
    """Helper function which trains the VQE according to `circ` and `ham`."""
    params_history, loss_list, fluctuations = [], [], []
    circ.set_parameters(initial_parameters)

    if niterations is not None:
        iteration_count = 0

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

        nonlocal iteration_count
        iteration_count += 1

        if niterations is not None and iteration_count % nmessage == 0:
            logging.info(f"Optimization iteration {iteration_count}/{niterations}")

        if iteration_count >= niterations:
            raise StopIteration("Maximum number of iterations reached.")

    callbacks(initial_parameters)

    # fix numpy seed to ensure replicability of the experiment
    logging.info("Minimize the energy")

    try:
        results = vqe.minimize(
            initial_parameters,
            method=optimizer,
            callback=callbacks,
            tol=tol,
        )
    except StopIteration as e:
        logging.info(str(e))

    return results, params_history, loss_list, fluctuations, vqe


def rotate_h_with_vqe(hamiltonian, vqe):
    """Rotate `hamiltonian` using the unitary representing the `vqe`."""
    # inherit backend from hamiltonian and circuit from vqe
    backend = hamiltonian.backend
    circuit = vqe.circuit
    # create circuit matrix and compute the rotation
    matrix_circ = np.matrix(backend.to_numpy(circuit.unitary()))
    matrix_circ_dagger = backend.cast(matrix_circ.getH())
    matrix_circ = backend.cast(matrix_circ)
    new_hamiltonian = matrix_circ_dagger @ hamiltonian.matrix @ matrix_circ
    return new_hamiltonian


def apply_dbi_steps(dbi, nsteps, stepsize=0.01, optimize_step=False):
    """Apply `nsteps` of `dbi` to `hamiltonian`."""
    step = stepsize
    logging.info(f"Applying {nsteps} steps of DBI to the given hamiltonian.")
    for _ in range(nsteps):
        if optimize_step:
            # Change logging level to reduce verbosity
            logging.getLogger().setLevel(logging.WARNING)
            step = dbi.hyperopt_step(
                step_min=1e-4, step_max=1, max_evals=50, verbose=True
            )
            # Restore the original logging level
            logging.getLogger().setLevel(logging.INFO)
        dbi(step=step, d=dbi.diagonal_h_matrix)
    return dbi.h
