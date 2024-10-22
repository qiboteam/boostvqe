import argparse
import json
import logging
import pathlib
from functools import partial
from typing import Optional

import numpy as np

# qibo's
from qibo import Circuit, gates, hamiltonians, set_backend
from qibo.backends import GlobalBackend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

# boostvqe's
from boostvqe.plotscripts import plot_gradients, plot_loss
from boostvqe.training_utils import Model, vqe_loss
from boostvqe.utils import (
    DBI_D_MATRIX,
    DBI_ENERGIES,
    DBI_FLUCTUATIONS,
    DBI_STEPS,
    FLUCTUATION_FILE,
    GRADS_FILE,
    HAMILTONIAN_FILE,
    LOSS_FILE,
    SEED,
    TOL,
    apply_dbi_steps,
    create_folder,
    generate_path,
    results_dump,
    rotate_h_with_vqe,
    train_vqe,
)

NSHOTS = 100


def dbqa_vqe(
    circuit: Circuit,  # in place of ansatz
    output_folder: str,
    backend: Optional[str] = "qibojit",
    platform: Optional[str] = None,
    optimizer: str = "Powell",
    optimizer_options: dict = {},
    tol: float = TOL,
    decay_rate_lr: float = 1.0,
    nboost: int = 1,
    boost_frequency: int = 10,
    dbi_steps: int = 1,
    store_h: bool = False,
    hamiltonian: str = "XXZ",
    seed: int = SEED,
    nshots: int = NSHOTS,
    nlayers: int = 0,
    mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.single_commutator,
):
    """
    Args:

       circuit (Circuit): VQE circuit.

       output_folder (str): where to dump the results.

       backend (str, default: "qibojit"):
            Specifies the quantum backend to use. Options include different backend implementations
            such as "qibojit" or "qibotf".

        platform (str, default: None):
            Specifies the platform to run on, typically used for running on GPUs (e.g., "CUDA").

        optimizer (str, default: "Powell"):
            Optimizer used for VQE training. Options include "Powell", "BFGS", and others.

        optimizer_options (str, optional):
            Custom options to fine-tune the behavior of the optimizer.

        tol (float, default: TOL):
            Absolute precision tolerance to stop VQE training once the desired convergence is achieved.

        decay_rate_lr (float, default: 1.0):
            Learning rate decay factor used when the optimizer is SGD (stochastic gradient descent).

        nboost (int, default: 1):
            Number of times DBI (Deterministic Boost Iteration) is applied in the optimization process.
            If set to 1, no boosting is performed.

        boost_frequency (int, default: 10):
            Number of optimization steps between DBI boosting calls.

        dbi_steps (int, default: 1):
            Number of DBI iterations performed each time DBI is called.

        stepsize (float, default: 0.01):
            Step size used during DBI updates.

        store_h (bool, default: False):
            If this flag is set, the Hamiltonian `H` is stored at each iteration.

        hamiltonian (str, default: "XXZ"):
            Specifies the Hamiltonian to use, chosen from predefined options in the `qibo.hamiltonians` module.

        seed (str, default: SEED):
            Random seed to ensure reproducibility of results.

        nshots (int, optional):
            Number of shots (measurements) to use during the quantum computation.

        nlayers (int):
            Number of circuit layers.

        mode:
            Define the DBI Generator.
    """

    output_dict = locals()
    del output_dict["circuit"]
    output_dict["nqubits"] = circuit.nqubits
    # output_dict["nlayers"] = nlayers  # TODO: maybe remove it
    output_dict["mode"] = str(mode)
    # set backend and number of classical threads

    if platform is not None:
        set_backend(backend=backend, platform=platform)
    else:
        set_backend(backend=backend)
        platform = GlobalBackend().platform

    nqubits = circuit.nqubits
    # setup the results folder
    logging.info("Set VQE")
    path = pathlib.Path(
        create_folder(
            generate_path(
                output_folder,
                optimizer,
                nqubits,
                seed,
                decay_rate_lr,
                nlayers,
            )
        )
    )
    ham = getattr(Model, hamiltonian)(nqubits)  # TODO : use only Model and not str
    target_energy = np.real(np.min(np.asarray(ham.eigenvalues())))

    # construct circuit from parsed ansatz name
    circ0 = circuit

    logging.info(circ0.draw())

    circ = circ0.copy(deep=True)
    backend = ham.backend
    zero_state = backend.zero_state(nqubits)

    loss = partial(vqe_loss, nshots=nshots)

    # fix numpy seed to ensure replicability of the experiment
    np.random.seed(int(seed))
    initial_parameters = np.random.uniform(-np.pi, np.pi, len(circ.get_parameters()))

    # vqe lists
    params_history, loss_history, grads_history, fluctuations = {}, {}, {}, {}
    # dbi lists
    boost_energies, boost_fluctuations_dbi, boost_steps, boost_d_matrix = {}, {}, {}, {}
    # hamiltonian history
    fun_eval, hamiltonians_history = [], []
    hamiltonians_history.append(ham.matrix)
    new_hamiltonian = ham
    nboost += 1
    for b in range(nboost):
        logging.info(f"Running {b+1}/{nboost} max optimization rounds.")
        boost_energies[b], boost_fluctuations_dbi[b] = [], []
        params_history[b], loss_history[b], fluctuations[b] = [], [], []
        # train vqe
        (
            partial_results,
            partial_params_history,
            partial_loss_history,
            partial_grads_history,
            partial_fluctuations,
            vqe,
        ) = train_vqe(
            circ,
            ham,  # Fixed hamiltonian
            optimizer,
            initial_parameters,
            tol,
            niterations=boost_frequency,
            nmessage=1,
            loss=loss,
            training_options=optimizer_options,
        )
        # append results to global lists
        params_history[b] = np.array(partial_params_history)
        loss_history[b] = np.array(partial_loss_history)
        grads_history[b] = np.array(partial_grads_history)
        fluctuations[b] = np.array(partial_fluctuations)
        # this works with scipy.optimize.minimize only
        if optimizer not in ["sgd", "cma"]:
            fun_eval.append(int(partial_results[2].nfev))

        # build new hamiltonian using trained VQE
        if b != nboost - 1:
            new_hamiltonian_matrix = rotate_h_with_vqe(hamiltonian=ham, vqe=vqe)
            new_hamiltonian = hamiltonians.Hamiltonian(
                nqubits, matrix=new_hamiltonian_matrix
            )
            hamiltonians_history.extend(new_hamiltonian_matrix)
            # Initialize DBI
            dbi = DoubleBracketIteration(
                hamiltonian=new_hamiltonian,
                mode=mode,
            )

            zero_state_t = np.transpose([zero_state])
            energy_h0 = float(dbi.h.expectation(np.array(zero_state_t)))
            fluctuations_h0 = float(dbi.h.energy_fluctuation(zero_state_t))

            # apply DBI
            (
                dbi_hamiltonians,
                dbi_energies,
                dbi_fluctuations,
                dbi_steps,
                dbi_d_matrix,
                dbi_operators,
            ) = apply_dbi_steps(dbi=dbi, nsteps=dbi_steps)
            # Update the circuit appending the DBI generator
            # and the old circuit with non trainable circuit
            dbi_operators = [
                ham.backend.cast(np.matrix(ham.backend.to_numpy(operator)))
                for operator in dbi_operators
            ]

            old_circ_matrix = circ.unitary()
            # Remove measurement gates
            # Add the DBI operators and the unitary circuit matrix to the circuit
            # We are using the dagger operators because in Qibo the DBI step
            # is implemented as V*H*V_dagger
            circ = circ0.copy(deep=True)
            for gate in reversed([old_circ_matrix] + dbi_operators):
                circ.add(gates.Unitary(gate, *range(nqubits), trainable=False))
            hamiltonians_history.extend(dbi_hamiltonians)
            # append dbi results
            dbi_fluctuations.insert(0, fluctuations_h0)
            dbi_energies.insert(0, energy_h0)
            boost_fluctuations_dbi[b] = np.array(dbi_fluctuations)
            boost_energies[b] = np.array(dbi_energies)
            boost_steps[b] = np.array(dbi_steps)
            boost_d_matrix[b] = np.array(dbi_d_matrix)
            initial_parameters = np.zeros(len(initial_parameters))
            circ.set_parameters(initial_parameters)

            # reduce the learning rate after DBI has been applied
            if "learning_rate" in optimizer_options:
                optimizer_options["learning_rate"] *= decay_rate_lr

    best_loss = min(np.min(array) for array in loss_history.values())

    opt_results = partial_results[2]
    # save final results
    output_dict.update(
        {
            "true_ground_energy": target_energy,
            "feval": list(fun_eval),
            "energy": float(vqe.hamiltonian.expectation(zero_state)),
            "fluctuations": float(vqe.hamiltonian.dense.energy_fluctuation(zero_state)),
            "reached_accuracy": float(np.abs(target_energy - best_loss)),
        }
    )
    # this works only with scipy.optimize.minimize
    if optimizer not in ["sgd", "cma"]:
        output_dict.update(
            {
                "best_loss": float(opt_results.fun),
                "success": bool(opt_results.success),
                "message": opt_results.message,
                "feval": list(fun_eval),
            }
        )
    np.savez(
        path / LOSS_FILE,
        **{json.dumps(key): np.array(value) for key, value in loss_history.items()},
    )
    np.savez(
        path / GRADS_FILE,
        **{json.dumps(key): np.array(value) for key, value in grads_history.items()},
    )
    np.savez(
        path / FLUCTUATION_FILE,
        **{json.dumps(key): np.array(value) for key, value in fluctuations.items()},
    )
    if store_h:
        np.savez(path / HAMILTONIAN_FILE, *hamiltonians_history)
    np.savez(
        path / DBI_ENERGIES,
        **{json.dumps(key): np.array(value) for key, value in boost_energies.items()},
    )
    np.savez(
        path / DBI_D_MATRIX,
        **{json.dumps(key): np.array(value) for key, value in boost_d_matrix.items()},
    )
    np.savez(
        path / DBI_STEPS,
        **{json.dumps(key): np.array(value) for key, value in boost_steps.items()},
    )
    np.savez(
        path / DBI_FLUCTUATIONS,
        **{
            json.dumps(key): np.array(value)
            for key, value in boost_fluctuations_dbi.items()
        },
    )

    logging.info("Dump the results")
    results_dump(path, params_history, output_dict)
    plot_loss(
        path=path,
        title="Energy history",
    )
    plot_gradients(path=path, title="Grads history")
