import argparse
import json
import logging
import pathlib
from functools import partial

import numpy as np

# qibo's
import qibo
from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

# boostvqe's
from boostvqe.ansatze import build_circuit
from boostvqe.plotscripts import plot_gradients, plot_loss
from boostvqe.training_utils import Ham, vqe_loss
from boostvqe.utils import (
    DBI_D_MATRIX,
    DBI_ENERGIES,
    DBI_FLUCTUATIONS,
    DBI_STEPS,
    DELTA,
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

logging.basicConfig(level=logging.INFO)


def main(args):
    """VQE training."""
    # set backend and number of classical threads

    if args.platform is not None:
        qibo.set_backend(backend=args.backend, platform=args.platform)
    else:
        qibo.set_backend(backend=args.backend)
        args.platform = GlobalBackend().platform

    if args.optimizer_options is None:
        opt_options = {}
    else:
        opt_options = json.loads(args.optimizer_options)

    # setup the results folder
    logging.info("Set VQE")
    path = pathlib.Path(create_folder(generate_path(args)))
    ham = getattr(Ham, args.hamiltonian)(args.nqubits)
    target_energy = np.real(np.min(np.asarray(ham.eigenvalues())))
    circ0 = build_circuit(
        nqubits=args.nqubits,
        nlayers=args.nlayers,
    )
    circ = circ0.copy(deep=True)
    backend = ham.backend
    zero_state = backend.zero_state(args.nqubits)

    loss = partial(vqe_loss, nshots=args.nshots)

    # fix numpy seed to ensure replicability of the experiment
    np.random.seed(int(args.seed))
    initial_parameters = np.random.uniform(-np.pi, np.pi, len(circ.get_parameters()))

    # vqe lists
    params_history, loss_history, grads_history, fluctuations = {}, {}, {}, {}
    # dbi lists
    boost_energies, boost_fluctuations_dbi, boost_steps, boost_d_matrix = {}, {}, {}, {}
    # hamiltonian history
    fun_eval, hamiltonians_history = [], []
    hamiltonians_history.append(ham.matrix)
    new_hamiltonian = ham
    args.nboost += 1
    for b in range(args.nboost):
        logging.info(f"Running {b+1}/{args.nboost} max optimization rounds.")
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
            args.optimizer,
            initial_parameters,
            args.tol,
            niterations=args.boost_frequency,
            nmessage=1,
            loss=loss,
            training_options=opt_options,
        )
        # append results to global lists
        params_history[b] = np.array(partial_params_history)
        loss_history[b] = np.array(partial_loss_history)
        grads_history[b] = np.array(partial_grads_history)
        fluctuations[b] = np.array(partial_fluctuations)
        # this works with scipy.optimize.minimize only
        if args.optimizer not in ["sgd", "cma"]:
            fun_eval.append(int(partial_results[2].nfev))

        # build new hamiltonian using trained VQE
        if b != args.nboost - 1:
            new_hamiltonian_matrix = rotate_h_with_vqe(hamiltonian=ham, vqe=vqe)
            new_hamiltonian = hamiltonians.Hamiltonian(
                args.nqubits, matrix=new_hamiltonian_matrix
            )
            hamiltonians_history.extend(new_hamiltonian_matrix)
            # Initialize DBI
            dbi = DoubleBracketIteration(
                hamiltonian=new_hamiltonian,
                mode=DoubleBracketGeneratorType.single_commutator,
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
            ) = apply_dbi_steps(
                dbi=dbi, nsteps=args.dbi_steps, optimize_step=args.optimize_dbi_step
            )
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
                circ.add(gates.Unitary(gate, *range(circ.nqubits), trainable=False))
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
            if "learning_rate" in opt_options:
                opt_options["learning_rate"] *= args.decay_rate_lr

    best_loss = min(np.min(array) for array in loss_history.values())

    opt_results = partial_results[2]
    # save final results
    output_dict = vars(args)
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
    if args.optimizer not in ["sgd", "cma"]:
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
    if args.store_h:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--backend", default="qibojit", type=str, help="Qibo backend")
    parser.add_argument(
        "--platform", default=None, type=str, help="Qibo platform (used to run on GPU)"
    )
    parser.add_argument(
        "--nthreads", default=1, type=int, help="Number of threads used by the script."
    )
    parser.add_argument(
        "--optimizer", default="Powell", type=str, help="Optimizer used by VQE"
    )
    parser.add_argument(
        "--optimizer_options",
        type=str,
        help="Options to customize the optimizer training",
    )
    parser.add_argument(
        "--tol", default=TOL, type=float, help="Absolute precision to stop VQE training"
    )
    parser.add_argument(
        "--decay_rate_lr",
        default=1.0,
        type=float,
        help="Decay factor of the learning rate if sgd is used",
    )
    parser.add_argument(
        "--nqubits", default=6, type=int, help="Number of qubits for Hamiltonian / VQE"
    )
    parser.add_argument(
        "--nlayers", default=1, type=int, help="Number of layers for VQE"
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Folder where data will be stored",
    )
    parser.add_argument(
        "--nboost",
        type=int,
        default=1,
        help="Number of times the DBI is used in the new optimization routine. If 1, no optimization is run.",
    )
    parser.add_argument(
        "--boost_frequency",
        type=int,
        default=10,
        help="Number of optimization steps which separate two DBI boosting calls.",
    )
    parser.add_argument(
        "--dbi_steps",
        type=int,
        default=1,
        help="Number of DBI iterations every time the DBI is called.",
    )
    parser.add_argument("--stepsize", type=float, default=0.01, help="DBI step size.")
    parser.add_argument(
        "--optimize_dbi_step",
        type=bool,
        default=False,
        help="Set to True to hyperoptimize the DBI step size.",
    )
    parser.add_argument(
        "--store_h",
        action=argparse.BooleanOptionalAction,
        help="H is stored for each iteration",
    )
    parser.add_argument(
        "--hamiltonian",
        type=str,
        default="XXZ",
        help="Hamiltonian available in qibo.hamiltonians.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--nshots",
        type=int,
        help="number of shots",
    )
    args = parser.parse_args()
    main(args)
