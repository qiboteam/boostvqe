import argparse
import logging
import pathlib
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.backends import GlobalBackend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.models.variational import VQE

from ansatze import build_circuit
from plotscripts import plot_results
from utils import (
    DBI_FILE,
    DBI_RESULTS,
    FLUCTUATION_FILE2,
    HAMILTONIAN_FILE,
    LOSS_FILE2,
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    TOL,
    dump_json,
    json_load,
    train_vqe,
)

logging.basicConfig(level=logging.INFO)


def main(args):
    """
    Load the VQE training and then apply `NSTEPS` of DBI.
    """
    if args.platform is not None:
        qibo.set_backend(backend=args.backend, platform=args.platform)
    else:
        qibo.set_backend(backend=args.backend)
        args.platform = GlobalBackend().platform

    data = json_load(f"{args.folder}/{OPTIMIZATION_FILE}")
    ham_matrix = np.load(f"{args.folder}/{HAMILTONIAN_FILE}")
    circ_params = np.load(f"{args.folder}/{PARAMS_FILE}")[args.starting_from_epoch]

    # loading VQE circuit and hamiltonian
    ham = hamiltonians.Hamiltonian(nqubits=data["nqubits"], matrix=ham_matrix)
    circ = build_circuit(nqubits=data["nqubits"], nlayers=data["nlayers"])
    circ.set_parameters(circ_params)
    backend = ham.backend

    vqe = VQE(circuit=circ, hamiltonian=ham)
    loss_list, fluctuations, params_history = [], [], []

    # loop over the number of boosts
    for i in range(args.optimization_steps):
        # reset iterations counter
        opt_iteration = 0

        logging.info(f"Applying the DBI boosting N° {i+1} of {args.optimization_steps}")

        # update hamiltonian
        matrix_circ = np.matrix(backend.to_numpy(circ.unitary()))
        matrix_circ_dagger = backend.cast(matrix_circ.getH())
        matrix_circ = backend.cast(matrix_circ)
        new_ham = matrix_circ_dagger @ ham.matrix @ matrix_circ

        # Initialize DBI
        dbi = DoubleBracketIteration(
            hamiltonian=qibo.hamiltonians.Hamiltonian(data["nqubits"], matrix=new_ham),
            mode=DoubleBracketGeneratorType.group_commutator,
        )

        step = args.stepsize

        # apply DBI dbi_steps times
        for j in range(args.dbi_steps):
            if args.step_opt:
                step = dbi.hyperopt_step(
                    step_min=1e-4, step_max=1, max_evals=1000, verbose=True
                )
            logging.info(
                f"Applying DBI step {j}/{args.dbi_steps} with step size: {step}"
            )
            dbi(step=step, d=dbi.diagonal_h_matrix)

        print(args)

        if args.optimization_steps > 1:
            while opt_iteration <= args.boost_frequency:
                initial_parameters = circ.get_parameters()
                results = vqe.minimize(
                    initial_parameters,
                    method=args.optimizer,
                    callback=callbacks_builder,
                    tol=TOL,
                )

    zero_state = backend.zero_state(data["nqubits"])
    ene_fluct_dbi = dbi.energy_fluctuation(zero_state)
    energy = dbi.h.expectation(zero_state)
    logging.info(f"Energy: {energy}")
    logging.info(f"Energy fluctuation: {ene_fluct_dbi}")

    # TODO: dump results and params_history
    output_dict = {
        "energy": float(energy),
        "fluctuations": float(ene_fluct_dbi),
        "tol": TOL,
    }
    # Dump
    folder = pathlib.Path(args.folder)
    np.save(file=folder / LOSS_FILE2, arr=loss_list)
    np.save(file=folder / FLUCTUATION_FILE2, arr=fluctuations)
    np.save(file=folder / DBI_FILE, arr=dbi.h.matrix)
    dump_json(folder / DBI_RESULTS, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--backend", default="qibojit", type=str)
    parser.add_argument("--platform", default=None, type=str)
    parser.add_argument("--nthreads", default=1, type=int)
    parser.add_argument(
        "--folder", type=str, help="Path to the folder in which training data are saved"
    )
    parser.add_argument(
        "--starting_from_epoch",
        type=int,
        default=-1,
        help="From which training epoch loading the parameters.",
    )
    parser.add_argument(
        "--optimization_steps",
        type=int,
        default=1,
        help="Number of times the DBI is used in the new optimization routine. If 1, no optimization is run.",
    )
    parser.add_argument(
        "--boost_frequency",
        type=int,
        default=None,
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
        "--step_opt",
        type=bool,
        default=False,
        help="Set to True to hyperoptimize the DBI step size.",
    )
    args = parser.parse_args()
    main(args)
    plot_results(Path(args.folder))
