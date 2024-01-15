import argparse
import logging
import pathlib

import numpy as np
import qibo
from qibo import hamiltonians

from ansatze import build_circuit
from plotscripts import plot_results
from utils import (
    FLUCTUATION_FILE,
    LOSS_FILE,
    SEED,
    create_folder,
    generate_path,
    results_dump,
    train_vqe,
)

logging.basicConfig(level=logging.INFO)


def main(args):
    """VQE training."""
    # set backend and number of classical threads
    qibo.set_backend(backend=args.backend, platform=args.platform)
    qibo.set_threads(args.nthreads)

    # setup the results folder
    logging.info("Set VQE")
    path = pathlib.Path(create_folder(generate_path(args)))

    # build hamiltonian and variational quantum circuit
    ham = hamiltonians.XXZ(nqubits=args.nqubits)
    circ = build_circuit(nqubits=args.nqubits, nlayers=args.nlayers)

    # print the circuit
    logging.info("\n" + circ.draw())
    np.random.seed(SEED)
    initial_parameters = np.random.randn(len(circ.get_parameters()))

    results, params_history, loss_list, fluctuations = train_vqe(
        circ, ham, args.optimizer, initial_parameters, args.tol
    )
    opt_results = results[2]
    # save final results
    output_dict = {
        "nqubits": args.nqubits,
        "nlayers": args.nlayers,
        "optimizer": args.optimizer,
        "best_loss": float(opt_results.fun),
        "true_ground_energy": min(ham.eigenvalues()),
        "success": opt_results.success,
        "message": opt_results.message,
        "backend": args.backend,
        "platform": args.platform,
        "tol": args.tol,
    }
    np.save(path / LOSS_FILE, arr=loss_list)
    np.save(path / FLUCTUATION_FILE, arr=fluctuations)

    logging.info("Dump the results")
    results_dump(path, params_history, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQE training hyper-parameters.")
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--optimizer", default="Powell", type=str)
    parser.add_argument("--output_folder", default=None, type=str)
    parser.add_argument("--backend", default="qibojit", type=str)
    parser.add_argument("--platform", default="dummy", type=str)
    parser.add_argument("--nthreads", default=1, type=int)
    parser.add_argument("--tol", default=None, type=float)
    args = parser.parse_args()
    main(args)
    path = generate_path(args)
    plot_results(pathlib.Path(path))
