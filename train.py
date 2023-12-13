import argparse
import os
import json

import numpy as np

import qibo
from qibo.models.variational import VQE
from qibo import hamiltonians

from ansatze import build_circuit

parser = argparse.ArgumentParser(description='VQE training hyper-parameters.')
parser.add_argument("--nqubits", default=6, type=int)
parser.add_argument("--nlayers", default=5, type=int)
parser.add_argument("--optimizer", default="Powell", type=str)
parser.add_argument("--output_folder", default=None, type=str)
parser.add_argument("--backend", default="qibojit", type=str)
parser.add_argument("--platform", default=None, type=str)
parser.add_argument("--nthreads", default=1, type=int)

def main(args):
    """VQE training and DBI boosting."""
    # set backend and number of classical threads
    qibo.set_backend(backend=args.backend, platform=args.platform)
    qibo.set_threads(args.nthreads)

    # setup the results folder
    if args.output_folder is None:
        path = f"./results/{args.optimizer}_{args.nqubits}q_{args.nlayers}l"
    else:
        path = args.output_folder
    os.system(f"mkdir {path}")

    # build hamiltonian and variational quantum circuit
    h = hamiltonians.XXZ(nqubits=args.nqubits)
    c = build_circuit(nqubits=args.nqubits, nlayers=args.nlayers)

    # just print the circuit
    print(c.draw())
    nparams = len(c.get_parameters())

    # initialize VQE
    vqe = VQE(circuit=c, hamiltonian=h)

    # fix numpy seed to ensure replicability of the experiment
    np.random.seed(42)
    initial_parameters = np.random.randn(nparams)

    results = vqe.minimize(initial_parameters, method=args.optimizer)
    opt_results = results[2]

    # save final results
    np.save(file=f"{path}/best_parameters.npy", arr=results[1])
    output_dict = {
        "nqubits": args.nqubits,
        "nlayers": args.nlayers,
        "optimizer": args.optimizer,
        "best_loss": float(opt_results.fun),
        "success": opt_results.success,
        "message": opt_results.message,
        "backend": args.backend,
        "platform": args.platform
    }

    with open(f"{path}/optimization_results.json", "w") as file:
        json.dump(output_dict, file, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
