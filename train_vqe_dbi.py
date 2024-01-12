import argparse
import logging
import pathlib
from pathlib import Path

import numpy as np
import qibo
from qibo.backends import construct_backend
from qibo import hamiltonians
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from ansatze import build_circuit
from plotscripts import plot_matrix, plot_results
from utils import OPTIMIZATION_FILE, PARAMS_FILE, HAMILTONIAN_FILE, dump_json, json_load

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")
DBI_FILE = "dbi_matrix"
DBI_REULTS = "dbi_output.json"


def main(args):
    """
    Load the VQE training and then apply `NSTEPS` of DBI.
    """
    #  load numpy objects
    data = json_load(f"{args.folder}/{OPTIMIZATION_FILE}")
    ham_matrix = np.load(f"{args.folder}/{HAMILTONIAN_FILE}")
    circ_params = np.load(f"{args.folder}/{PARAMS_FILE}")[args.starting_from_epoch]

    # construct backend 
    backend = construct_backend(backend=data["backend"], platform=data["platform"])

    # loading VQE circuit and hamiltonian
    ham = hamiltonians.Hamiltonian(nqubits=data["nqubits"], matrix=ham_matrix)
    circ = build_circuit(nqubits=data["nqubits"], nlayers=data["nlayers"])
    circ.set_parameters(circ_params)

    # new hamiltonian shape after applying VQE unitary
    matrix_circ = np.matrix(circ.unitary())
    new_ham = matrix_circ.getH() @ ham.matrix @ matrix_circ

    # Initialize DBI
    dbi = DoubleBracketIteration(
        hamiltonian=qibo.hamiltonians.Hamiltonian(data["nqubits"], matrix=new_ham),
        mode=DoubleBracketGeneratorType.group_commutator,
    )

    if args.step_opt:
        step = dbi.hyperopt_step(
            step_min=1e-4, step_max=1, max_evals=1000.0, verbose=True
        )
    else:
        step = args.stepsize

    plot_matrix(dbi.h.matrix, path=args.folder, title="Before")

    # one dbi step
    hist = []
    for i in range(NSTEPS):
        print(f"Step at iteration {i}/{NSTEPS}: {step}")
        dbi(step=step, d=dbi.diagonal_h_matrix)
        hist.append(dbi.off_diagonal_norm)

    zero_state = backend.zero_state(data["nqubits"])
    ene_fluct_dbi = dbi.energy_fluctuation(zero_state)
    energy = dbi.h.expectation(zero_state)
    logging.info(f"Energy: {energy}")
    logging.info(f"Energy fluctuation: {ene_fluct_dbi}")
    output_dict = {
        "energy": energy,
        "fluctuations": ene_fluct_dbi,
    }
    folder = pathlib.Path(args.folder)
    np.save(file=folder / DBI_FILE, arr=dbi.h.matrix)
    dump_json(folder / DBI_REULTS, output_dict)

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix, path=args.folder, title="After")
    plot_results(Path(args.folder), energy_dbi=(energy, ene_fluct_dbi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        help="Path to the folder in which training data are saved"
        )
    parser.add_argument(
        "--starting_from_epoch", 
        type=int, 
        default=-1, 
        help="From which training epoch loading the parameters."
        )
    parser.add_argument(
        "--boost_steps", 
        type=int, 
        default=None, 
        help="Number of times the DBI procedure is going to be used."
        )
    parser.add_argument(
        "--dbi_steps", 
        type=int, 
        default=1,
        help="Number of DBI iterations every time the DBI is called."
        )
    parser.add_argument(
        "--stepsize", 
        type=float, 
        default=0.01,
        help="DBI step size."
        )
    parser.add_argument(
        "--step_opt", 
        type=bool, 
        default=False,
        help="Set to True to hyperoptimize the DBI step size."
        )
    main(parser.parse_args())
