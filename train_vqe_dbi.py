import argparse
import logging
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.backends.numpy import NumpyBackend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from ansatze import build_circuit
from plotscripts import plot_matrix, plot_results
from utils import OPTIMIZATION_FILE, PARAMS_FILE, json_load

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")
NSTEPS = 1
STEP = 1e-1
DBI_FILE = "dbi_matrix"


def main(args):
    """
    Load the VQE training and then apply `NSTEPS` of DBI.
    """
    data = json_load(f"{args.folder}/{OPTIMIZATION_FILE}")

    ham = hamiltonians.XXZ(nqubits=data["nqubits"])  # TODO: move out
    circ = build_circuit(nqubits=data["nqubits"], nlayers=data["nlayers"])
    circ_params = np.load(f"{args.folder}/{PARAMS_FILE}")[-1]
    circ.set_parameters(circ_params)
    matrix_circ = np.matrix(circ.unitary())
    new_ham = matrix_circ.getH() @ ham.matrix @ matrix_circ

    # Initialize DBI
    dbi = DoubleBracketIteration(
        hamiltonian=qibo.hamiltonians.Hamiltonian(data["nqubits"], matrix=new_ham),
        mode=DoubleBracketGeneratorType.group_commutator,
    )

    step = STEP
    plot_matrix(dbi.h.matrix, path=args.folder, title="Before")

    # one dbi step
    hist = []
    for i in range(NSTEPS):
        print(f"Step at iteration {i}/{NSTEPS}: {step}")
        dbi(step=step, d=dbi.diagonal_h_matrix)
        hist.append(dbi.off_diagonal_norm)
    zero_state = NumpyBackend().zero_state(data["nqubits"])
    ene_fluct_dbi = dbi.energy_fluctuation(zero_state)
    energy = dbi.h.expectation(zero_state)
    logging.info(f"Energy: {energy}")
    logging.info(f"Energy fluctuation: {ene_fluct_dbi}")
    np.save(file=f"{args.folder}/{DBI_FILE}", arr=dbi.h.matrix)

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix, path=args.folder, title="After")
    plot_results(Path(args.folder), energy_dbi=(energy, ene_fluct_dbi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--folder", type=str)
    main(parser.parse_args())
