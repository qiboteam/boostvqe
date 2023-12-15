import argparse

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from ansatze import build_circuit
from plotscripts import plot_loss, plot_matrix
from utils import OPTIMIZATION_FILE, PARAMS_FILE, json_load

qibo.set_backend("numpy")
NSTEPS = 1


def main(args):
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

    # hyperoptimize step
    # TODO: Maybe remove it
    step = dbi.hyperopt_step(step_min=1e-4, step_max=1, max_evals=100, verbose=True)
    step = 1e-1
    plot_matrix(dbi.h.matrix, title="Before")

    hist = []
    # one dbi step
    for i in range(NSTEPS):
        print(f"Step at iteration {i}/{NSTEPS}: {step}")
        dbi(step=step, d=dbi.diagonal_h_matrix)
        hist.append(dbi.off_diagonal_norm)
    zero_state = np.array([1] + [0] * (2 ** data["nqubits"] - 1))
    ene_fluct_dbi = dbi.energy_fluctuation(zero_state)
    print(ene_fluct_dbi)
    print(dbi.h.expectation(zero_state))
    plot_loss(loss_history=hist, title="hist")

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix, title="After")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--folder", type=str)
    main(parser.parse_args())
