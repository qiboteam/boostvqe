import argparse
import logging
import pathlib
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
from plotscripts import plot_results
from utils import (
    DBI_FILE,
    DBI_REULTS,
    FLUCTUATION_FILE2,
    LOSS_FILE2,
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    dump_json,
    json_load,
    train_vqe,
)

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")
NSTEPS = 1
STEP = 1e-1


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
    if args.step_opt:
        step = dbi.hyperopt_step(
            step_min=1e-4, step_max=1, max_evals=1000.0, verbose=True
        )
    else:
        step = STEP

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
    logging.info("Train VQE")
    circ.set_parameters([0] * len(circ_params))
    results, params_history, loss_list, fluctuations = train_vqe(
        circ, dbi.h, data["optimizer"], [0] * len(circ_params)
    )
    # TODO: dump results and params_history
    output_dict = {
        "energy": energy,
        "fluctuations": ene_fluct_dbi,
    }
    # Dump
    folder = pathlib.Path(args.folder)
    np.save(file=folder / LOSS_FILE2, arr=loss_list)
    np.save(file=folder / FLUCTUATION_FILE2, arr=fluctuations)
    np.save(file=folder / DBI_FILE, arr=dbi.h.matrix)
    dump_json(folder / DBI_REULTS, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--folder", type=str)
    parser.add_argument("--step_opt", default=False, type=bool)
    args = parser.parse_args()
    main(args)
    plot_results(Path(args.folder))
