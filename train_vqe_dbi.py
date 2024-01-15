import argparse
import logging
import pathlib
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.backends import GlobalBackend
from qibo.backends.numpy import NumpyBackend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from ansatze import build_circuit
from plotscripts import plot_matrix, plot_results
from utils import OPTIMIZATION_FILE, PARAMS_FILE, dump_json, json_load

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")
NSTEPS = 1
STEP = 1e-1
DBI_FILE = "dbi_matrix"
DBI_RESULTS = "dbi_output.json"


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

    ham = hamiltonians.XXZ(nqubits=data["nqubits"])  # TODO: move out
    circ = build_circuit(nqubits=data["nqubits"], nlayers=data["nlayers"])
    circ_params = np.load(f"{args.folder}/{PARAMS_FILE}")[-1]
    circ.set_parameters(circ_params)
    matrix_circ = np.matrix(GlobalBackend().to_numpy(circ.unitary()))
    matrix_circ_dagger = GlobalBackend().cast(matrix_circ.getH())
    matrix_circ = GlobalBackend().cast(matrix_circ)
    new_ham = matrix_circ_dagger @ ham.matrix @ matrix_circ

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
        step = args.dbi_stepsize

    plot_matrix(dbi.h.matrix, path=args.folder, title="Before")

    # one dbi step
    hist = []
    for _ in range(args.dbi_nsteps):
        dbi(step=step, d=dbi.diagonal_h_matrix)
        hist.append(dbi.off_diagonal_norm)

    zero_state = GlobalBackend().zero_state(data["nqubits"])
    ene_fluct_dbi = dbi.energy_fluctuation(zero_state)
    energy = dbi.h.expectation(zero_state)
    logging.info(f"Energy: {energy}")
    logging.info(f"Energy fluctuation: {ene_fluct_dbi}")
    output_dict = {
        "energy": float(energy),
        "fluctuations": float(ene_fluct_dbi),
    }
    folder = pathlib.Path(args.folder)
    np.save(file=folder / DBI_FILE, arr=dbi.h.matrix)
    dump_json(folder / DBI_RESULTS, output_dict)

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix, path=args.folder, title="After")
    plot_results(Path(args.folder), energy_dbi=(energy, ene_fluct_dbi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQE with DBI training hyper-parameters."
    )
    parser.add_argument("--folder", type=str)
    parser.add_argument("--step_opt", default=False, type=bool)
    parser.add_argument("--backend", default="qibojit", type=str)
    parser.add_argument("--platform", default=None, type=str)
    parser.add_argument("--nthreads", default=1, type=int)
    parser.add_argument("--dbi_nsteps", default=NSTEPS, type=int)
    parser.add_argument("--dbi_stepsize", default=STEP, type=float)
    main(parser.parse_args())
