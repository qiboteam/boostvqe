import argparse
import json
from pathlib import Path

import numpy as np
from qibo import hamiltonians
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.models.variational import VQE

from ansatze import build_circuit
from plotscripts import plot_loss, plot_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str)

NSTEPS = 100


def main(args):
    path = Path(args.results_path)
    res_path = path / "optimization_results.json"

    with open(res_path) as file:
        results = json.load(file)

    # construct hamiltonian according to VQE's dimensionality
    hamiltonian = hamiltonians.XXZ(nqubits=results["nqubits"])

    # exact ground state of the target hamiltonian
    ground_state = hamiltonian.ground_state()

    # itialize DBI
    dbi = DoubleBracketIteration(
        hamiltonian=hamiltonian, mode=DoubleBracketGeneratorType.canonical
    )

    # hyperoptimize step
    step = dbi.hyperopt_step(step_min=1e-4, step_max=1, max_evals=100, verbose=True)

    plot_matrix(dbi.h.matrix, title="Before")

    hist = []
    # one dbi step
    for i in range(NSTEPS):
        step = dbi.hyperopt_step(
            step_min=1e-4, step_max=1, max_evals=100, verbose=False
        )
        print(f"Step at iteration {i}/{NSTEPS}: {step}")
        dbi(step=step)
        hist.append(dbi.off_diagonal_norm)

    ene_fluct_dbi = dbi.energy_fluctuation(ground_state)
    plot_loss(loss_history=hist, title="hist")

    # ------------------------------------------------ Upload trained VQE

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix, title="After")

    # create VQE circuit with target number of qubits
    circuit = build_circuit(nqubits=results["nqubits"], nlayers=results["nlayers"])

    # upload trained model parameters
    params = np.load(path / "best_parameters.npy")
    circuit.set_parameters(params)

    # load the VQE
    vqe = VQE(circuit=circuit, hamiltonian=hamiltonian)
    ene_fluct_vqe = vqe.energy_fluctuation(ground_state)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
