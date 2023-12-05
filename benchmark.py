import argparse
import json 
from pathlib import Path

from qibo.models.dbi.double_bracket import DoubleBracketGeneratorType, DoubleBracketIteration
from qibo import hamiltonians

from pltoscripts import plot_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str)


def main(args):
    path = Path(args.results_path)
    res_path = path / "optimization_results.json"

    with open(res_path, "r") as file:
        results = json.load(file)
    
    # construct hamiltonian according to VQE's dimensionality
    hamiltonian = hamiltonians.XXZ(nqubits=results["nqubits"])

    # exact ground state of the target hamiltonian
    ground_state = hamiltonian.ground_state()

    # itialize DBI
    dbi = DoubleBracketIteration(
        hamiltonian=hamiltonian, 
        mode=DoubleBracketGeneratorType.canonical
    )

    # hyperoptimize step
    step = dbi.hyperopt_step(
        step_min = 1e-4,
        step_max = 1,
        max_evals = 1000,
        verbose = True
    )

    # one dbi step
    dbi(step=step)

    # plot hamiltonian's matrix
    plot_matrix(dbi.h.matrix)

    # TODO: upload VQE configuration
    # TODO: compute energy fluctuation of VQE's hamiltonian and DBI's one over 
    #       the ground state of `hamiltonian`
    # TODO: check numerical fluctuations. In this case we have a negative zero 
    #       under square root (it crashes)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
