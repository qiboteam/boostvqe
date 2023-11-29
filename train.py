import argparse

import numpy as np

from qibo.models.variational import VQE
from qibo import hamiltonians
from ansatze import build_circuit

parser = argparse.ArgumentParser(description='boostvqe hyper-parameters.')
parser.add_argument("--nqubits", default=6, type=int)
parser.add_argument("--nlayers", default=5, type=int)


def main(args):
    """VQE training and DBI boosting."""
    # build hamiltonian and variational quantum circuit
    h = hamiltonians.XXZ(nqubits=args.nqubits)
    c = build_circuit(nqubits=args.nqubits, nlayers=args.nlayers)

    # just print the circuit
    print(c.draw())
    nparams = len(c.get_parameters())

    # initialize VQE
    vqe = VQE(circuit=c, hamiltonian=h)
    initial_parameters = np.random.randn(nparams)

    result = vqe.minimize(initial_parameters)
    # visualize the results
    print(result)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
