from copy import deepcopy

import numpy as np
from qibo import gates, hamiltonians
from qibo.models.dbi.double_bracket import deepcopy
from qibo.symbols import Z

NITER = 20


def loss_shots(
    params,
    circ,
    ham,
    delta,
    nshots,
):
    """Train the VQE with the shots, this function is specific to the XXZ Hamiltonian"""

    # Diagonal Hamiltonian to evaluate the expactation value
    # WARNING: This is not the Hamiltonian that we want to minimize
    hamiltonian = sum(Z(i) * Z(i + 1) for i in range(circ.nqubits - 1))
    hamiltonian += Z(0) * Z(circ.nqubits - 1)
    hamiltonian = hamiltonians.SymbolicHamiltonian(hamiltonian)
    circ.set_parameters(params)
    coefficients = [1, 1, delta]
    mgates = ["X", "Y", "Z"]
    expectation_value = 0
    for i, mgate in enumerate(mgates):
        circ1 = circ.copy(deep=True)
        if mgate != "Z":  # FIXME: Bug in Qibo
            circ1.queue.pop()
            circ1.add(gates.M(*range(circ1.nqubits), basis=getattr(gates, mgate)))
        print(circ1.draw())
        print(circ1.unitary().shape)
        result = circ1(nshots=nshots)
        expectation_value += coefficients[i] * hamiltonian.expectation_from_samples(
            result.frequencies()
        )
        print(expectation_value)
    return expectation_value