import numpy as np
from qibo import gates
from qibo.symbols import Z


def train_vqe(
    circ,
    delta,
    mode,
    optimizer,
    initial_parameters,
    tol,
    niterations,
    nshots,
    nmessage=1,
):
    """Train the VQE with the shots, this function is specific to the XXZ Hamiltonian"""

    hamiltonian = sum(Z(i) * Z(i + 1) for i in range(3))
    hamiltonian += Z(0) * Z(3)
    hamiltonian = hamiltonians.SymbolicHamiltonian(hamiltonian)
    coefficients = [1, 1, delta]

    for epoch in range(niterations):
        mgates = ["X", "Y", "Z"]
        expectation_value = 0
        for i, mgate in enumerate(mgates):
            circ.queue.pop()
            circ.add(gates.M(*range(circ.nqubits), basis=getattr(gates, mgate)))
            result = circuit(nshots=nshots)
            expectation_value += coefficients[i] * hamiltonian.expectation_from_samples(
                result.frequencies()
            )
