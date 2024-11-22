from qibo.hamiltonians import SymbolicHamiltonian
from boostvqe.models.dbi.double_bracket_evolution_oracles import *
from functools import reduce
import numpy as np
from qibo import hamiltonians
import matplotlib.pyplot as plt

n_qubits = 3
h_coeff = 1
hamiltonian = SymbolicHamiltonian(nqubits=n_qubits)

oracle = TFIM_EvolutionOracle(h=hamiltonian, evolution_oracle_type="trotter", steps=1, B_a=0, order=2)

circuit = oracle.circuit(t_duration=1.0)

unitary = circuit.unitary()
def multikron(matrix_list):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrix_list (list): List of matrices as ``ndarray``.

    Returns:
        ndarray: Kronecker product of all matrices in ``matrix_list``.
    """
    return reduce(np.kron, matrix_list)

from numpy.linalg import norm

def our_TFIM(nqubits, h: float = 0.0, dense: bool = True, backend=None):
    def multikron(matrix_list):
        """Calculates Kronecker product of a list of matrices."""
        return reduce(np.kron, matrix_list)

    from qibo.backends import matrices

    matrix = (
        - multikron([matrices.X, matrices.X]) - h * multikron([matrices.Z, matrices.I])
    )
    terms = [hamiltonians.terms.HamiltonianTerm(matrix, i, i + 1) for i in range(nqubits - 1)]
    terms.append(hamiltonians.terms.HamiltonianTerm(matrix, nqubits - 1, 0))
    ham = SymbolicHamiltonian(backend=backend)
    ham.terms = terms
    return ham

ham = our_TFIM(nqubits=n_qubits, h=h_coeff, dense=False)
truth = ham.exp(1)
verification_norm = []
for step in range(1, 21):
    oracle = TFIM_EvolutionOracle(h=hamiltonian, evolution_oracle_type="trotter", steps=step, B_a=h_coeff, order=2)
    circuit = oracle.circuit(t_duration=1.0)
    unitary = circuit.unitary()
    verification_norm.append(norm(truth-unitary))


x = np.array([i for i in range(1, 21)])
plt.plot(x, verification_norm, 'o')
plt.title("verification of TFIM 2 CNOT implementation")
plt.xlabel("steps")
plt.ylabel("norm of difference")

plt.show()