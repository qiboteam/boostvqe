from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np
from copy import deepcopy

def xxz_hamiltonian(n, delta=0.5, select=None):
    """Returns the XXZ model Hamiltonian for n qubits and a given delta in Qiskit.
    select = "XX", "YY", "ZZ" or None. None (default) returns XXZ, while the others select
    only the corresponding Pauli terms.
    """
    
    # Initialize lists to store the Pauli strings and coefficients
    pauli_strings = []
    coefficients = []

    for i in range(n):
        # XX term (X_i * X_{i+1})
        x_term = ['I'] * n
        x_term[i] = 'X'
        x_term[(i + 1)%n] = 'X'
        if select == None or select == 'XX':
            pauli_strings.append(''.join(x_term))
            coefficients.append(1.0)
        
        # YY term (Y_i * Y_{i+1})
        y_term = ['I'] * n
        y_term[i] = 'Y'
        y_term[(i + 1)%n] = 'Y'
        if select == None or select == 'YY':
            pauli_strings.append(''.join(y_term))
            coefficients.append(1.0)
        
        # ZZ term (Z_i * Z_{i+1})
        z_term = ['I'] * n
        z_term[i] = 'Z'
        z_term[(i + 1)%n] = 'Z'
        if select == None or select == 'ZZ':
            pauli_strings.append(''.join(z_term))
            coefficients.append(delta)

    # Create the SparsePauliOp object
    paulis = [Pauli(p) for p in pauli_strings]
    hamiltonian = SparsePauliOp(paulis, np.array(coefficients))
    
    return hamiltonian

def binary_code_to_index(key):
    index = 0
    size = len(key)
    for i in range(size):
        index += int(key[i])* 2 ** (size - 1 - i)
    return index

def sample_to_expectation(obs, distribution):
    # check observable diagonal
    # if (
    # np.count_nonzero(
    #     obs - np.diag(np.diagonal(obs))
    # )
    # != 0
    # ):
    #     print( "Observable is not diagonal.")
    keys = list(distribution.keys())
    freq = list(distribution.values())
    expval = 0
    for i, k in enumerate(keys):
        index = binary_code_to_index(k)
        expval += obs[index, index] * freq[i]
    return np.real(expval)

def rotate_circuit_XYZ(qc):
    """Generate 
    Args:
        qc: qiskit circuit
    Returns: modified circuits for measuring in computational, 'X', and 'Y' basis.
    """
    nqubits = qc.num_qubits
    # X
    qc_x = deepcopy(qc)
    for i in range(nqubits):
        qc_x.h(i)
    qc_x.measure_all()
    # Y
    qc_y = deepcopy(qc)
    for i in range(nqubits):
        qc_y.sdg(i)
        qc_y.h(i)
    qc_y.measure_all()
    # Z
    qc_z = deepcopy(qc)
    qc_z.measure_all()
    return [qc_x, qc_y, qc_z]

def compute_expectation_value_from_results(
    results,
    measurement,
    operator,
) -> float:
    energy = 0
    
    