from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np
from copy import deepcopy
import pandas as pd

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

def report_ionq(vqe_analytical, gci_analytical, ham_matrix, expval_vqe, expval_gci, expval_vqe_noise, expval_gci_noise):
    eigenvalues, eigenstates = eigh(ham_matrix)
    ground_state_energy = eigenvalues[0]
    vqe_energy = vqe_analytical
    gci_energy = gci_analytical
    gap = float(eigenvalues[1] - eigenvalues[0])
    return (
        dict(
            nqubits = int(np.log(len(ham_matrix))/np.log(2)),
            gci_energy=float(gci_energy),
            vqe_energy=float(vqe_energy),
            vqe_energy_emulator=float(expval_vqe),
            gci_energy_emulator=float(expval_gci),
            vqe_energy_emulator_noise=float(expval_vqe_noise),
            gci_energy_emulator_noise=float(expval_gci_noise),
            target_energy=ground_state_energy,
            diff_vqe_target=vqe_energy - ground_state_energy,
            diff_gci_target=gci_energy - ground_state_energy,
            diff_vqe_target_emulator=expval_vqe - ground_state_energy,
            diff_gci_target_emulator=expval_gci - ground_state_energy,
            diff_vqe_target_emulator_noise=expval_vqe_noise - ground_state_energy,
            diff_gci_target_emulator_noise=expval_gci_noise - ground_state_energy,
            gap=gap,
            diff_vqe_target_perc=abs(vqe_energy - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc=abs(gci_energy - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_vqe_target_perc_emulator=abs(expval_vqe - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc_emulator=abs(expval_gci - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_vqe_target_perc_emulator_noise=abs(expval_vqe_noise - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc_emulator_noise=abs(expval_gci_noise - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            fidelity_witness_vqe=1 - (vqe_energy - ground_state_energy) / gap,
            fidelity_witness_gci=1 - (gci_energy - ground_state_energy) / gap,
            fidelity_witness_vqe_emulator=1 - (expval_vqe - ground_state_energy) / gap,
            fidelity_witness_gci_emulator=1 - (expval_gci - ground_state_energy) / gap,
            fidelity_witness_vqe_emulator_noise=1 - (expval_vqe_noise - ground_state_energy) / gap,
            fidelity_witness_gci_emulator_noise=1 - (expval_gci_noise - ground_state_energy) / gap,
        )
    )
    
def report_table(report):
    df = pd.DataFrame({
        "Analytical": [
            report['vqe_energy'],
            report['gci_energy'],
            report['diff_vqe_target'],
            report['diff_gci_target'],
            report['diff_vqe_target_perc'],
            report['diff_gci_target_perc'],
            report['fidelity_witness_vqe'],
            report['fidelity_witness_gci']
        ],
        "Emulator": [
            report['vqe_energy_emulator'],
            report['gci_energy_emulator'],
            report['diff_vqe_target_emulator'],
            report['diff_gci_target_emulator'],
            report['diff_vqe_target_perc_emulator'],
            report['diff_gci_target_perc_emulator'],
            report['fidelity_witness_vqe_emulator'],
            report['fidelity_witness_gci_emulator']
        ],
        "Emulator with Noise": [
            report['vqe_energy_emulator_noise'],
            report['gci_energy_emulator_noise'],
            report['diff_vqe_target_emulator_noise'],
            report['diff_gci_target_emulator_noise'],
            report['diff_vqe_target_perc_emulator_noise'],
            report['diff_gci_target_perc_emulator_noise'],
            report['fidelity_witness_vqe_emulator_noise'],
            report['fidelity_witness_gci_emulator_noise']
        ]
    }, index=[
        "VQE energy",
        "GCI energy",
        "Difference to target (VQE)",
        "Difference to target (GCI)",
        "Percentage difference to target (VQE)",
        "Percentage difference to target (GCI)",
        "Fidelity witness (VQE)",
        "Fidelity witness (GCI)"
    ])
    return df