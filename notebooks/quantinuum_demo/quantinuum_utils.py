from pytket.utils.operators import QubitPauliOperator
from pytket.partition import measurement_reduction, MeasurementBitMap, MeasurementSetup, PauliPartitionStrat
from pytket.backends.backendresult import BackendResult
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
import pandas as pd


def compute_expectation_paulistring(
    distribution: dict[tuple[int, ...], float], bitmap: MeasurementBitMap
) -> float:
    '''
    This function assumes that the bitmap is in the correct measurement basis
    and evaluates Pauli operators composed of Pauli.Z and Pauli.I.
    It calculates the expectation by counting the parity of the qubits being
    flipped.
    '''
    value = 0
    for bitstring, probability in distribution.items():
        value += probability * (sum(bitstring[i] for i in bitmap.bits) % 2)
    return ((-1) ** bitmap.invert) * (-2 * value + 1)


def compute_expectation_value_from_results(
    results: list[BackendResult],
    measurement_setup: MeasurementSetup,
    operator: QubitPauliOperator,
) -> float:
    '''
    This function loops with the measurement_setup corresponding to the
    hamiltonian, select the corresponding string_coef, results index and
    calculates the total expectation of the input hamiltonian.
    '''
    energy = 0
    for pauli_string, bitmaps in measurement_setup.results.items():
        string_coeff = operator.get(pauli_string, 0.0)
        if string_coeff != 0:
            for bm in bitmaps:
                index = bm.circ_index
                distribution = results[index].get_distribution()
                value = compute_expectation_paulistring(distribution, bm)
                energy += complex(value * string_coeff).real
    return energy

def create_qubit_pauli_string(nqubits, specify_ls, coef):
    '''
    specify_ls: {index:Pauli.X/Y/Z}
    '''
    term = {}
    specified_ids = list(specify_ls.keys())
    for i in range(nqubits):
        if i in specified_ids:
            term.update({Qubit(i):specify_ls[i]})
        else:
            term.update({Qubit(i):Pauli.I})

    return {QubitPauliString(term):coef}

def report(vqe_circ, gci_circ, hamiltonian, expval_vqe, expval_gci, expval_vqe_noise, expval_gci_noise):
    energies = hamiltonian.eigenvalues()
    ground_state_energy = float(energies[0])
    vqe_energy = float(hamiltonian.expectation(vqe_circ().state()))
    gci_energy = float(hamiltonian.expectation(gci_circ().state()))
    gap = float(energies[1] - energies[0])
    return (
        dict(
            nqubits=hamiltonian.nqubits,
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
            fidelity_witness_vqe=1 - abs(vqe_energy - ground_state_energy) / gap,
            fidelity_witness_gci=1 - abs(gci_energy - ground_state_energy) / gap,
            fidelity_witness_vqe_emulator=1 - abs(expval_vqe - ground_state_energy) / gap,
            fidelity_witness_gci_emulator=1 - abs(expval_gci - ground_state_energy) / gap,
            fidelity_witness_vqe_emulator_noise=1 - abs(expval_vqe_noise - ground_state_energy) / gap,
            fidelity_witness_gci_emulator_noise=1 - abs(expval_gci_noise - ground_state_energy) / gap,
        )
    )
    
def report_table(report):
    # Creating a DataFrame for the table
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