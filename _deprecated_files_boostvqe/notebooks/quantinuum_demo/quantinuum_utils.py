from pytket.utils.operators import QubitPauliOperator
from pytket.partition import measurement_reduction, MeasurementBitMap, MeasurementSetup, PauliPartitionStrat
from pytket.backends.backendresult import BackendResult
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
import pandas as pd
import qnexus as qnx
from pytket.backends.status import (  # pylint: disable=unused-import
    WAITING_STATUS,
    StatusEnum,
)
from pathlib import Path
import numpy as np



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

def report(vqe_circ, gci_circ, hamiltonian, total_shots, expval_vqe, expval_gci, expval_vqe_noise, expval_gci_noise):
    energies = hamiltonian.eigenvalues()
    ground_state_energy = float(energies[0])
    vqe_energy = float(hamiltonian.expectation(vqe_circ().state()))
    gci_energy = float(hamiltonian.expectation(gci_circ().state()))
    gap = float(energies[1] - energies[0])
    vqe_energy_witness_emulator = [1 - abs(expval - ground_state_energy) / gap for expval in expval_vqe]
    vqe_noise_energy_witness_emulator = [1 - abs(expval - ground_state_energy) / gap for expval in expval_vqe_noise]
    gci_energy_witness_emulator = [1 - abs(expval - ground_state_energy) / gap for expval in expval_gci]
    gci_noise_energy_witness_emulator = [1 - abs(expval - ground_state_energy) / gap for expval in expval_gci_noise]
    return (
        dict(
            nqubits=hamiltonian.nqubits,
            runs=len(expval_vqe),
            total_shots=total_shots,
            gci_energy=float(gci_energy),
            vqe_energy=float(vqe_energy),
            vqe_energy_emulator=float(np.mean(expval_vqe)),
            gci_energy_emulator=float(np.mean(expval_gci)),
            vqe_energy_emulator_noise=float(np.mean(expval_vqe_noise)),
            gci_energy_emulator_noise=float(np.mean(expval_gci_noise)),
            vqe_delta_energy_emulator=np.std(expval_vqe),
            gci_delta_energy_emulator=np.std(expval_gci),
            vqe_delta_energy_emulator_noise=float(np.std(expval_vqe_noise)),
            gci_delta_energy_emulator_noise=float(np.std(expval_gci_noise)),
            target_energy=ground_state_energy,
            diff_vqe_target=vqe_energy - ground_state_energy,
            diff_gci_target=gci_energy - ground_state_energy,
            diff_vqe_target_emulator=float(np.mean(expval_vqe)) - ground_state_energy,
            diff_gci_target_emulator=float(np.mean(expval_gci)) - ground_state_energy,
            diff_vqe_target_emulator_noise=float(np.mean(expval_vqe_noise)) - ground_state_energy,
            diff_gci_target_emulator_noise=float(np.mean(expval_gci_noise)) - ground_state_energy,
            gap=gap,
            diff_vqe_target_perc=abs(vqe_energy - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc=abs(gci_energy - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_vqe_target_perc_emulator=abs(float(np.mean(expval_vqe)) - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc_emulator=abs(float(np.mean(expval_gci)) - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_vqe_target_perc_emulator_noise=abs(float(np.mean(expval_vqe_noise)) - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc_emulator_noise=abs(float(np.mean(expval_gci_noise)) - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            fidelity_witness_vqe=1 - abs(vqe_energy - ground_state_energy) / gap,
            fidelity_witness_gci=1 - abs(gci_energy - ground_state_energy) / gap,
            fidelity_witness_vqe_emulator=float(np.mean(vqe_energy_witness_emulator)),
            fidelity_witness_gci_emulator=float(np.mean(gci_energy_witness_emulator)),
            fidelity_witness_vqe_emulator_noise=float(np.mean(vqe_noise_energy_witness_emulator)),
            fidelity_witness_gci_emulator_noise=float(np.mean(gci_noise_energy_witness_emulator)),
            fidelity_witness_std_vqe_emulator=float(np.std(vqe_energy_witness_emulator)),
            fidelity_witness_std_gci_emulator=float(np.std(gci_energy_witness_emulator)),
            fidelity_witness_std_vqe_emulator_noise=float(np.std(vqe_noise_energy_witness_emulator)),
            fidelity_witness_std_gci_emulator_noise=float(np.std(gci_noise_energy_witness_emulator)),
        )
    )
    
    

def report_table(report):
    """
    Creates a DataFrame report including means and standard deviations.

    Parameters:
    - report (dict): Dictionary containing report metrics with means and standard deviations.

    Returns:
    - pd.DataFrame: Formatted DataFrame with metrics and their uncertainties.
    """
    # Define a helper function to format mean and std with ± sign
    def format_mean_std(mean, std, percentage=False):
        if percentage:
            return f"{mean:.2f}% ± {std:.2f}%"
        else:
            return f"{mean:.4f} ± {std:.4f}"
    # Creating a DataFrame for the table with mean ± std format
    df = pd.DataFrame({
        "Analytical": [
            f"{report['vqe_energy']:.4f}",
            f"{report['gci_energy']:.4f}",
            f"{report['diff_vqe_target']:.4f}",
            f"{report['diff_gci_target']:.4f}",
            f"{report['diff_vqe_target_perc']:.2f}%",
            f"{report['diff_gci_target_perc']:.2f}%",
            f"{report['fidelity_witness_vqe']:.4f}",
            f"{report['fidelity_witness_gci']:.4f}"
        ],
        "Emulator": [
            format_mean_std(report['vqe_energy_emulator'], report['vqe_delta_energy_emulator']),
            format_mean_std(report['gci_energy_emulator'], report['gci_delta_energy_emulator']),
            format_mean_std(report['diff_vqe_target_emulator'], report['vqe_delta_energy_emulator']),
            format_mean_std(report['diff_gci_target_emulator'], report['gci_delta_energy_emulator']),
            format_mean_std(report['diff_vqe_target_perc_emulator'], report['vqe_delta_energy_emulator']/abs(report['target_energy'])*100, percentage=True),
            format_mean_std(report['diff_gci_target_perc_emulator'], report['gci_delta_energy_emulator']/abs(report['target_energy'])*100, percentage=True),
            format_mean_std(report['fidelity_witness_vqe_emulator'], report['fidelity_witness_std_vqe_emulator']),
            format_mean_std(report['fidelity_witness_gci_emulator'], report['fidelity_witness_std_vqe_emulator'])
        ],
        "Emulator with Noise": [
            format_mean_std(report['vqe_energy_emulator_noise'], report['vqe_delta_energy_emulator_noise']),
            format_mean_std(report['gci_energy_emulator_noise'], report['gci_delta_energy_emulator_noise']),
            format_mean_std(report['diff_vqe_target_emulator_noise'], report['vqe_delta_energy_emulator_noise']),
            format_mean_std(report['diff_gci_target_emulator_noise'], report['gci_delta_energy_emulator_noise']),
            format_mean_std(report['diff_vqe_target_perc_emulator_noise'], report['vqe_delta_energy_emulator_noise']/abs(report['target_energy'])*100, percentage=True),
            format_mean_std(report['diff_gci_target_perc_emulator_noise'], report['gci_delta_energy_emulator_noise']/abs(report['target_energy'])*100, percentage=True),
            format_mean_std(report['fidelity_witness_vqe_emulator_noise'], report['fidelity_witness_std_vqe_emulator_noise']),
            format_mean_std(report['fidelity_witness_gci_emulator_noise'], report['fidelity_witness_std_gci_emulator_noise'])
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
    styled_df = df.style.set_caption(f"Boostvqe emulation results from {report['runs']} runs of {report['total_shots']} shots")
    return styled_df

def job_ref_from_path_list(path_list):
    # obtain a list of job references from a list of paths
    job_ref_list = []
    for path in path_list:
        job_ref = qnx.filesystem.load(
            path=Path.cwd() / path
        )
        job_ref_list.append(job_ref)
    return job_ref_list

def count_imcomplete_jobs(job_refs):
    """
    Prints a report of how many jobs are completed and how many are not.

    Parameters:
    - job_refs (list): A list of job reference identifiers.
    """
    # Initialize counters
    completed = 0
    not_completed = 0
    errors = 0

    # Iterate through each job reference
    for job_ref in job_refs:
        try:
            # Retrieve the job status
            status = qnx.jobs.status(job_ref)
            
            # Check if the job status is COMPLETED
            if status.status == StatusEnum.COMPLETED:
                completed += 1
            else:
                not_completed += 1

        except Exception as e:
            # Handle any exceptions that occur during status retrieval
            print(f"Error checking status for job '{job_ref}': {e}")
            errors += 1
            not_completed += 1  # Optionally consider errored jobs as not completed

    # Calculate total jobs processed
    total_jobs = len(job_refs)

    return not_completed

def load_job_results(job_refs):
    job_results = []
    for job_ref in job_refs:
        job_result = [job.download_result() for job in qnx.jobs.results(job_ref)]
        job_results.append(job_result)
    return job_results