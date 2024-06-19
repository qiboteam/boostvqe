import json
import logging
from pathlib import Path

import numpy as np
from qibo import get_backend

from boostvqe.ansatze import VQE, compute_gradients
import json
import time
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians, set_backend
qibo.set_backend("numpy")
from boostvqe.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    )

from boostvqe.models.dbi.group_commutator_iteration_transpiler import *
from boostvqe.models.dbi.double_bracket_evolution_oracles import *

from boostvqe.ansatze import VQE, build_circuit

from qibo import symbols, hamiltonians
from copy import deepcopy
from boostvqe.compiling_XXZ import *

import matplotlib.pyplot as plt






OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.pdf"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations"
LOSS_FILE = "energies"
GRADS_FILE = "gradients"
HAMILTONIAN_FILE = "hamiltonian_matrix.npz"
SEED = 42
TOL = 1e-10
DBI_ENERGIES = "dbi_energies"
DBI_FLUCTUATIONS = "dbi_fluctuations"
DBI_STEPS = "dbi_steps"
DBI_D_MATRIX = "dbi_d_matrices"


logging.basicConfig(level=logging.INFO)


def generate_path(args) -> str:
    """Generate path according to job parameters"""
    if args.output_folder is None:
        output_folder = "results"
    else:
        output_folder = args.output_folder
    return f"./{output_folder}/{args.optimizer}_{args.nqubits}q_{args.nlayers}l_{args.seed}"


def create_folder(path: str) -> Path:
    """Create folder and returns path"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    """Dump"""
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    json_file = Path(f"{path}/{OPTIMIZATION_FILE}")
    dump_json(json_file, output_dict)


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=4))


def json_load(path: str):
    f = open(path)
    return json.load(f)


def callback_energy_fluctuations(params, circuit, hamiltonian):
    """Evaluate the energy fluctuations"""
    circ = circuit.copy(deep=True)
    circ.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circ)
    final_state = result.state()
    return hamiltonian.energy_fluctuation(final_state)


def train_vqe(
    circ,
    ham,
    optimizer,
    initial_parameters,
    tol,
    loss,
    niterations=None,
    nmessage=1,
    training_options=None,
):
    """Helper function which trains the VQE according to `circ` and `ham`."""
    params_history, loss_list, fluctuations, grads_history = (
        [],
        [],
        [],
        [],
    )

    if training_options is None:
        options = {}
    else:
        options = training_options

    circ.set_parameters(initial_parameters)

    vqe = VQE(
        circuit=circ,
        hamiltonian=ham,
    )

    def callbacks(
        params,
        vqe=vqe,
        loss_list=loss_list,
        loss_fluctuation=fluctuations,
        params_history=params_history,
        grads_history=grads_history,
        loss=loss,
    ):
        """
        Callback function that updates the energy, the energy fluctuations and
        the parameters lists.
        """
        energy = loss(params, vqe.circuit, vqe.hamiltonian)
        loss_list.append(float(energy))
        loss_fluctuation.append(
            callback_energy_fluctuations(params, vqe.circuit, vqe.hamiltonian)
        )
        params_history.append(params)
        grads_history.append(
            compute_gradients(
                parameters=params, circuit=circ.copy(deep=True), hamiltonian=ham
            )
        )

        iteration_count = len(loss_list) - 1

        if niterations is not None and iteration_count % nmessage == 0:
            logging.info(f"Optimization iteration {iteration_count}/{niterations}")
            logging.info(f"Loss {energy:.5}")

    callbacks(initial_parameters)
    logging.info("Minimize the energy")

    results = vqe.minimize(
        initial_parameters,
        method=optimizer,
        callback=callbacks,
        tol=tol,
        loss_func=loss,
        options=options,
    )

    return (
        results,
        params_history,
        loss_list,
        grads_history,
        fluctuations,
        vqe,
    )


def rotate_h_with_vqe(hamiltonian, vqe):
    """Rotate `hamiltonian` using the unitary representing the `vqe`."""
    # inherit backend from hamiltonian and circuit from vqe
    backend = hamiltonian.backend
    circuit = vqe.circuit
    # create circuit matrix and compute the rotation
    matrix_circ = np.matrix(backend.to_numpy(circuit.fuse().unitary()))
    matrix_circ_dagger = backend.cast(matrix_circ.getH())
    matrix_circ = backend.cast(matrix_circ)
    new_hamiltonian = np.matmul(
        matrix_circ_dagger, np.matmul(hamiltonian.matrix, matrix_circ)
    )
    return new_hamiltonian


def apply_dbi_steps(dbi, nsteps, stepsize=0.01, optimize_step=False):
    """Apply `nsteps` of `dbi` to `hamiltonian`."""
    step = stepsize
    energies, fluctuations, hamiltonians, steps, d_matrix = [], [], [], [], []
    logging.info(f"Applying {nsteps} steps of DBI to the given hamiltonian.")
    operators = []
    for _ in range(nsteps):
        if optimize_step:
            # Change logging level to reduce verbosity
            logging.getLogger().setLevel(logging.WARNING)
            step = dbi.hyperopt_step(
                step_min=1e-4, step_max=.01, max_evals=50, verbose=True
            )
            # Restore the original logging level
            logging.getLogger().setLevel(logging.INFO)
        operators.append(dbi(step=step, d=dbi.diagonal_h_matrix))
        steps.append(step)
        d_matrix.append(np.diag(dbi.diagonal_h_matrix))
        zero_state = np.transpose([dbi.h.backend.zero_state(dbi.h.nqubits)])

        energies.append(dbi.h.expectation(zero_state))
        fluctuations.append(dbi.energy_fluctuation(zero_state))
        hamiltonians.append(dbi.h.matrix)

        logging.info(f"DBI energies: {energies}")
    return hamiltonians, energies, fluctuations, steps, d_matrix, operators

def test_gc_step(dbi):
    return None

def apply_gci_circuits(dbi, nsteps, stepsize=0.01, optimize_step=False):
    """Apply `nsteps` of `dbi` to `hamiltonian`."""
    step = stepsize
    energies, fluctuations, hamiltonians, steps, d_matrix = [], [], [], [], []
    operators = []
    for _ in range(nsteps):
        if optimize_step:
            # Change logging level to reduce verbosity
            logging.getLogger().setLevel(logging.WARNING)
            step = dbi.hyperopt_step(
                step_min=1e-4, step_max=.01, max_evals=50, verbose=True
            )
            # Restore the original logging level
            logging.getLogger().setLevel(logging.INFO)
        operators.append(dbi(step=step, d=dbi.diagonal_h_matrix))
        steps.append(step)
        d_matrix.append(np.diag(dbi.diagonal_h_matrix))
        zero_state = np.transpose([dbi.h.backend.zero_state(dbi.h.nqubits)])

        energies.append(dbi.h.expectation(zero_state))
        fluctuations.append(dbi.energy_fluctuation(zero_state))
        hamiltonians.append(dbi.h.matrix)

        logging.info(f"DBI energies: {energies}")
    return hamiltonians, energies, fluctuations, steps, d_matrix, operators


def print_vqe_comparison_report(gci):
    gci_loss = gci.loss()
    print(f"VQE energy is {round(gci.vqe_energy,5)} and the DBQA yields {round(gci_loss,5)}. \n\
The target energy is {round(gci.h.target_energy,5)} which means the difference is for VQE \
    {round(gci.vqe_energy-gci.h.target_energy,5)} and of the DBQA {round(gci_loss-gci.h.target_energy,5)} \
        which can be compared to the spectral gap {round(gci.h.gap,5)}.\n\
The relative difference is for VQE {round(abs(gci.vqe_energy-gci.h.target_energy)/abs(gci.h.target_energy)*100,5)}% \
    and for DBQA {round(abs(gci_loss-gci.h.target_energy)/abs(gci.h.target_energy)*100,5)}%.\
The energetic fidelity witness for the ground state for the\n\
      VQE is {round(1- abs(gci.vqe_energy-gci.h.target_energy)/abs(gci.h.gap),5)} \n\
        and DBQA {round(1- abs(gci_loss-gci.h.target_energy)/abs(gci.h.gap),5)}\
")


def initialize_gci_from_vqe( path = "../results/vqe_data/with_params/10q7l/sgd_10q_7l_42/",
                            target_epoch = 2000,
                            dbi_steps = 1,
                            mode_dbr = DoubleBracketRotationType.group_commutator_third_order_reduced):

    # upload system configuration and parameters for all the training
    with open(path + "optimization_results.json") as file:
        config = json.load(file)

    losses = dict(np.load(path + "energies.npz"))["0"]
    params = np.load(path + f"parameters/params_ite{target_epoch}.npy")

    nqubits = config["nqubits"]
    # build circuit, hamiltonian and VQE
    circuit = build_circuit(nqubits, config["nlayers"], "numpy")       
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5)

    vqe = VQE(circuit, hamiltonian)
    # set target parameters into the VQE
    vqe.circuit.set_parameters(params)

    eo_xxz = XXZ_EvolutionOracle(nqubits, steps = 1, order = 2)
    # implement the rotate by VQE on the level of circuits
    fsoe  = VQERotatedEvolutionOracle(eo_xxz, vqe)
    # init gci with the vqe-rotated hamiltonian
    gci  = GroupCommutatorIterationWithEvolutionOracles(input_hamiltonian_evolution_oracle=fsoe, 
            mode_double_bracket_rotation=mode_dbr)
    
    eigenergies = hamiltonian.eigenvalues()
    target_energy = np.min(eigenergies)
    gci.h.target_energy = target_energy
    eigenergies.sort()
    gap = eigenergies[1] - target_energy
    gci.h.gap = gap

    gci.vqe_energy = hamiltonian.expectation(vqe.circuit().state())

   
    b_list = [1+np.sin(x/3)for x in range(10)]
    gci.eo_d = MagneticFieldEvolutionOracle(b_list,name = "D(B = 1+sin(x/3))")
    gci.default_step_grid = np.linspace(0.003,0.004,10)
    return gci


def select_recursion_step_circuit(gci, 
                    mode_dbr_list = [DoubleBracketRotationType.group_commutator_third_order_reduced], 
                    eo_d = None,
                    step_grid = np.linspace(1e-3,3e-2,10),
                    please_be_visual = False):
    """ Returns: circuit of the step, code of the strategy"""

    if eo_d is None:
        eo_d = gci.eo_d    
    
    minimal_losses = []
    all_losses = []
    minimizer_s = []
    for i,mode in enumerate(mode_dbr_list):

        gci.mode_double_bracket_rotation = mode
        s, l, ls = gci.choose_step(d = eo_d,step_grid = step_grid, mode_dbr = mode)
        #here optimize over gradient descent
        minimal_losses.append(l)
        minimizer_s.append(s)

        if please_be_visual:
            plt.plot(step_grid,ls)
            plt.yticks([ls[0],l, ls[-1]])
            plt.xticks([step_grid[0],s,step_grid[-1]])
            plt.title(mode.name)
            plt.show()
            

    minimizer_dbr_id = np.argmin(minimal_losses)
    
    return mode_dbr_list[minimizer_dbr_id], minimizer_s[minimizer_dbr_id], eo_d

def execute_selected_recursion_step( gci, mode_dbr, minimizer_s, eo_d, please_be_verbose = False ):
    gci.mode_double_bracket_rotation = mode_dbr
    if please_be_verbose:
        gci.print_gate_count_report()
        print_vqe_comparison_report(gci)
    gci(minimizer_s, eo_d)
    if please_be_verbose:
        gci.print_gate_count_report() 
        print_vqe_comparison_report(gci)
    return gci

