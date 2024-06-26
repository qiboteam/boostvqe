import json
import logging
import time
from pathlib import Path

import numpy as np
import qibo
from qibo import get_backend, hamiltonians, set_backend

from boostvqe.ansatze import VQE, compute_gradients

qibo.set_backend("numpy")
from copy import deepcopy

import matplotlib.pyplot as plt
from qibo import hamiltonians, symbols

from boostvqe.ansatze import VQE, build_circuit
from boostvqe.compiling_XXZ import *
from boostvqe.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from boostvqe.models.dbi.double_bracket_evolution_oracles import *
from boostvqe.models.dbi.group_commutator_iteration_transpiler import *

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.pdf"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations"
LOSS_FILE = "energies"
GRADS_FILE = "gradients"
HAMILTONIAN_FILE = "hamiltonian_matrix.npz"
SEED = 42
DELTA = 0.5
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
                step_min=1e-4, step_max=0.01, max_evals=50, verbose=True
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


def initialize_gci_from_vqe(
    nqubits=10,
    nlayers=7,
    seed=42,
    target_epoch=2000,
    mode_dbr=DoubleBracketRotationType.group_commutator_third_order_reduced,
):
    path = f"../results/vqe_data/with_params/{nqubits}q{nlayers}l/sgd_{nqubits}q_{nlayers}l_{seed}/"

    # upload system configuration and parameters for all the training
    with open(path + "optimization_results.json") as file:
        config = json.load(file)

    losses = dict(np.load(path + "energies.npz"))["0"]
    params = np.load(path + f"parameters/params_ite{target_epoch}.npy")

    nqubits = config["nqubits"]
    # build circuit, hamiltonian and VQE
    circuit = build_circuit(nqubits, config["nlayers"])
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5)

    vqe = VQE(circuit, hamiltonian)
    # set target parameters into the VQE
    vqe.circuit.set_parameters(params)

    eo_xxz = XXZ_EvolutionOracle(nqubits, steps=1, order=2)
    # implement the rotate by VQE on the level of circuits
    fsoe = VQERotatedEvolutionOracle(eo_xxz, vqe)
    # init gci with the vqe-rotated hamiltonian
    gci = VQEBoostingGroupCommutatorIteration(
        input_hamiltonian_evolution_oracle=fsoe, mode_double_bracket_rotation=mode_dbr
    )

    return gci


def select_recursion_step_circuit(
    gci,
    mode_dbr_list=[DoubleBracketRotationType.group_commutator_third_order_reduced],
    eo_d=None,
    step_grid=np.linspace(1e-3, 3e-2, 10),
    please_be_visual=False,
    save_path=None,
):
    """Returns: circuit of the step, code of the strategy"""

    if eo_d is None:
        eo_d = gci.eo_d

    minimal_losses = []
    all_losses = []
    minimizer_s = []
    for i, mode in enumerate(mode_dbr_list):
        gci.mode_double_bracket_rotation = mode
        s, l, ls = gci.choose_step(d=eo_d, step_grid=step_grid, mode_dbr=mode)
        # here optimize over gradient descent
        minimal_losses.append(l)
        minimizer_s.append(s)

        if please_be_visual:
            plt.plot(step_grid, ls)
            plt.yticks([ls[0], l, ls[-1]])
            plt.xticks([step_grid[0], s, step_grid[-1]])
            plt.title(mode.name)
            if save_path is None:
                save_path = f"{gci.path}figs/gci_boost_{gci.mode_double_bracket_rotation}_s={s}.pdf"
            if gci.please_save_fig_to_pdf is True:
                plt.savefig(save_path, format="pdf")
            plt.show()

    minimizer_dbr_id = np.argmin(minimal_losses)

    return mode_dbr_list[minimizer_dbr_id], minimizer_s[minimizer_dbr_id], eo_d


from boostvqe.models.dbi.utils_gci_optimization import *


def select_recursion_step_gd_circuit(
    gci,
    mode_dbr_list=[DoubleBracketRotationType.group_commutator_third_order],
    eo_d=None,
    step_grid=np.linspace(1e-5, 3e-2, 30),
    lr_range=(1e-3, 1),
    threshold=1e-4,
    max_eval_gd=30,
    nmb_gd_epochs=0,
    please_be_visual=False,
    please_be_verbose=True,
    save_path="gci_step",
):
    """Returns: circuit of the step, code of the strategy"""

    if eo_d is None:
        eo_d = gci.eo_d
    if eo_d.name == "B Field":
        n_local = 1
        params = eo_d.b_list
    elif eo_d.name == "H_ClassicalIsing(B,J)":
        n_local = 2
        params = eo_d.b_list + eo_d.j_list
    else:
        raise_error(ValueError, "Evolution oracle type not supported.")

    minimal_losses = []
    minimizer_s = []
    minimizer_eo_d = []
    for i, mode in enumerate(mode_dbr_list):
        gci.mode_double_bracket_rotation = mode
        # returns min_s, min_loss, loss_list
        s, l, ls = gci.choose_step(d=eo_d, step_grid=step_grid, mode_dbr=mode)

        for epoch in range(nmb_gd_epochs):
            ls = []
            s_min, s_max = step_grid[0], step_grid[-1]
            lr_min, lr_max = lr_range[0], lr_range[-1]
            eo_d, s, l, eval_dict, params, best_lr = choose_gd_params(
                gci,
                n_local,
                params,
                l,
                s,
                s_min,
                s_max,
                lr_min,
                lr_max,
                threshold,
                max_eval_gd,
            )

        minimal_losses.append(l)
        minimizer_s.append(s)
        minimizer_eo_d.append(eo_d)

        if please_be_visual:
            if not nmb_gd_epochs:
                plt.plot(step_grid, ls)
                plt.yticks([ls[0], l, ls[-1]])
                plt.xticks([step_grid[0], s, step_grid[-1]])
            else:
                plot_lr_s_loss(eval_dict)
            if save_path is None:
                save_path = f"{gci.path}figs/gci_boost_{gci.mode_double_bracket_rotation}_s={s}.pdf"
            if gci.please_save_fig_to_pdf is True:
                plt.savefig(save_path, format="pdf")
            plt.show()

    minimizer_dbr_id = np.argmin(minimal_losses)
    if please_be_verbose:
        print(
            f"Just finished the selection: better loss {minimal_losses[minimizer_dbr_id]} for mode {mode_dbr_list[minimizer_dbr_id]},\
                  with duration s={minimizer_s[minimizer_dbr_id]}, and eo_d name = {minimizer_eo_d[minimizer_dbr_id].name}"
        )
    return (
        mode_dbr_list[minimizer_dbr_id],
        minimizer_s[minimizer_dbr_id],
        minimal_losses[minimizer_dbr_id],
        minimizer_eo_d[minimizer_dbr_id],
    )


def execute_gci_boost(
    nqubits=10,
    nlayers=7,
    seed=42,
    target_epoch=200,
    nmb_gci_steps=1,
    nmb_gd_epochs=0,
    eo_d=None,
    mode_dbr_list=[  # DoubleBracketRotationType.group_commutator_reduced,
        # DoubleBracketRotationType.group_commutator_mix_twice,
        # DoubleBracketRotationType.group_commutator_reduced_twice,
        DoubleBracketRotationType.group_commutator_third_order_reduced,
        # DoubleBracketRotationType.group_commutator_third_order_reduced_twice
    ],
    please_be_verbose=False,
    please_be_visual=False,
):
    if please_be_verbose:
        print(f"Initilizing gci:\n")
    gci = initialize_gci_from_vqe(
        nqubits=nqubits, nlayers=nlayers, seed=seed, target_epoch=target_epoch
    )
    if eo_d is not None:
        gci.eo_d = eo_d

    if please_be_verbose:
        print(
            f"The gci mode is {gci.mode_double_bracket_rotation} rotation with {gci.eo_d.name} as the oracle.\n"
        )
        print_vqe_comparison_report(gci)
    boosting_callback_data = {}
    for gci_step_nmb in range(nmb_gci_steps):
        mode_dbr, minimizer_s, minimal_loss, eo_d = select_recursion_step_gd_circuit(
            gci,
            mode_dbr_list=mode_dbr_list,
            step_grid=np.linspace(1e-5, 2e-2, 30),
            lr_range=(1e-3, 1),
            nmb_gd_epochs=nmb_gd_epochs,
            threshold=1e-4,
            max_eval_gd=30,
            please_be_visual=please_be_visual,
            save_path="gci_step",
        )

        gci.mode_double_bracket_rotation = mode_dbr
        gci.eo_d = eo_d
        gci(minimizer_s)

        if please_be_verbose:
            print(f"Executing gci step {gci_step_nmb+1}:\n")
            print(
                f"The selected data is {gci.mode_double_bracket_rotation} rotation with {gci.eo_d.name} for the duration s = {minimizer_s}."
            )
            print("--- the report after execution:\n")
            print_vqe_comparison_report(gci)
            print("==== the execution report ends here")
        boosting_callback_data[gci_step_nmb] = gci.get_vqe_boosting_data()

    return gci, boosting_callback_data


def get_eo_d_initializations(nqubits, eo_d_name="B Field"):
    if eo_d_name == "B Field":
        return [
            MagneticFieldEvolutionOracle([4 - np.sin(x / 3) for x in range(nqubits)]),
            MagneticFieldEvolutionOracle([1 + np.sin(x / 3) for x in range(nqubits)]),
            MagneticFieldEvolutionOracle([1] * nqubits),
            MagneticFieldEvolutionOracle(np.linspace(0, 2, nqubits)),
            MagneticFieldEvolutionOracle(np.linspace(0, 1, nqubits)),
        ]
    elif eo_d_name == "H_ClassicalIsing(B,J)":
        return [IsingNNEvolutionOracle([0] * nqubits, [1] * nqubits)]


def print_vqe_comparison_report(gci, nmb_digits_rounding=2):
    rounded_values = gci.get_vqe_boosting_data()
    for key in rounded_values:
        if isinstance(rounded_values[key], float):
            rounded_values[key] = round(rounded_values[key], nmb_digits_rounding)
    print(
        f"\
The target energy is {rounded_values['target_energy']}\n\
The VQE energy is {rounded_values['vqe_energy']} \n\
The DBQA energy is {rounded_values['gci_loss']}. \n\
The difference is for VQE is {rounded_values['diff_vqe_target']} \n\
and for the DBQA {rounded_values['diff_gci_target']} \n\
which can be compared to the spectral gap {rounded_values['gap']}.\n\
The relative difference is \n\
    - for VQE {rounded_values['diff_vqe_target_perc']}% \n\
    - for DBQA {rounded_values['diff_gci_target_perc']}%.\n\
The energetic fidelity witness of the ground state is: \n\
    - for the VQE  {rounded_values['fidelity_witness_vqe']} \n\
    - for DBQA {rounded_values['fidelity_witness_gci']}\n\
The true fidelity is \n\
    - for the VQE  {rounded_values['fidelity_vqe']}\n\
    - for DBQA {rounded_values['fidelity_gci']}\n\
                "
    )
    gate_count = gci.print_gate_count_report()


from mpl_toolkits.mplot3d import Axes3D


def plot_lr_s_loss(eval_dict):
    lr = [key[0] for key in eval_dict.keys()]
    s = [key[1] for key in eval_dict.keys()]
    loss = [value for value in eval_dict.values()]
    min_loss_index = loss.index(min(loss))
    min_lr = lr[min_loss_index]
    min_s = s[min_loss_index]
    min_loss = loss[min_loss_index]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(lr, s, loss, c=loss, cmap="viridis")
    ax.scatter([min_lr], [min_s], [min_loss], color="red", s=100, label="Minimum Loss")
    ax.text(
        min_lr,
        min_s,
        min_loss,
        f"({np.round(min_lr,2)}, {np.round(min_s,2)}, {np.round(min_loss,4)})",
        color="red",
    )
    colorbar = plt.colorbar(sc)
    colorbar.set_label("Loss")
    ax.set_xlabel("Learning Rate (lr)")
    ax.set_ylabel("Step (s)")
    ax.set_zlabel("Loss")
    ax.legend()
    ax.set_title("3D Scatter Plot of (lr, s): loss")
