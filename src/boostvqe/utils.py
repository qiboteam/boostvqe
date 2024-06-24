import json
import logging
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
from qibo import hamiltonians
from scipy import optimize

from boostvqe.ansatze import VQE, build_circuit, compute_gradients
from boostvqe.compiling_XXZ import *
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
    circuit = build_circuit(nqubits, config["nlayers"], "numpy")
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5)

    vqe = VQE(circuit, hamiltonian)
    # set target parameters into the VQE
    vqe.circuit.set_parameters(params)

    eo_xxz = XXZ_EvolutionOracle(nqubits, steps=1, order=2)
    # implement the rotate by VQE on the level of circuits
    fsoe = VQERotatedEvolutionOracle(eo_xxz, vqe)
    # init gci with the vqe-rotated hamiltonian
    gci = GroupCommutatorIterationWithEvolutionOracles(
        input_hamiltonian_evolution_oracle=fsoe, mode_double_bracket_rotation=mode_dbr
    )

    gci.vqe = vqe

    eigenergies = hamiltonian.eigenvalues()
    target_energy = np.min(eigenergies)
    gci.h.target_energy = target_energy
    eigenergies.sort()
    gap = eigenergies[1] - target_energy
    gci.h.gap = gap
    gci.h.ground_state = hamiltonian.eigenvectors()[0]

    gci.vqe_energy = hamiltonian.expectation(vqe.circuit().state())

    b_list = [1 + np.sin(x / 3) for x in range(10)]
    gci.eo_d = MagneticFieldEvolutionOracle(b_list, name="D(B = 1+sin(x/3))")
    gci.default_step_grid = np.linspace(0.003, 0.004, 10)

    gci.path = path
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
    eo_d=None,
    optimization_method="cma",
    optimization_config={"maxiter": 100},
    mode_dbr=DoubleBracketRotationType.group_commutator_third_order_reduced,
    please_be_verbose=False,
    please_be_visual=False,
):
    """
    Execute GCI boost with variational optimization of the diagonalizing operator D.

    Supported ``optimization_config`` in case of chosen method "sgd":

            optimization_config={
                "nmb_gd_epochs": 1,
            }

    Supported ``optimization_config`` in case of other chosen methods:

            optimization_config={
                "maxiter": 100,
            }
    """
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

    for gci_step_nmb in range(nmb_gci_steps):
        logging.info(
            f"Optimizing GCI step {gci_step_nmb+1} with optimizer {optimization_method}"
        )
        it = time.time()
        if optimization_method == "sgd":
            _, best_s, _, eo_d = select_recursion_step_gd_circuit(
                gci,
                mode_dbr_list=[mode_dbr],
                step_grid=np.linspace(1e-5, 2e-2, 30),
                lr_range=(1e-3, 1),
                nmb_gd_epochs=optimization_config["nmb_gd_epochs"],
                threshold=1e-4,
                max_eval_gd=30,
                please_be_visual=please_be_visual,
                save_path="gci_step",
            )
        else:
            if gci_step_nmb == 0:
                p0 = [0.01]
                p0.extend([4 - np.sin(x / 3) for x in range(nqubits)])
            else:
                p0 = [best_s]
                p0.extend(best_b)
            optimized_params = optimize_D(
                params=p0,
                gci=gci,
                method=optimization_method,
                maxiter=optimization_config["maxiter"],
            )
            best_s = optimized_params[0]
            best_b = optimized_params[1:]
            eo_d = MagneticFieldEvolutionOracle(best_b)

        logging.info(f"Total optimization time required: {time.time() - it} seconds")

        gci.mode_double_bracket_rotation = mode_dbr
        gci.eo_d = eo_d
        print(gci.loss(best_s, eo_d))
        gci(best_s)

        if please_be_verbose:
            print(f"Executing gci step {gci_step_nmb}:\n")
            print(
                f"The selected data is {gci.mode_double_bracket_rotation} rotation with {gci.eo_d.name} for the duration s = {best_s}."
            )
            print("--- the report after execution:\n")
            print_vqe_comparison_report(gci)
            print("==== the execution report ends here")

    return gci


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


def gnd_state_fidelity_witness(gci, e_state=None):
    if e_state is None:
        e_state = gci.loss()
    return 1 - (e_state - gci.h.target_energy) / gci.h.gap


def gnd_state_fidelity(gci):
    input_state = gci.get_composed_circuit()().state()
    return abs(gci.h.ground_state.T.conj() @ input_state) ** 2


def print_vqe_comparison_report(gci):
    gci_loss = gci.loss()
    print(
        f"VQE energy is {round(gci.vqe_energy,5)} and the DBQA yields {round(gci_loss,5)}. \n\
            The target energy is {round(gci.h.target_energy,5)} which means the difference is for VQE \
            {round(gci.vqe_energy-gci.h.target_energy,5)} and of the DBQA {round(gci_loss-gci.h.target_energy,5)} \
            which can be compared to the spectral gap {round(gci.h.gap,5)}.\n\
            The relative difference is for VQE {round(abs(gci.vqe_energy-gci.h.target_energy)/abs(gci.h.target_energy)*100,5)}% \
            and for DBQA {round(abs(gci_loss-gci.h.target_energy)/abs(gci.h.target_energy)*100,5)}%.\
            The energetic fidelity witness for the ground state for the\n\
            VQE is {round(1- abs(gci.vqe_energy-gci.h.target_energy)/abs(gci.h.gap),5)} \n\
            and DBQA {round(1- abs(gci_loss-gci.h.target_energy)/abs(gci.h.gap),5)}\n\
            The true fidelity is {round(gnd_state_fidelity(gci),5)} (see boostvqe issue https://github.com/qiboteam/boostvqe/issues/51 why this value seems wrong)\n\
            and DBQA {round(gnd_state_fidelity_witness(gci,gci_loss),5)}\
"
    )
    gci.print_gate_count_report()


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


def callback_D_optimization(params, gci, loss_history, params_history):
    params_history.append(params)
    eo_d = MagneticFieldEvolutionOracle(params[1:])
    loss_history.append(gci.loss(params[0], eo_d))


def loss_function_D(params, gci):
    """``params`` has shape [s0, b_list_0]."""
    eo = MagneticFieldEvolutionOracle(params[1:])
    return gci.loss(params[0], eo)


def optimize_D(
    params, gci, method, s_bounds=(1e-4, 1e-1), b_bounds=(0.0, 9.0), maxiter=100
):
    """Optimize Ising GCI model using chosen optimization `method`."""

    # evolutionary strategy
    if method == "cma":
        lower_bounds = s_bounds[0] + b_bounds[0] * (len(params) - 1)
        upper_bounds = s_bounds[1] + b_bounds[1] * (len(params) - 1)
        bounds = [lower_bounds, upper_bounds]
        opt_results = cma.fmin(
            loss_function_D,
            sigma0=0.5,
            x0=params,
            args=(gci,),
            options={"bounds": bounds, "maxiter": maxiter},
        )
        return opt_results[0]
    # scipy optimizations
    else:
        bounds = [s_bounds]
        for _ in range(len(params) - 1):
            bounds.append(b_bounds)
        # dual annealing algorithm
        if method == "annealing":
            opt_results = optimize.dual_annealing(
                func=loss_function_D,
                x0=params,
                bounds=bounds,
                args=(gci,),
                maxiter=maxiter,
            )
        elif method == "differential_evolution":
            opt_results = optimize.differential_evolution(
                func=loss_function_D,
                x0=params,
                bounds=bounds,
                args=(gci,),
                maxiter=maxiter,
            )
        elif method == "DIRECT":
            opt_results = optimize.direct(
                func=loss_function_D,
                bounds=bounds,
                args=(gci,),
                maxiter=maxiter,
            )
        elif method == "basinhopping":
            opt_results = optimize.basinhopping(
                func=loss_function_D,
                x0=params,
                niter=maxiter,
                minimizer_kwargs={"method": "Powell", "args": (gci,)},
            )
        # scipy local minimizers
        else:
            opt_results = optimize.minimize(
                fun=loss_function_D,
                x0=params,
                bounds=bounds,
                args=(gci,),
                method=method,
                options={"disp": 1, "maxiter": maxiter},
            )
    return opt_results.x
