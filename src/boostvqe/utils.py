import copy
import json
import logging
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
from qibo import hamiltonians
from qibo.models.dbi.utils_scheduling import hyperopt_step
from scipy import optimize

from boostvqe import ansatze
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
DELTA = 0.5
TOL = 1e-10
DBI_ENERGIES = "dbi_energies"
DBI_FLUCTUATIONS = "dbi_fluctuations"
DBI_STEPS = "dbi_steps"
DBI_D_MATRIX = "dbi_d_matrices"


logging.basicConfig(level=logging.INFO)


def generate_path(
    output_folder,
    optimizer,
    nqubits,
    seed,
    decay_rate_lr,
    nlayers,
) -> str:
    """Generate path according to job parameters"""
    if output_folder is None:
        output_folder = "results"
    else:
        output_folder = output_folder
    return f"./{output_folder}/{optimizer}_{nqubits}q_{nlayers}l_{seed}s_{decay_rate_lr}decay"


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
    return hamiltonian.dense.energy_fluctuation(final_state)


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

    vqe = ansatze.VQE(
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
        params_history.append(copy.deepcopy(params))
        grads_history.append(
            ansatze.compute_gradients(
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
    matrix_circ = np.matrix(backend.to_numpy(circuit.unitary()))
    matrix_circ_dagger = backend.cast(np.array(matrix_circ.getH()))
    matrix_circ = backend.cast(matrix_circ)
    new_hamiltonian = matrix_circ_dagger @ hamiltonian.matrix @ np.array(matrix_circ)
    return new_hamiltonian


def apply_dbi_steps(dbi, nsteps, d_type=None, method=None, time_step=0.01, **kwargs):
    """Apply `nsteps` of `dbi` to `hamiltonian`."""
    nqubits = dbi.nqubits

    p0 = [time_step]
    if d_type is not None:
        if d_type == MagneticFieldEvolutionOracle:
            p0.extend([4 - np.sin(x / 3) for x in range(nqubits)])
        elif d_type == IsingNNEvolutionOracle:
            p0.extend([4 - np.sin(x / 3) for x in range(nqubits)] + nqubits * [1])
    energies, fluctuations, hamiltonians, steps, d_matrix = [], [], [], [], []
    logging.info(f"Applying {nsteps} steps of DBI to the given hamiltonian.")
    operators = []
    for _ in range(nsteps):
        logging.info(f"step {_+1}")

        if d_type is not None:
            optimized_params, opt_dict = optimize_d_for_dbi(
                p0, copy.deepcopy(dbi), d_type, method, **kwargs
            )
            step = optimized_params[0]
            new_d = d_type.load(optimized_params[1:]).h.matrix
        else:
            step = p0[0]
            new_d = dbi.diagonal_h_matrix

        operator = dbi(step=step, d=new_d)

        operators.append(operator)
        steps.append(step)
        d_matrix.append(new_d)
        zero_state = np.transpose([dbi.h.backend.zero_state(dbi.h.nqubits)])

        logging.info(f"\nH matrix: {dbi.h.matrix}\n")

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
    # circuit = build_circuit(nqubits, config["nlayers"], "numpy")
    circuit = getattr(ansatze, config["ansatz"])(config["nqubits"], config["nlayers"])

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=0.5)

    vqe = ansatze.VQE(circuit, hamiltonian)
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
    mode,
    eo_d_type,
    params,
    step_grid=np.linspace(1e-5, 3e-2, 30),
    lr_range=(1e-3, 1),
    threshold=1e-4,
    max_eval_gd=30,
    nmb_gd_epochs=0,
):
    """Returns: circuit of the step, code of the strategy"""

    # minimal_losses = []
    # minimizer_s = []
    # minimizer_eo_d = []
    eo_d = eo_d_type.load(params)
    # returns min_s, min_loss, loss_list
    s, l, ls = gci.choose_step(d=eo_d, step_grid=step_grid, mode_dbr=mode)
    for epoch in range(nmb_gd_epochs):
        ls = []
        s_min, s_max = step_grid[0], step_grid[-1]
        lr_min, lr_max = lr_range[0], lr_range[-1]
        eo_d, s, l, eval_dict, params, best_lr = choose_gd_params(
            gci=gci,
            eo_d_type=eo_d_type,
            params=params,
            loss_0=l,
            s_0=s,
            s_min=s_min,
            s_max=s_max,
            lr_min=lr_min,
            lr_max=lr_max,
            threshold=threshold,
            max_eval=max_eval_gd,
            mode=mode,
        )

    print(
        f"Just finished the selection: better loss {l} for mode {mode},\
                with duration s={s}, and eo_d name = {eo_d.__class__.__name__}"
    )
    return (
        mode,
        s,
        l,
        eo_d,
    )


def callback_D_optimization(params, gci, loss_history, params_history):
    params_history.append(params)
    gci.eo_d.params = params[1:]
    # eo_d = MagneticFieldEvolutionOracle.from_b(params[1:])
    loss_history.append(gci.loss(params[0]))


def loss_function_D(gci_params, gci, eo_d_type, mode):
    """``params`` has shape [s0, b_list_0]."""
    return gci.loss(gci_params[0], eo_d_type.load(gci_params[1:]), mode)


def optimize_D(
    params,
    gci,
    eo_d_type,
    mode,
    method,
    s_bounds=(1e-4, 1e-1),
    b_bounds=(0.0, 9.0),
    maxiter=100,
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
            args=(gci, eo_d_type, mode),
            options={"bounds": bounds, "maxiter": maxiter},
        )
        result_dict = convert_numpy(opt_results[-2].result._asdict())
        return opt_results[0], {f"{method}_extras": result_dict}
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
                args=(gci, eo_d_type, mode),
                maxiter=maxiter,
            )
        elif method == "differential_evolution":
            opt_results = optimize.differential_evolution(
                func=loss_function_D,
                x0=params,
                bounds=bounds,
                args=(gci, eo_d_type, mode),
                maxiter=maxiter,
            )
        elif method == "DIRECT":
            opt_results = optimize.direct(
                func=loss_function_D,
                bounds=bounds,
                args=(gci, eo_d_type, mode),
                maxiter=maxiter,
            )
        elif method == "basinhopping":
            opt_results = optimize.basinhopping(
                func=loss_function_D,
                x0=params,
                niter=maxiter,
                minimizer_kwargs={"method": "Powell", "args": (gci, eo_d_type, mode)},
            )
        # scipy local minimizers
        else:
            opt_results = optimize.minimize(
                fun=loss_function_D,
                x0=params,
                bounds=bounds,
                args=(gci, eo_d_type, mode),
                method=method,
                options={"disp": 1, "maxiter": maxiter},
            )
    return opt_results.x, {f"{method}_extras": convert_numpy(dict(opt_results))}


def optimize_d_for_dbi(
    params,
    dbi,
    d_type,
    method,
    s_bounds=(-1e-1, 1e-1),
    b_bounds=(0.0, 9.0),
    maxiter=100,
):
    """Optimize Ising GCI model using chosen optimization `method`."""

    # evolutionary strategy
    if method == "cma":
        lower_bounds = s_bounds[0] + b_bounds[0] * (len(params) - 1)
        upper_bounds = s_bounds[1] + b_bounds[1] * (len(params) - 1)
        bounds = [lower_bounds, upper_bounds]
        opt_results = cma.fmin(
            loss_function_d_dbi,
            sigma0=0.5,
            x0=params,
            args=(dbi, d_type),
            options={"bounds": bounds, "maxiter": maxiter},
        )
        result_dict = convert_numpy(opt_results[-2].result._asdict())
        return opt_results[0], {f"{method}_extras": result_dict}
    # scipy optimizations
    else:
        bounds = [s_bounds]
        for _ in range(len(params) - 1):
            bounds.append(b_bounds)
        # dual annealing algorithm
        if method == "annealing":
            opt_results = optimize.dual_annealing(
                func=loss_function_d_dbi,
                x0=params,
                bounds=bounds,
                args=(dbi, d_type),
                maxiter=maxiter,
            )
        elif method == "differential_evolution":
            opt_results = optimize.differential_evolution(
                func=loss_function_d_dbi,
                x0=params,
                bounds=bounds,
                args=(dbi, d_type),
                maxiter=maxiter,
            )
        elif method == "DIRECT":
            opt_results = optimize.direct(
                func=loss_function_d_dbi,
                bounds=bounds,
                args=(dbi, d_type),
                maxiter=maxiter,
            )
        elif method == "basinhopping":
            opt_results = optimize.basinhopping(
                func=loss_function_d_dbi,
                x0=params,
                niter=maxiter,
                minimizer_kwargs={"method": "Powell", "args": (dbi, d_type)},
            )
        # scipy local minimizers
        else:
            opt_results = optimize.minimize(
                fun=loss_function_d_dbi,
                x0=params,
                bounds=bounds,
                args=(dbi, d_type),
                method=method,
                options={"disp": 1, "maxiter": maxiter},
            )
    return opt_results.x, {f"{method}_extras": convert_numpy(dict(opt_results))}


def loss_function_d_dbi(dbi_params, dbi, d_type):
    """``params`` has shape [s0, b_list_0]."""
    test_dbi = copy.deepcopy(dbi)
    d = d_type.load(dbi_params[1:]).h.matrix
    test_dbi(step=dbi_params[0], d=d)
    zero_state = test_dbi.backend.zero_state(test_dbi.nqubits)
    return test_dbi.h.expectation(zero_state)


def loss_function_D(gci_params, gci, eo_d_type, mode):
    """``params`` has shape [s0, b_list_0]."""
    return gci.loss(gci_params[0], eo_d_type.load(gci_params[1:]), mode)


def convert_numpy(obj):
    """Convert numpy objects into python types which can be dumped."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj
