import argparse
import json
import logging
import pathlib
import time
from copy import deepcopy

import numpy as np
import qibo

from qibo import hamiltonians
from qibo.backends import construct_backend
from qibo.quantum_info.metrics import fidelity

from boostvqe.ansatze import VQE, build_circuit_RBS
from boostvqe.models.dbi import double_bracket_evolution_oracles
from boostvqe.models.dbi.double_bracket_evolution_oracles import (
    FrameShiftedEvolutionOracle,
    IsingNNEvolutionOracle,
    MagneticFieldEvolutionOracle,
    XXZ_EvolutionOracle,
)
from boostvqe.models.dbi.group_commutator_iteration_transpiler import (
    DoubleBracketRotationType,
    GroupCommutatorIterationWithEvolutionOracles,
)
from boostvqe.utils import (
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    build_circuit_RBS,
    optimize_D,
    select_recursion_step_gd_circuit,
)

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")


def dump_config(config: dict, path):
    config["path"] = config["path"]
    config["db_rotation"] = config["db_rotation"].name
    (path / "config.json").write_text(json.dumps(config, indent=4))


def main(args):
    """VQE training."""
    path = pathlib.Path(args.path)
    dump_path = (
        path
        / f"{args.db_rotation.name}_{args.optimization_method}_{args.epoch}e_{args.steps}s"
    )
    dump_path.mkdir(parents=True, exist_ok=True)

    config = json.loads((path / OPTIMIZATION_FILE).read_text())
    dump_config(deepcopy(vars(args)), path=dump_path)

    if args.optimization_config is None:
        opt_options = {}
    else:
        opt_options = json.loads(args.optimization_config)

    try:
        params = np.load(path / f"parameters/params_ite{args.epoch}.npy")
    except FileNotFoundError:
        params = np.array(
            np.load(path / PARAMS_FILE, allow_pickle=True).tolist()[0][args.epoch]
        )

    nqubits = config["nqubits"]
    nlayers = config["nlayers"]
    vqe_backend = construct_backend(backend=config["backend"])
    # TODO: remove delta hardcoded
    print
    hamiltonian = getattr(hamiltonians, config["hamiltonian"])(
        nqubits=nqubits, delta=0.5, backend=vqe_backend
    )
    vqe = VQE(
        build_circuit_RBS(
            nqubits=nqubits,
            nlayers=nlayers,
        ),
        hamiltonian=hamiltonian,
    )
    vqe.circuit.set_parameters(params)

    base_oracle = XXZ_EvolutionOracle.from_nqubits(
        nqubits=nqubits, delta=0.5, steps=args.steps, order=args.order
    )
    oracle = FrameShiftedEvolutionOracle.from_evolution_oracle(
        before_circuit=vqe.circuit.invert(),
        after_circuit=vqe.circuit,
        base_evolution_oracle=base_oracle,
    )

    gci = GroupCommutatorIterationWithEvolutionOracles(
        oracle,
        args.db_rotation,
    )

    # TODO: remove hardcoded magnetic field
    eo_d_type = getattr(double_bracket_evolution_oracles, args.eo_d)

    print(
        f"The gci mode is {gci.double_bracket_rotation_type} rotation with {eo_d_type.__name__} as the oracle.\n"
    )
    metadata = {}

    for gci_step_nmb in range(args.steps):
        logging.info(
            "\n################################################################################\n"
            + f"Optimizing GCI step {gci_step_nmb+1} with optimizer {args.optimization_method}"
            + "\n################################################################################\n"
        )
        it = time.time()
        if args.optimization_method == "sgd":
            params = (
                [4 - np.sin(x / 3) for x in range(nqubits)]
                if eo_d_type == MagneticFieldEvolutionOracle
                else [4 - np.sin(x / 3) for x in range(nqubits)] + nqubits * [1]
            )
            mode, best_s, best_b, eo_d = select_recursion_step_gd_circuit(
                gci,
                mode=args.db_rotation,
                eo_d_type=eo_d_type,
                params=params,
                step_grid=np.linspace(1e-5, 2e-2, 30),
                lr_range=(1e-3, 1),
                nmb_gd_epochs=opt_options["gd_epochs"],
                threshold=1e-4,
                max_eval_gd=30,
            )

            opt_dict = {"sgd_extras": "To be defined"}

        else:
            if gci_step_nmb == 0:
                p0 = [0.01]
                if eo_d_type == MagneticFieldEvolutionOracle:
                    p0.extend([4 - np.sin(x / 3) for x in range(nqubits)])
                elif eo_d_type == IsingNNEvolutionOracle:
                    p0.extend(
                        [4 - np.sin(x / 3) for x in range(nqubits)] + nqubits * [1]
                    )

            else:
                p0 = [best_s]
                p0.extend(best_b)
            optimized_params, opt_dict = optimize_D(
                params=p0,
                gci=gci,
                eo_d_type=eo_d_type,
                mode=args.db_rotation,
                method=args.optimization_method,
                **opt_options,
            )
            best_s = optimized_params[0]
            best_b = optimized_params[1:]
            eo_d = eo_d_type.load(best_b)

        step_data = dict(
            best_s=best_s,
            eo_d_name=eo_d.__class__.__name__,
            eo_d_params=eo_d.params,
        )
        logging.info(f"Total optimization time required: {time.time() - it} seconds")
        gci.mode_double_bracket_rotation = args.db_rotation

        gci(best_s, eo_d, args.db_rotation)

        this_report = report(vqe, hamiltonian, gci, best_s, eo_d, args.db_rotation)
        print_report(this_report)
        metadata[gci_step_nmb + 1] = this_report | step_data | opt_dict

    (dump_path / "boosting_data.json").write_text(json.dumps(metadata, indent=4))


def report(vqe, hamiltonian, gci, step, eo_d, mode):
    energies = hamiltonian.eigenvalues()
    ground_state_energy = float(energies[0])
    vqe_energy = float(hamiltonian.expectation(vqe.circuit().state()))
    gci_loss = float(gci.loss(step, eo_d, mode))
    gap = float(energies[1] - energies[0])

    return (
        dict(
            nqubits=hamiltonian.nqubits,
            gci_loss=float(gci_loss),
            vqe_energy=float(vqe_energy),
            target_energy=ground_state_energy,
            diff_vqe_target=vqe_energy - ground_state_energy,
            diff_gci_target=gci_loss - ground_state_energy,
            gap=gap,
            diff_vqe_target_perc=abs(vqe_energy - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            diff_gci_target_perc=abs(gci_loss - ground_state_energy)
            / abs(ground_state_energy)
            * 100,
            fidelity_witness_vqe=1 - (vqe_energy - ground_state_energy) / gap,
            fidelity_witness_gci=1 - (gci_loss - ground_state_energy) / gap,
            fidelity_vqe=fidelity(vqe.circuit().state(), hamiltonian.ground_state()),
            fidelity_gci=fidelity(
                gci.get_composed_circuit()().state(), hamiltonian.ground_state()
            ),
        )
        | gci.get_gate_count_dict()
    )


def print_report(report: dict):
    print(
        f"\
    The target energy is {report['target_energy']}\n\
    The VQE energy is {report['vqe_energy']} \n\
    The DBQA energy is {report['gci_loss']}. \n\
    The difference is for VQE is {report['diff_vqe_target']} \n\
    and for the DBQA {report['diff_gci_target']} \n\
    which can be compared to the spectral gap {report['gap']}.\n\
    The relative difference is \n\
        - for VQE {report['diff_vqe_target_perc']}% \n\
        - for DBQA {report['diff_gci_target_perc']}%.\n\
    The energetic fidelity witness of the ground state is: \n\
        - for the VQE  {report['fidelity_witness_vqe']} \n\
        - for DBQA {report['fidelity_witness_gci']}\n\
    The true fidelity is \n\
        - for the VQE  {report['fidelity_vqe']}\n\
        - for DBQA {report['fidelity_gci']}\n\
                    "
    )
    print(
        f"The boosting circuit used {report['nmb_cnot']} CNOT gates coming from compiled XXZ evolution and {report['nmb_cz']} CZ gates from VQE.\n\
For {report['nqubits']} qubits this gives n_CNOT/n_qubits = {report['nmb_cnot_relative']} and n_CZ/n_qubits = {report['nmb_cz_relative']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument("--backend", default="qibojit", type=str, help="Qibo backend")
    parser.add_argument("--path", type=str, help="Output folder")
    parser.add_argument(
        "--epoch", default=-1, type=int, help="VQE epoch where DBI will be applied."
    )
    parser.add_argument(
        "--eo_d",
        default="IsingNNEvolutionOracle",
        help="Evolution Oracle D operator. Can be either MagneticFieldEvolutionOracle or IsingNNEvolutionOracle.",
    )
    parser.add_argument("--steps", default=3, type=int, help="DBI steps")
    parser.add_argument("--order", default=2, type=int, help="Suzuki-Trotter order")
    parser.add_argument(
        "--db_rotation",
        type=lambda arg: DoubleBracketRotationType[arg],
        choices=DoubleBracketRotationType,
        default="group_commutator_third_order_reduced",
        help="DB rotation type.",
    )
    parser.add_argument(
        "--optimization_method", default="sgd", type=str, help="Optimization method"
    )
    parser.add_argument(
        "--optimization_config",
        type=str,
        help="Options to customize the optimizer training.",
    )
    args = parser.parse_args()
    main(args)
