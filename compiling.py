import argparse
import json
import logging
import pathlib
import time

import numpy as np
import qibo

qibo.set_backend("numpy")
from qibo import hamiltonians
from qibo.backends import construct_backend
from qibo.quantum_info.metrics import fidelity

from boostvqe.ansatze import VQE, build_circuit
from boostvqe.models.dbi.double_bracket_evolution_oracles import (
    FrameShiftedEvolutionOracle,
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
    build_circuit,
    optimize_D,
    select_recursion_step_gd_circuit,
)

logging.basicConfig(level=logging.INFO)


def main(args):
    """VQE training."""
    path = pathlib.Path(args.path)
    config = json.loads((path / OPTIMIZATION_FILE).read_text())

    if args.optimization_config is None:
        opt_options = {}
    else:
        opt_options = json.loads(args.optimization_config)

    # TODO: improve loading of params
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
    hamiltonian = getattr(hamiltonians, config["hamiltonian"])(
        nqubits=nqubits, delta=0.5, backend=vqe_backend
    )
    vqe = VQE(
        build_circuit(
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
    eo_d = MagneticFieldEvolutionOracle.from_b(
        [4 - np.sin(x / 3) for x in range(nqubits)]
    )
    gci.eo_d = eo_d
    print(
        f"The gci mode is {gci.double_bracket_rotation_type} rotation with {eo_d.__class__.__name__} as the oracle.\n"
    )
    metadata = {}

    print_report(report(vqe, hamiltonian, gci))
    for gci_step_nmb in range(args.steps):
        logging.info(
            f"Optimizing GCI step {gci_step_nmb+1} with optimizer {args.optimization_method}"
        )
        it = time.time()
        if args.optimization_method == "sgd":
            _, best_s, _, eo_d = select_recursion_step_gd_circuit(
                gci,
                mode_dbr_list=[args.db_rotation],
                step_grid=np.linspace(1e-5, 2e-2, 30),
                lr_range=(1e-3, 1),
                nmb_gd_epochs=opt_options["gd_epochs"],
                threshold=1e-4,
                max_eval_gd=30,
                please_be_visual=False,
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
                method=args.optimization_method,
                **opt_options,
            )
            best_s = optimized_params[0]
            best_b = optimized_params[1:]
            eo_d = MagneticFieldEvolutionOracle.from_b(best_b)

        step_data = dict(
            best_s=best_s,
            eo_d_name=eo_d.__class__.__name__,
            eo_d_params=eo_d.params,
        )
        logging.info(f"Total optimization time required: {time.time() - it} seconds")
        metadata[gci_step_nmb] = report(vqe, hamiltonian, gci) | step_data
        gci.mode_double_bracket_rotation = args.db_rotation
        gci.eo_d = eo_d
        gci(best_s)
        print_report(report(vqe, hamiltonian, gci))

    (path / "boosting_data.json").write_text(json.dumps(metadata, indent=4))


def report(vqe, hamiltonian, gci):
    energies = hamiltonian.eigenvalues()
    ground_state_energy = float(energies[0])
    vqe_energy = float(hamiltonian.expectation(vqe.circuit().state()))
    gci_loss = float(gci.loss())
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
    parser.add_argument("--steps", default=2, type=int, help="DBI steps")
    parser.add_argument("--order", default=2, type=int, help="Suzuki-Trotter order")
    parser.add_argument(
        "--db_rotation",
        type=lambda arg: DoubleBracketRotationType[arg],
        choices=DoubleBracketRotationType,
        default="group_commutator_reduced",
        help="DB rotation type.",
    )
    parser.add_argument(
        "--eo_d_name", default="B Field", type=str, help="D initialization"
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
