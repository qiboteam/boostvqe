import argparse
import json
import logging
import pathlib

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.backends import construct_backend

from boostvqe.ansatze import VQE, build_circuit
from boostvqe.models.dbi.double_bracket_evolution_oracles import (
    MagneticFieldEvolutionOracle,
    VQERotatedEvolutionOracle,
    XXZ_EvolutionOracle,
)
from boostvqe.models.dbi.group_commutator_iteration_transpiler import (
    DoubleBracketRotationType,
    VQEBoostingGroupCommutatorIteration,
)
from boostvqe.utils import (
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    build_circuit,
    get_eo_d_initializations,
    print_vqe_comparison_report,
    select_recursion_step_gd_circuit,
)

logging.basicConfig(level=logging.INFO)


def main(args):
    """VQE training."""
    path = args.path

    config = json.loads((path / OPTIMIZATION_FILE).read_text())

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

    base_oracle = XXZ_EvolutionOracle(
        nqubits=nqubits, steps=args.steps, order=args.order
    )
    oracle = VQERotatedEvolutionOracle(base_oracle, vqe)

    gci = VQEBoostingGroupCommutatorIteration(
        input_hamiltonian_evolution_oracle=oracle,
        mode_double_bracket_rotation=args.db_rotation,
    )
    # TODO: remove hardcoded magnetic field
    eo_d = MagneticFieldEvolutionOracle([4 - np.sin(x / 3) for x in range(nqubits)])

    gci.eo_d = eo_d
    print(
        f"The gci mode is {gci.mode_double_bracket_rotation} rotation with {gci.eo_d.name} as the oracle.\n"
    )
    print_vqe_comparison_report(gci)
    boosting_callback_data = {}
    for gci_step_nmb in range(args.steps):
        mode_dbr, minimizer_s, minimal_loss, eo_d = select_recursion_step_gd_circuit(
            gci,
            mode_dbr_list=[args.db_rotation],
            step_grid=np.linspace(1e-5, 2e-2, 30),
            lr_range=(1e-3, 1),
            nmb_gd_epochs=args.gd_steps,
            threshold=1e-4,
            max_eval_gd=30,
            save_path=args.path.name,
        )

        gci.mode_double_bracket_rotation = mode_dbr
        gci.eo_d = eo_d
        gci(minimizer_s)
        print(f"Executing gci step {gci_step_nmb+1}:\n")
        print(
            f"The selected data is {gci.mode_double_bracket_rotation} rotation with {gci.eo_d.name} for the duration s = {minimizer_s}."
        )
        print("--- the report after execution:\n")
        print_vqe_comparison_report(gci)
        print("==== the execution report ends here")
        boosting_callback_data[gci_step_nmb] = gci.get_vqe_boosting_data()
    # TODO: store metadata
    # (args.path / "boosting_data.json").write_text(json.dumps(boosting_callback_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument("--backend", default="qibojit", type=str, help="Qibo backend")
    parser.add_argument("--path", type=pathlib.Path, help="Output folder")
    parser.add_argument(
        "--epoch", default=-1, type=int, help="VQE epoch where DBI will be applied."
    )
    parser.add_argument("--steps", default=2, type=int, help="DBI steps")
    parser.add_argument(
        "--gd_steps", default=1, type=int, help="Gradient descent steps"
    )
    parser.add_argument("--order", default=2, type=int, help="Suzuki-Trotter order")
    parser.add_argument(
        "--db_rotation",
        default=DoubleBracketRotationType.group_commutator_reduced_twice,
        type=DoubleBracketRotationType,
        help="DB rotation type.",
    )
    parser.add_argument(
        "--eo_d_name", default="B Field", type=str, help="D initialization"
    )
    args = parser.parse_args()
    main(args)
