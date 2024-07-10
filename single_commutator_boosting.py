import argparse
import copy
import json
import logging
import pathlib
import time

import numpy as np
import qibo
from qibo import hamiltonians, set_backend
from qibo.backends import construct_backend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from scipy.optimize import minimize

from boostvqe import ansatze
from boostvqe.models.dbi import double_bracket_evolution_oracles
from boostvqe.training_utils import Model
from boostvqe.utils import (  # build_circuit_RBS,
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    apply_dbi_steps,
    rotate_h_with_vqe,
)

logging.basicConfig(level=logging.ERROR)
qibo.set_backend("numpy")


def main(args):
    path = pathlib.Path(args.path)
    dump_path = path / f"single_commutator_hyperopt_{args.epoch}e_{args.steps}s"
    dump_path.mkdir(parents=True, exist_ok=True)

    config = json.loads((path / OPTIMIZATION_FILE).read_text())
    # dump_config(deepcopy(vars(args)), path=dump_path)

    try:
        params = np.load(path / f"parameters/params_ite{args.epoch}.npy")
    except FileNotFoundError:
        params = np.array(
            np.load(path / PARAMS_FILE, allow_pickle=True).tolist()[0][args.epoch]
        )

    nqubits = config["nqubits"]
    nlayers = config["nlayers"]
    vqe_backend = construct_backend(backend=config["backend"])
    hamiltonian = getattr(Model, config["hamiltonian"])(config["nqubits"])

    eo_d_type = getattr(double_bracket_evolution_oracles, args.eo_d)
    if args.optimization_config is None:
        opt_options = {}
    else:
        opt_options = json.loads(args.optimization_config)
    # construct circuit from parsed ansatz name
    circ = getattr(ansatze, config["ansatz"])(config["nqubits"], config["nlayers"])

    vqe = ansatze.VQE(
        circuit=circ,
        hamiltonian=hamiltonian,
    )
    print(vqe.circuit.draw())
    vqe.circuit.set_parameters(params)

    zero_state = np.asarray(hamiltonian.backend.zero_state(config["nqubits"]))
    target_energy = np.min(np.array(hamiltonian.eigenvalues()))

    # set target parameters into the VQE
    vqe.circuit.set_parameters(params)
    vqe_state = vqe.circuit().state()

    ene1 = hamiltonian.expectation(vqe_state)

    print("Rotating with VQE")
    new_hamiltonian_matrix = np.array(
        rotate_h_with_vqe(hamiltonian=hamiltonian, vqe=vqe)
    )
    new_hamiltonian = hamiltonians.Hamiltonian(nqubits, matrix=new_hamiltonian_matrix)

    dbi = DoubleBracketIteration(
        hamiltonian=new_hamiltonian,
        mode=DoubleBracketGeneratorType.single_commutator,
    )

    zero_state_t = np.transpose([zero_state])
    energy_h0 = float(dbi.h.expectation(np.array(zero_state_t)))
    fluctuations_h0 = float(dbi.h.energy_fluctuation(zero_state_t))
    dbi_results = apply_dbi_steps(
        dbi=dbi,
        nsteps=args.steps,
        d_type=eo_d_type,
        method=args.optimization_method,
        **opt_options,
    )

    dbi_energies = dbi_results[1]
    dict_results = {
        "VQE energy": float(ene1),
        "Ground state": float(target_energy),
        "dbi_energies": dbi_energies,
    }

    (dump_path / "boosting_data.json").write_text(json.dumps(dict_results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument("--path", type=str, help="Output folder")
    parser.add_argument(
        "--epoch", default=-1, type=int, help="VQE epoch where DBI will be applied."
    )
    parser.add_argument("--steps", default=1, type=int, help="DBI steps")
    parser.add_argument(
        "--optimization_method", default="sgd", type=str, help="Optimization method"
    )
    parser.add_argument(
        "--eo_d",
        default="IsingNNEvolutionOracle",
        help="Evolution Oracle D operator. Can be either MagneticFieldEvolutionOracle or IsingNNEvolutionOracle.",
    )
    parser.add_argument(
        "--optimization_config",
        type=str,
        help="Options to customize the optimizer training.",
    )
    args = parser.parse_args()
    main(args)
