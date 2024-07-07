import json
import time
import argparse
import logging
import copy
import pathlib

import numpy as np
from scipy.optimize import minimize

import qibo
from qibo import hamiltonians, set_backend
from qibo.backends import construct_backend

from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from boostvqe.ansatze import (
    VQE, 
    build_circuit,
    build_circuit_RBS,
)

from boostvqe.training_utils import Model

from boostvqe.utils import (
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    apply_dbi_steps, 
    rotate_h_with_vqe,
)

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")

def cost_function(s, dbi, zero_state):
    """Compute the final energy we get applying step `s` to DBI."""
    original_h = copy.deepcopy(dbi.h)
    dbi(step=s, d=dbi.diagonal_h_matrix)
    cost = dbi.h.expectation(np.asarray(zero_state))
    dbi.h = original_h
    return cost

def optimize_s(dbi, zero_state):
    """Optimize DBI stepsize using Powell minimizer."""
    opt_result = minimize(
        cost_function, 
        x0=[0.01], 
        args=(dbi, zero_state),
        method="Powell",
        options={"maxiter": 200}
    )
    return opt_result

def main(args):
    path = pathlib.Path(args.path)
    dump_path = (
        path
        / f"single_commutator_hyperopt_{args.epoch}e_{args.steps}s"
    )
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
    # TODO: remove delta hardcoded
    hamiltonian = getattr(Model, config["hamiltonian"])(config["nqubits"])

    if config["ansatz"] == "hw_preserving":
        circ = build_circuit_RBS(
            nqubits=config["nqubits"],
            nlayers=config["nlayers"],
        )
    elif config["ansatz"] == "hdw_efficient":
        circ = build_circuit(
            nqubits=config["nqubits"],
            nlayers=config["nlayers"],
        )

    vqe = VQE(
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
    new_hamiltonian_matrix = np.array(rotate_h_with_vqe(hamiltonian=hamiltonian, vqe=vqe))
    new_hamiltonian = hamiltonians.Hamiltonian(
        nqubits, matrix=new_hamiltonian_matrix
    )


    dbi = DoubleBracketIteration(
        hamiltonian=new_hamiltonian,
        mode=DoubleBracketGeneratorType.single_commutator,
    )

    zero_state_t = np.transpose([zero_state])
    energy_h0 = float(dbi.h.expectation(np.array(zero_state_t)))
    fluctuations_h0 = float(dbi.h.energy_fluctuation(zero_state_t))

    energies = []

    for s in range(args.steps):
        logging.info(f"Optimizing step {s+1}")
        s = optimize_s(dbi, zero_state).x
        dbi(step=s, d=dbi.diagonal_h_matrix)
        this_energy = dbi.h.expectation(zero_state)
        logging.info(f"Best found energy: {this_energy}")
        energies.append(this_energy)
    
    dict_results = {
        "VQE energy": float(ene1),
        "Ground state": float(target_energy),
        "dbi_energies": energies
    }
    
    (dump_path / "boosting_data.json").write_text(json.dumps(dict_results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument("--path", type=str, help="Output folder")
    parser.add_argument(
        "--epoch", default=-1, type=int, help="VQE epoch where DBI will be applied."
    )
    parser.add_argument("--steps", default=3, type=int, help="DBI steps")

    args = parser.parse_args()
    main(args)