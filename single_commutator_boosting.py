import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians, set_backend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from boostvqe.ansatze import VQE, build_circuit
from boostvqe.utils import (
    OPTIMIZATION_FILE,
    PARAMS_FILE,
    build_circuit, 
    apply_dbi_steps, 
    rotate_h_with_vqe,
)

logging.basicConfig(level=logging.INFO)
qibo.set_backend("numpy")

def main(args):
    path = pathlib.Path(args.path)
    dump_path = (
        path
        / f"single_commutator_{args.optimization_method}_{args.epoch}e_{args.steps}s"
    )
    dump_path.mkdir(parents=True, exist_ok=True)

    config = json.loads((path / OPTIMIZATION_FILE).read_text())
    dump_config(deepcopy(vars(args)), path=dump_path)

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

    zero_state = hamiltonian.backend.zero_state(config["nqubits"])
    target_energy = np.min(hamiltonian.eigenvalues())

    # set target parameters into the VQE
    vqe.circuit.set_parameters(params)
    vqe_state = vqe.circuit().state()

    ene1 = hamiltonian.expectation(vqe_state)

    dbi = DoubleBracketIteration(
        hamiltonian=new_hamiltonian,
        mode=DoubleBracketGeneratorType.single_commutator,
    )

    zero_state_t = np.transpose([zero_state])
    energy_h0 = float(dbi.h.expectation(np.array(zero_state_t)))
    fluctuations_h0 = float(dbi.h.energy_fluctuation(zero_state_t))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument("--path", type=str, help="Output folder")
    parser.add_argument(
        "--epoch", default=-1, type=int, help="VQE epoch where DBI will be applied."
    )
    parser.add_argument("--steps", default=3, type=int, help="DBI steps")

    args = parser.parse_args()
    main(args)