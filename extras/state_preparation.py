

import logging

import numpy as np
import matplotlib.pyplot as plt

from qibo import hamiltonians, gates, set_backend
from qibo.models import VQE
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from boostvqe.ansatze import build_circuit, compute_gradients
from boostvqe.utils import (
    apply_dbi_steps,
    rotate_h_with_vqe,
)

SEEDS = np.arange(1, 101, 2)
NQUBITS = np.arange(2, 13, 2)
NLAYERS = 20
NSTEPS = 2
VERBOSE = False

set_backend("numpy")
logging.basicConfig(level=logging.INFO)

for q in NQUBITS:
    q = int(q)

    logging.info(f"Executing for {q} qubits")
    for s in SEEDS:
        logging.info(f"and with andom seed: {s}")
        c = build_circuit(q, NLAYERS)
        h = hamiltonians.XXZ(nqubits=q, delta=0.5)
        vqe = VQE(circuit=c, hamiltonian=h)

        np.random.seed(s)
        params = np.random.uniform(-np.pi, np.pi, len(c.get_parameters()))

        if VERBOSE:
            logging.info("Calculating gradients before DBI")

        np.save(arr=compute_gradients(params, c, h), file=f"sp_data/init_{q}q_{s}s")

        rotated_h_matrix = rotate_h_with_vqe(hamiltonian=h, vqe=vqe)
        rotated_h = hamiltonians.Hamiltonian(q, matrix=rotated_h_matrix)

        dbi = DoubleBracketIteration(
            hamiltonian=rotated_h,
            mode=DoubleBracketGeneratorType.single_commutator,
        )

        if VERBOSE:
            logging.info(f"Rotating the hamiltonian using {NSTEPS} DBI steps")
        _, _, _, _, _, dbi_operators = apply_dbi_steps(dbi=dbi, nsteps=NSTEPS)

        dbi_operators = [
            h.backend.cast(np.matrix(h.backend.to_numpy(operator)))
            for operator in dbi_operators
        ]

        c_copy = c.copy(deep=True)
        c_matrix = c.unitary()

        for gate in reversed([c_matrix] + dbi_operators):
            c_copy.add(gates.Unitary(gate, *range(c_copy.nqubits), trainable=False))

        if VERBOSE:
            logging.info("Calculating gradients after DBI")
        c_copy.set_parameters(params)
        np.save(arr=compute_gradients(params, c_copy, h), file=f"sp_data/final_{q}q_{s}s")


