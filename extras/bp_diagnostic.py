import logging
import os

import numpy as np
import qibo
from qibo import hamiltonians

from boostvqe.ansatze import build_circuit, compute_gradients

qibo.set_backend(backend="qibojit", platform="cupy")
logging.basicConfig(level=logging.INFO)

NL = 100
NQ = 21
J = 2

qubits = np.arange(2, NQ, 2)
layers = np.arange(2, NL, 5)

NRUNS = 50

if not os.path.exists("./gradients"):
    # Create the directory since it does not exist
    os.makedirs("./gradients")

grads_vars = np.zeros((len(layers), len(qubits)))

for i, l in enumerate(layers):
    for j, q in enumerate(qubits):
        # initialize model and hamiltonian
        logging.info(f"Running with {q} qubits and {l} layers")
        c = build_circuit(int(q), int(l))
        h = hamiltonians.TFIM(nqubits=int(q), h=q)

        gradients = []
        for n in range(NRUNS):
            p = np.random.uniform(-np.pi, np.pi, len(c.get_parameters()))
            gradients.append(
                np.real(compute_gradients(parameters=p, circuit=c, hamiltonian=h))
            )

        np.save(file=f"gradients/grads_l{l}_q{q}", arr=gradients)
        grads_vars[i][j] = np.var(gradients, axis=0)[J]

np.save(file="gradients_vars", arr=grads_vars)
