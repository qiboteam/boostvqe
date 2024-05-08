import matplotlib.pyplot as plt
import numpy as np
import qibo
import tqdm
from qibo import hamiltonians

from boostvqe.ansatze import build_circuit, compute_gradients

qibo.set_backend("numpy")

NL = 60
NQ = 5
J = 2

qubits = np.arange(2, NQ, 1)
layers = np.arange(2, NL, 5)

NRUNS = 50

grads_vars = np.zeros((len(layers), len(qubits)))

for i, l in enumerate(layers):
    for j, q in enumerate(qubits):
        # initialize model and hamiltonian
        print(f"Running with {q} qubits and {l} layers")
        c = build_circuit(int(q), int(l))
        h = hamiltonians.TFIM(nqubits=int(q), h=q)

        gradients = []
        for n in tqdm.tqdm(range(NRUNS)):
            p = np.random.uniform(-np.pi, np.pi, len(c.get_parameters()))
            gradients.append(
                np.real(compute_gradients(parameters=p, circuit=c, hamiltonian=h))
            )

        np.save(file=f"gradients/grads_l{l}_q{q}", arr=gradients)
        grads_vars[i][j] = np.var(gradients, axis=0)[J]

np.save(file="gradients_vars", arr=grads_vars)
