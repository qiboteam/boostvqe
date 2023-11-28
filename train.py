import numpy as np

from qibo.models.variational import VQE
from qibo import hamiltonians

from ansatze import build_floquet

nqubits = 4
nlayers = 5

h = hamiltonians.XXZ(nqubits=nqubits)
c = build_circuit(nqubits=nqubits, nlayers=nlayers)
print(c.draw())

nparams = len(c.get_parameters())

# initialize VQE
vqe = VQE(circuit=c, hamiltonian=h)
initial_parameters = np.random.randn(nparams)


result = vqe.minimize(initial_parameters)

print(result)


