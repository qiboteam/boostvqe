from qibo import hamiltonians
import numpy as np
from boostvqe.compiling_XXZ import *

t = 0.01
steps = 3
delta=0.5
nqubits=6

h_xxz = hamiltonians.XXZ(nqubits=nqubits, delta = delta)
u = h_xxz.exp(t)
circ = nqubit_XXZ_decomposition(nqubits=nqubits,t=t,delta=delta,steps=steps)
v = circ.unitary()
print(np.linalg.norm(u-np.exp(nqubits*steps*1j*np.pi/4)*v))
print(circ.draw())