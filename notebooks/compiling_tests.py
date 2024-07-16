

from qibo import hamiltonians
import numpy as np
from boostvqe.compiling_XXZ import *
import qibo
qibo.set_backend("numpy")

from boostvqe.compiling_XXZ import *
from boostvqe.models.dbi import double_bracket_evolution_oracles
from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.double_bracket_evolution_oracles import (
    FrameShiftedEvolutionOracle,
    IsingNNEvolutionOracle,
    MagneticFieldEvolutionOracle,
    XXZ_EvolutionOracle,
)



t = 0.01
steps = 3
delta=0.5
nqubits=6

h_xxz = hamiltonians.XXZ(nqubits=nqubits, delta = delta)
u = h_xxz.exp(t)

# this is the Hamiltonian that we are studying
eo_xxz = XXZ_EvolutionOracle.from_nqubits(
    nqubits=nqubits, delta=0.5, steps=10, order=2
)
eo_xxz.steps = 10
circ = eo_xxz.circuit(t)
v = circ.unitary()
print(np.linalg.norm(u-vw_xxz_compiling_phase(nqubits=nqubits,steps = 10, order =2)*v))

params = [4 - np.sin(x / 3) for x in range(nqubits)]
print(params)

eo_d_type = IsingNNEvolutionOracle
eo_d = eo_d_type.load([0]*5 + [.5]*5)

u= eo_d.h.exp(t)

circ = eo_d.circuit(t)
v = circ.unitary()
print(np.linalg.norm(u-v))


