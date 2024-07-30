import numpy as np
from qibo import hamiltonians, set_backend
from boostvqe.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from boostvqe.models.dbi.utils import *
from boostvqe.models.dbi.utils_scheduling import *
from boostvqe.models.dbi.utils_dbr_strategies import *

from qibo import hamiltonians
from boostvqe.compiling_XXZ import *

import matplotlib.pyplot as plt
from utils import *

run_param_rc(16)

def h_TFIM(nqubits, h):
    hamiltonian = SymbolicHamiltonian(
                sum(
                    [
                        symbols.X(j) * symbols.X(j + 1)
                        + h * symbols.Z(j)
                        for j in range(nqubits - 1)
                    ]
                    + [
                        symbols.X(nqubits - 1) * symbols.X(0)
                        + h * symbols.Z(nqubits - 1)
                    ]
                ),
                nqubits=nqubits,
            )
    return hamiltonian.dense

def initialize_dbi(nqubits, model, param):
    if model == "XXZ":
        hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=param)
    if model == "TFIM_qibo":
        hamiltonian = hamiltonians.TFIM(nqubits=nqubits, h=param)
    if model == "TFIM":
        hamiltonian = h_TFIM(nqubits=nqubits, h=param)
    dbi = DoubleBracketIteration(hamiltonian=hamiltonian)
    return dbi

def plot_sigma_time(dbi, d, mode, s_space):
    dbi.mode = mode
    return [dbi.loss(step=s, d=d) for s in s_space]

# dbi = initialize_dbi(5, "TFIM", 2)
dbi = initialize_dbi(5, "XXZ", 0.3)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
s_space = np.linspace(1e-4,0.5,100)
modes = [DoubleBracketGeneratorType.single_commutator,
         DoubleBracketGeneratorType.group_commutator,
         DoubleBracketGeneratorType.group_commutator_3]
plots = []
for mode in modes:
    plots.append(plot_sigma_time(dbi, dbi.diagonal_h_matrix, mode, s_space))

mode_names = [r'$e^{-s\hat W_k}$',
              r'$\hat V^{GC}_k$',
              r'$\hat V^{3rd GC}_k$']
s_min = []
for i,mode in enumerate(modes):
    plt.plot(s_space, plots[i], label=mode_names[i])
    s_min.append(s_space[np.argmin(plots[i])])
# plt.xticks(s_min)
plt.ylabel(r'Off-diagonal norm $||\sigma(\hat H_{k+1})||$')
plt.xlabel(r'DBR duration $s$')
plt.legend()
plt.savefig('group_commutator_XXZ.pdf')