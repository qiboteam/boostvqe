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

dbi = initialize_dbi(5, "XXZ", 0.5)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
poly_fit_ls = []
n_ls = [2,8,10]
s_space = np.linspace(1e-3, 0.2, 120)
loss_ls = []
# actual losses
for s in s_space:
    loss_ls.append(dbi.loss(step=s, d=dbi.diagonal_h_matrix))
# polynomial fits
for n in n_ls:
    coef = np.real(off_diagonal_norm_polynomial_expansion_coef(dbi, d=dbi.diagonal_h_matrix, n=n))
    coef_int = [c/(i+1) for i,c in enumerate(reversed(coef))]
    poly_fit_ls.append([np.sqrt(sum(c*s**(i+1) for i,c in enumerate(coef_int))+dbi.off_diagonal_norm**2) for s in s_space])
    print(polynomial_step(dbi, n=n, d=dbi.diagonal_h_matrix, n_max=10))
    

crop_iter = -1
colors = plt.get_cmap('tab10').colors
run_param_rc(15)
plt.plot(s_space[:crop_iter], loss_ls[:crop_iter], label="DBR")
min_s = [0]
min_s.append(s_space[loss_ls.index(min(loss_ls[:crop_iter]))])
for i,n in enumerate(n_ls):
    color = colors[(i+1) % len(colors)]
    plt.plot(s_space[:crop_iter], poly_fit_ls[i][:crop_iter], label=f"Degree $n$={n}", linestyle='--', color=color)
    min_s.append(s_space[poly_fit_ls[i].index(min(poly_fit_ls[i][:crop_iter]))])
    # plt.axvline(s_space[poly_fit_ls[i].index(min(poly_fit_ls[i][:crop_iter]))], color=color, linestyle='--')
plt.xticks(min_s)
plt.legend()
plt.xlabel(r"DBR duration $s$")
plt.ylabel(r"Off-diagonal norm $||\sigma(e^{sW}He^{-sW})||$")
plt.savefig("polynomial_fit.pdf") 