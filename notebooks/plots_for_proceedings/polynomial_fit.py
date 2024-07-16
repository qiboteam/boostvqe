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

def initialize_dbi(nqubits, model, param):
    if model == "XXZ":
        hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=param)
    if model == "TFIM":
        hamiltonian = hamiltonians.TFIM(nqubits=nqubits, h=param)
    dbi = DoubleBracketIteration(hamiltonian=hamiltonian)
    return dbi

dbi = initialize_dbi(6, "TFIM", 4)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
poly_fit_ls = []
n_ls = [2,8,10]
s_space = np.linspace(1e-3, 0.05, 120)
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
plt.plot(s_space[:crop_iter], loss_ls[:crop_iter], label="DBR")
plt.axvline(s_space[loss_ls.index(min(loss_ls[:crop_iter]))])
for i,n in enumerate(n_ls):
    color = colors[(i+1) % len(colors)]
    plt.plot(s_space[:crop_iter], poly_fit_ls[i][:crop_iter], label=f"Degree $n$={n}", color=color)
    plt.axvline(s_space[poly_fit_ls[i].index(min(poly_fit_ls[i][:crop_iter]))], color=color, linestyle='--')
plt.legend()
plt.xlabel(r"DBR duration $s$")
plt.ylabel(r"Off-diagonal norm $||\sigma(e^{sW}He^{-sW})||$")
plt.savefig("polynomial_fit.pdf") 