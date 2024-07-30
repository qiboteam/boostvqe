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

def s_to_plot(s_list):
    s_plot = [0]*(len(s_list)+1)
    for i in range(len(s_list)):
        s_plot[i+1] = s_plot[i] + s_list[i]
    return s_plot

def cut_list_at_threshold(sorted_list, threshold):
    for i, element in enumerate(sorted_list):
        if element > threshold:
            return sorted_list[:i]
    return sorted_list

def min_max_diag(dbi):
    diag = np.diag(dbi.diagonal_h_matrix)
    d_min = min(diag)
    d_max = max(diag)
    d_ls = np.linspace(d_min, d_max, 2**dbi.h.nqubits)
    return np.diag(d_ls)

def eigen_diag(dbi):
    eigenvalues, _ = np.linalg.eig(dbi.h.matrix)
    return np.diag(np.sort(eigenvalues))

def ising_diag(dbi, coef, pauli_operator_dict_2, **kwargs):
    return params_to_diagonal_operator(coef, dbi.h.nqubits, ParameterizationTypes.pauli, 2, pauli_operator_dict=pauli_operator_dict_2, **kwargs)
    
def magnetic_diag(dbi, coef, pauli_operator_dict_2, **kwargs):
    return params_to_diagonal_operator(coef, dbi.h.nqubits, ParameterizationTypes.pauli, 1, pauli_operator_dict=pauli_operator_dict_2, **kwargs)

dbi = initialize_dbi(5, "XXZ", 0.5)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
dbi.mode = DoubleBracketGeneratorType.single_commutator
dbi.scheduling = DoubleBracketScheduling.grid_search
nqubits = dbi.h.nqubits
pauli_operator_dict_2 = generate_pauli_operator_dict(nqubits, 2)
constant_magnetic_coef = [1] * nqubits
linear_magnetic_coef = np.linspace(1, nqubits+1, nqubits)
quadratic_magnetic_coef = [x**2/nqubits for x in linear_magnetic_coef]
# ising_coef = [0] * nqubits + [0.5 if abs(y - x) == 1 or abs(y - x) == (nqubits-1) else 0 for x, y in list(pauli_operator_dict_2)[dbi.h.nqubits:]]
# ising_coef = [0] * nqubits + [0.5 for x, y in list(pauli_operator_dict_2)[dbi.h.nqubits:]]
ising_coef = [0] * nqubits + [1 if abs(y - x) == 1 else 0 for x, y in list(pauli_operator_dict_2)[dbi.h.nqubits:]]
d_list = [min_max_diag(dbi), 
          eigen_diag(dbi), 
          magnetic_diag(dbi, constant_magnetic_coef, pauli_operator_dict_2),
          magnetic_diag(dbi, linear_magnetic_coef, pauli_operator_dict_2),
          magnetic_diag(dbi, quadratic_magnetic_coef, pauli_operator_dict_2),
          ising_diag(dbi, ising_coef, pauli_operator_dict_2),
          dbi.diagonal_h_matrix
          ]
print(np.array_equal(dbi.diagonal_h_matrix, ising_diag(dbi, ising_coef, pauli_operator_dict_2) ))
d_names = ['Min_max',
           'Eigen',
           'Constant magnetic',
           'Linear magnetic',
           'Quadratic magnetic',
           'Constant Ising',
           r'$\Delta(\hat H)$']

# plot 1: first step
s_space = np.linspace(0, 0.2, 100)
loss_ls_1_step = []
for d in d_list:
    loss = []
    for s in s_space:
        loss.append(dbi.loss(step=s, d=d))
    loss_ls_1_step.append(loss)

fig, ax = plt.subplots(2, figsize=(8,6), sharex=False)

for i, d in enumerate(d_list):
    ax[0].plot(s_space, loss_ls_1_step[i], label=d_names[i])
    
# plot 2: DBI
dbi.scheduling = DoubleBracketScheduling.hyperopt
NSTEPS = 15
s_plot = []
loss_plot = []
for d in d_list:
    dbi_eval = deepcopy(dbi)
    loss = [dbi_eval.off_diagonal_norm]
    s = []
    for _ in range(NSTEPS):
        s_min = dbi_eval.choose_step(d=d)
        dbi_eval(d=d, step=s_min)
        s.append(s_min)
        loss.append(dbi_eval.off_diagonal_norm)
    s = cut_list_at_threshold(s_to_plot(s), 1.6)
    s_plot.append(s)
    loss_plot.append(loss[:len(s)])

for i, d in enumerate(d_list):
    ax[1].plot(s_plot[i], loss_plot[i], label=d_names[i], marker='.')

ax[0].set_ylabel(r'$||\sigma(e^{s\hat W}\hat H)e^{-s\hat W}||$')
ax[1].set_ylabel(r'$||\sigma(\hat H)||$')
ax[1].set_xlabel(r'DBR duration $s$')
ax[1].legend(bbox_to_anchor=(1, -0.2),shadow=False, ncol=4)
plt.tight_layout()
fig.subplots_adjust(right=0.8, bottom=0.2)  
plt.savefig('BHMM_DBI_XXZ_5_0.5.pdf')