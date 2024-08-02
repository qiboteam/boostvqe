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

run_param_rc(15)
fig, ax = plt.subplots(figsize=(8,5.5))
dbi = initialize_dbi(5, "XXZ", 0.5)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
dbi.mode = DoubleBracketGeneratorType.single_commutator
nqubits = dbi.h.nqubits

d_names = [
            'GWW',
           'Pauli-Z products',
           'Entry-wise (GD)',
           'Magnetic (GD)',
           'ATA Ising (GD)',
           ]

NSTEPS = 15
loss_list = []
s_plot_list = []

plot_matrix_inset([0.4, 0.7, 0.25, 0.25], ax, np.abs(dbi.h.matrix))
# ax.annotate(' ', xy=(0.8,17), xytext=(0.05,18), arrowprops=dict(color='black'))

# GWW
dbi_eval = deepcopy(dbi)
dbi_eval.scheduling = DoubleBracketScheduling.grid_search
ls = [dbi_eval.off_diagonal_norm]
ss = []
for _ in range(NSTEPS):
    d = dbi_eval.diagonal_h_matrix
    s = dbi_eval.choose_step(d=d)
    dbi_eval(s, d=d)
    ls.append(dbi_eval.off_diagonal_norm)
    ss.append(s)
loss_list.append(ls)
s_plot_list.append(s_to_plot(ss))
final_gww = deepcopy(dbi_eval.h.matrix)



# pauli-Z
generate_local_Z = generate_Z_operators(nqubits)
Z_ops = list(generate_local_Z.values())
Z_names = list(generate_local_Z.keys())
dbi_eval = deepcopy(dbi)
ls = [dbi.off_diagonal_norm]
ss = [] 
for _ in range(NSTEPS):
    dbi_eval, idx, step, flip_sign = select_best_dbr_generator(dbi_eval, Z_ops, compare_canonical=False)
    d = Z_ops[idx]
    ls.append(dbi_eval.off_diagonal_norm)
    ss.append(step)
loss_list.append(ls)
s_plot_list.append(s_to_plot(ss))
final_pauli = deepcopy(dbi_eval.h.matrix)
plot_matrix_inset([0.6,0.09, 0.25, 0.25], ax, np.abs(final_pauli))
# ax.annotate(' ', xy=(1.1,5), xytext=(s_to_plot(ss)[-1],ls[-1]), arrowprops=dict(color='black'))

# gradient descent 
parameterizations = [ParameterizationTypes.computational,ParameterizationTypes.pauli, ParameterizationTypes.pauli]
pauli_op_dict = generate_pauli_operator_dict(nqubits, 2)
seed = 10
d_params = [np.random.random(2**nqubits), np.random.random(nqubits), np.random.random(int(nqubits*(nqubits-1)/2)) ]
for i,parameterization in enumerate(parameterizations):
    dbi_eval = deepcopy(dbi)
    dbi_eval.scheduling = DoubleBracketScheduling.grid_search
    ls = [dbi.off_diagonal_norm]
    ss = [] 
    d_param = d_params[i]
    for _ in range(NSTEPS):
        d_param, d, s = gradient_descent(dbi_eval, d_param, parameterization, pauli_op_dict, pauli_parameterization_order=i)
        ss.append(s)
        dbi_eval(step=s, d=d)
        ls.append(dbi_eval.off_diagonal_norm)
    loss_list.append(ls)
    s_plot_list.append(s_to_plot(ss))

for i, d_name in enumerate(d_names):
        ax.plot(s_plot_list[i], loss_list[i], label=d_name, marker='.')
    # plt.plot(loss_list[i], label=d_name, marker='.')
ax.set_ylabel(r'$||\sigma(\hat H_k)||$')
ax.set_xlabel(r'DBR duration $s$')
ax.axhline(5.4, linestyle='--', label="min(BHMM)", color='0.8', linewidth=1)
# plt.xlabel('Iterations')
# ax.legend(bbox_to_anchor=(1, -0.2),shadow=False, ncol=3)
ax.legend()
# plt.tight_layout() 
plt.savefig('variational_XXZ_5_0.5.pdf')