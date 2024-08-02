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
from utils import *

dbi = initialize_dbi(5, "XXZ", 0.5)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
dbi.mode = DoubleBracketGeneratorType.single_commutator
nqubits = dbi.h.nqubits
pauli_operator_dict_2 = generate_pauli_operator_dict(nqubits, 2)
constant_magnetic_coef = [1] * nqubits
linear_magnetic_coef = np.linspace(1, nqubits+1, nqubits)
quadratic_magnetic_coef = [x**2/nqubits for x in linear_magnetic_coef]
ising_coef = [0] * nqubits + [1 if abs(y - x) == 1 else 0 for x, y in list(pauli_operator_dict_2)[dbi.h.nqubits:]]

seed = 12
d_list_1 = [
    min_max_diag(dbi), 
    max_min_diag(dbi), 
    min_max_shuffle(dbi, seed), 
    min_max_sorted_random(dbi, seed), 
    eigen_diag(dbi),
    dbi.diagonal_h_matrix
]
d_names_1 = [
    r' Min-max',
    r' Max-min',
    r' Shuffled min-max',
    r' Sampled min-max',
    r' Eigenvalues spec$(\hat H_0)$',
    r' Dephasing $\Delta(\hat H_0)$'
]
colors_1 = [
    'C3', 'C1', 'C2', 'C8', 'C5', 'C0'
]
markers_1 = [
    'o', 'x', 'h', 'H', 'p', '*'
]
d_list_2 = [
        # magnetic_diag(dbi, constant_magnetic_coef, pauli_operator_dict_2),
        magnetic_diag(dbi, linear_magnetic_coef, pauli_operator_dict_2),
        magnetic_diag(dbi, quadratic_magnetic_coef, pauli_operator_dict_2),
        ising_diag(dbi, ising_coef, pauli_operator_dict_2),
        dbi.diagonal_h_matrix
]
d_names_2 = [
        # 'Constant magnetic',
        r' Linear magnetic',
        r' Quadratic magnetic',
        r' Constant NN-OBC Ising',
        r' Dephasing $\Delta(\hat H)$'
]
colors_2 = [
    # 'magenta',
    'orchid', 'pink', 'C9', 'C0'
]
markers_2 = [
    's', 'd', '^', '*'
]
run_param_rc(20)
fig, ax = plt.subplots(2,2, figsize=(14,10), sharey='row', sharex=False)

# plot DBR
s_space = np.linspace(0, 0.2, 100)
x_ticks_1 = [0]
y_ticks = [round(dbi.off_diagonal_norm,1)]
for i, d in enumerate(d_list_1):
    ls = [dbi.loss(d=d, step=s) for s in s_space]
    ls_min = min(ls)
    s_min = s_space[ls.index(ls_min)]
    x_ticks_1.append(round(s_min,3))
    y_ticks.append(round(ls_min,1))
    ax[0][0].plot(s_space, ls, label=d_names_1[i], color=colors_1[i])
    ax[0][0].plot(s_min, ls_min, color=colors_1[i], marker=markers_1[i], fillstyle='none' if i<2 else None)
ax[0][0].set_xticks(remove_duplicate_and_nearby_elements(x_ticks_1, 0.02))
ax[0][0].set_yticks(remove_duplicate_and_nearby_elements(y_ticks, 0.5))
ax[0][0].grid()
ax[0][0].set_ylabel(r'$||\sigma(e^{s\hat W}\hat H_0e^{-s\hat W})||$')
a = -.17
b = .97
ax[0][0].annotate('a)', xy = (a,b), xycoords='axes fraction')
 
 
x_ticks_2 = [0] 
for i, d in enumerate(d_list_2):
    ls = [dbi.loss(d=d, step=s) for s in s_space]
    ax[0][1].plot(s_space, ls, label=d_names_2[i], color=colors_2[i])
    ls_min = min(ls)
    s_min = s_space[ls.index(ls_min)]
    x_ticks_2.append(round(s_min,3))
    y_ticks.append(round(ls_min,1))
    ax[0][1].plot(s_min, ls_min, color=colors_2[i], marker=markers_2[i])
ax[0][1].set_xticks(remove_duplicate_and_nearby_elements(x_ticks_2, 0.021))
ax[0][1].set_yticks(remove_duplicate_and_nearby_elements(y_ticks, 0.3))
ax[0][1].grid()
a = -.08
b = .97
ax[0][1].annotate('b)', xy = (a,b), xycoords='axes fraction')

# plot DBI

last_loss = [round(dbi.off_diagonal_norm,1)]
last_s_1 = [0]
NSTEPS = 15
for i,d in enumerate(d_list_1):
    dbi_eval = deepcopy(dbi)
    dbi_eval.scheduling = DoubleBracketScheduling.grid_search
    loss = [dbi_eval.off_diagonal_norm]
    s = []
    for _ in range(NSTEPS):
        s_min = dbi_eval.choose_step(d=d)
        dbi_eval(d=d, step=s_min)
        s.append(s_min)
        loss.append(dbi_eval.off_diagonal_norm)
    # s = cut_list_at_threshold(s_to_plot(s), 1.6)
    s = s_to_plot(s)
    if i == 3:
        print('sample s :', s)
    if i > 1:
        last_loss.append(round(loss[-1],1))
        last_s_1.append(round(s[-1],2))
    if i < 2: 
        ax[1][0].plot(s, loss[:len(s)], label=d_names_1[i], marker=markers_1[i], fillstyle='none', color=colors_1[i])
    else:
        ax[1][0].plot(s, loss[:len(s)], label=d_names_1[i], marker=markers_1[i], color=colors_1[i])
    ls_min = min(loss)

ax[1][0].set_xticks(remove_duplicate_and_nearby_elements(last_s_1, 0.1))
ax[1][0].grid()
# ax[1][0].axhline(ls_min, color='grey', linestyle='--')
ax[1][0].set_ylabel(r'$||\sigma(\hat H_k)||$')
ax[1][0].legend()
a = -.17
b = .97
ax[1][0].annotate('c)', xy = (a,b), xycoords='axes fraction')


last_s_2 = [0]
for i,d in enumerate(d_list_2):
    dbi_eval = deepcopy(dbi)
    dbi_eval.scheduling = DoubleBracketScheduling.grid_search
    loss = [dbi_eval.off_diagonal_norm]
    s = []
    for _ in range(NSTEPS):
        s_min = dbi_eval.choose_step(d=d)
        dbi_eval(d=d, step=s_min)
        s.append(s_min)
        loss.append(dbi_eval.off_diagonal_norm)
    # s = cut_list_at_threshold(s_to_plot(s), 0.6)
    s = s_to_plot(s)
    if i < 2:
        print('linear and quadratic s :', s)
    if i > 1:
        last_loss.append(round(loss[-1],1))
        last_s_2.append(round(s[-1],2))
    # s = s_to_plot(s)
    ax[1][1].plot(s, loss[:len(s)], label=d_names_2[i], marker=markers_2[i], color=colors_2[i])
# plot minimum achieved by dephasing
ax[1][1].set_xticks(remove_duplicate_and_nearby_elements(last_s_2, 0.04))
ax[1][1].set_yticks(remove_duplicate_and_nearby_elements(last_loss, 0.5))
ax[1][1].grid()
# ax[1][1].axhline(ls_min, color='grey', linestyle='--', linewidth=1)
ax[1][1].legend()
a = -.08
b = .97
ax[1][1].annotate('d)', xy = (a,b), xycoords='axes fraction')

fig.text(0.45, 0.12, r'DBR duration $s$', ha='center')    
plt.tight_layout()
fig.subplots_adjust(right=0.9, bottom=0.2)  
plt.savefig('BHMM_re_XXZ.pdf')   