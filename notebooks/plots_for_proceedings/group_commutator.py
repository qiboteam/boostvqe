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

fig, ax = plt.subplots(1,2, figsize=(10,4))
def plot_sigma_time(dbi, d, mode, s_space):
    dbi.mode = mode
    return [dbi.loss(step=s, d=d) for s in s_space]

dbi = initialize_dbi(5, "TFIM", 3)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
s_space = np.linspace(1e-4,0.5,100)
modes = [DoubleBracketGeneratorType.single_commutator,
         DoubleBracketGeneratorType.group_commutator,
         DoubleBracketGeneratorType.group_commutator_3]
plots = []
for mode in modes:
    plots.append(plot_sigma_time(dbi, dbi.diagonal_h_matrix, mode, s_space))

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
mode_names = [r'$e^{-s\hat W_k}$',
              r'$\hat V_0^{(\text{GC})}$',
              r'$\hat Q_0^{(\text{HOPF})}$']
s_min = []
for i,mode in enumerate(modes):
    ax[0].plot(s_space, plots[i], label=mode_names[i])
    s_min.append(s_space[np.argmin(plots[i])])
    
dbi = initialize_dbi(5, "XXZ", 0.5)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
plots = []
for mode in modes:
    plots.append(plot_sigma_time(dbi, dbi.diagonal_h_matrix, mode, s_space))
s_min = []
for i,mode in enumerate(modes):
    ax[1].plot(s_space, plots[i], label=mode_names[i])
    s_min.append(s_space[np.argmin(plots[i])])
    
    
a = -.17
b = .97
ax[0].annotate('a)', xy = (a,b), xycoords='axes fraction')
a = -.08
b = .97
ax[1].annotate('b)', xy = (a,b), xycoords='axes fraction')
# plt.xticks(s_min)
ax[0].set_ylabel(r'$||\sigma(e^{s\hat W}\hat H_0e^{-s\hat W})||$')
# plt.xlabel(r'DBR duration $s$')
fig.text(0.45, 0.02, r'DBR duration $s$', ha='center')    
# plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('group_commutator_TFIM.pdf')