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

nqubits = 5
model = "XXZ"
param = 0.5
NSTEPS = 5
scheduling_names = ["grid search", "hyperopt", "stimulated annealing", "analytical"]
schedulings = [
    DoubleBracketScheduling.grid_search,
    DoubleBracketScheduling.hyperopt,
    DoubleBracketScheduling.simulated_annealing,
    DoubleBracketScheduling.polynomial_approximation,
]

fig, ax = plt.subplots(1, 2, figsize=(8,4.2), sharex=False)

# plot 1: first step
s_space = np.linspace(1e-3, 0.5, 120)
dbi = initialize_dbi(nqubits, model, param)
dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
dbi.mode = DoubleBracketGeneratorType.canonical
ls = []
for s in s_space:
    ls.append(dbi.loss(step=s))
ax[0].plot(s_space, ls, linestyle='--')
for j, scheduling in enumerate(schedulings):
    dbi = initialize_dbi(nqubits, model, param)
    dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
    dbi.mode = DoubleBracketGeneratorType.canonical
    dbi.scheduling = scheduling
    if scheduling_names[j] is "analytical":
        s = dbi.choose_step(n=8)
    else:
        s = dbi.choose_step(step_max=0.1)
    ax[0].axvline(s, label=scheduling_names[j], linestyle='-', color=f'C{j+1}')
ax[0].set_xlabel(r"DBR duration $s$")
ax[0].set_ylabel(r"$||\sigma(e^{sW}H_0e^{-sW})||$")
# ax[0].set_xticks([s,])

# plot 2: sigma-decrease

for j, scheduling in enumerate(schedulings):
    dbi = initialize_dbi(nqubits, model, param)
    dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
    dbi.mode = DoubleBracketGeneratorType.canonical
    dbi.scheduling = scheduling
    ls = [dbi.off_diagonal_norm]
    ss = []
    for _ in range(NSTEPS):
        d = dbi.diagonal_h_matrix
        if scheduling_names[j] is "analytical":
            s = dbi.choose_step(d=d, n=8)
        else:
            s = dbi.choose_step(d=d, step_min=1e-3, step_max=0.1)
        dbi(s, d=d)
        ls.append(dbi.off_diagonal_norm)
        ss.append(s)
    ax[1].plot(s_to_plot(ss), ls, label=scheduling_names[j], marker='.', color=f'C{j+1}')

ax[1].set_xlabel(r"DBR duration $s$")  
ax[1].set_ylabel(r'$||\sigma(\hat H)||$')
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', shadow=False, ncol=5)
# ax[1].set_xlabel(r"DBR duration $s$")   
plt.tight_layout()
fig.subplots_adjust(right=0.95, bottom=0.2)  
plt.savefig('scheduling_XXZ.pdf')
        
