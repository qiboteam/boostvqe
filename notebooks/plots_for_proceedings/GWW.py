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
models = ["TFIM", "XXZ"]
params = [3, 0.5]
NSTEPS = 10

fig, ax = plt.subplots(1,2, figsize=(8,3.5), sharex=False)

for i, model in enumerate(models):
    dbi = initialize_dbi(nqubits, model, params[i])
    dbi.cost = DoubleBracketCostFunction.off_diagonal_norm
    dbi.mode = DoubleBracketGeneratorType.single_commutator
    dbi.scheduling = DoubleBracketScheduling.grid_search
    loss_list = []
    s_plot_list = []
    for GWW in [False, True]:
        dbi_eval = deepcopy(dbi)
        ls = [dbi.off_diagonal_norm]
        ss = [] 
        print(GWW)
        for _ in range(NSTEPS):
            if GWW is True:
                label = "GWW"
                d = dbi_eval.diagonal_h_matrix
            else:
                d = dbi.diagonal_h_matrix
                label = r"$\Delta(H_0)$"
            s = dbi_eval.choose_step(d=d)
            dbi_eval(s, d=d)
            ls.append(dbi_eval.off_diagonal_norm)
            ss.append(s)
        ax[i].plot(s_to_plot(ss), ls, label=label, marker='.')
        if model == "TFIM":
            ax[i].set_title(f'{model} $n$={nqubits} $h$={params[i]}')
        else:
            ax[i].set_title(f'{model} $n$={nqubits} $\delta$={params[i]}')
        ax[i].set_xlabel(r"DBR duration $s$")  
        ax[i].legend()
ax[0].set_ylabel(r'$||\sigma(\hat H)||$')
# ax[1].set_xlabel(r"DBR duration $s$")   
plt.tight_layout()
# fig.subplots_adjust(right=0.8, bottom=0.2)  
plt.savefig('GWW.pdf')
    