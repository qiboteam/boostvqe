import numpy as np
from qibo import hamiltonians, set_backend
from boostvqe.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from boostvqe.models.dbi.utils import *
from boostvqe.models.dbi.utils_scheduling import *
from boostvqe.models.dbi.utils_dbr_strategies import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from qibo import hamiltonians
from boostvqe.compiling_XXZ import *
from utils import *

run_param_rc(15)
fig, ax = plt.subplots(1,2, figsize=(13,4.5), gridspec_kw={'width_ratios': [1.2,1]})
# plot a
NSTEPS = 6
cost_functions = [
    DoubleBracketCostFunction.off_diagonal_norm,
    DoubleBracketCostFunction.least_squares,
    DoubleBracketCostFunction.energy_fluctuation,
]
cost_names = [
    'Off-diagonal norm',
    'Least squares',
    'Energy fluctuation',
]
markers = [
     'o', 'x', '*'
]
sub_ax = inset_axes(
    parent_axes=ax[0],
    width="32%",
    height="32%",
    borderpad=1,  # padding between parent and inset axes
    loc='center right'
)
for i, cost in enumerate(cost_functions):
    dbi = initialize_dbi(5, "XXZ", 0.5)
    # d = eigen_diag(dbi)
    d = dbi.diagonal_h_matrix
    dbi.ref_state = np.zeros(2**dbi.h.nqubits)
    dbi.ref_state[8] = 1
    dbi.ref_state[3] = 0.5
    dbi.cost = cost
    loss_ls = [dbi.loss(step=0, d=d)]
    en_ls = [dbi.energy_fluctuation(state=dbi.ref_state)]
    least_ls = [dbi.least_squares(d=d)]
    norm_ls = [dbi.off_diagonal_norm]
    s_ls = []
    for _ in range(NSTEPS):
        s = dbi.choose_step(scheduling=DoubleBracketScheduling.hyperopt)
        s_ls.append(s)
        loss_ls.append(dbi.loss(step=s, d=d))
        dbi(step=s, d=d)
        en_ls.append(dbi.energy_fluctuation(state=dbi.ref_state))
        least_ls.append(dbi.least_squares(d=d))
        norm_ls.append(dbi.off_diagonal_norm)
    ax[0].plot(s_to_plot(s_ls), norm_ls, label=cost_names[i], marker=markers[i], fillstyle='none' if i!=2 else 'full')
    sub_ax.plot(s_to_plot(s_ls), en_ls, label=cost_names[i], marker=markers[i], fillstyle='none' if i!=2 else 'full')
ax[0].set_xlabel(r'DBR duration $s$')
sub_ax.set_xlabel(r'$s$')
ax[0].set_ylabel(r'Off-diagonal norm $||\sigma(\hat H)||$')
sub_ax.set_ylabel(r'$\Xi_k(\mu)$')
ax[0].legend(loc='lower left')
    
# plot b
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
ax[1].plot(s_space[:crop_iter], loss_ls[:crop_iter], label="DBR")
min_s = [0]
min_loss = []
min_s.append(s_space[loss_ls.index(min(loss_ls[:crop_iter]))])
for i,n in enumerate(n_ls):
    color = colors[(i+1) % len(colors)]
    ax[1].plot(s_space[:crop_iter], poly_fit_ls[i][:crop_iter], label=f"Degree $n$={n}", linestyle='--', color=color)
    min_loss.append(round(min(poly_fit_ls[i][:crop_iter]),1))
    min_s.append(round(s_space[poly_fit_ls[i].index(min(poly_fit_ls[i][:crop_iter]))],2))
    # plt.axvline(s_space[poly_fit_ls[i].index(min(poly_fit_ls[i][:crop_iter]))], color=color, linestyle='--')
ax[1].set_xticks(remove_duplicate_and_nearby_elements(min_s,0.01))
ax[1].set_yticks(remove_duplicate_and_nearby_elements(min_loss,0.1))
ax[1].grid()
ax[1].legend()
ax[1].set_xlabel(r"DBR duration $s$")
ax[1].set_ylabel(r"Off-diagonal norm $||\sigma(e^{sW}He^{-sW})||$")
plt.tight_layout()
fig.subplots_adjust(right=0.8, bottom=0.2)  
plt.savefig("fig1.pdf") 