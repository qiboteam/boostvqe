import numpy as np
from qibo import hamiltonians, set_backend
from boostvqe.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from boostvqe.models.dbi.utils import *
from boostvqe.models.dbi.utils_scheduling import *
from boostvqe.models.dbi.utils_dbr_strategies import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qibo import hamiltonians
from boostvqe.compiling_XXZ import *
import random


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

def max_min_diag(dbi):
    diag = np.diag(dbi.diagonal_h_matrix)
    d_min = min(diag)
    d_max = max(diag)
    d_ls = np.linspace(d_max, d_min, 2**dbi.h.nqubits)
    return np.diag(d_ls)

def min_max_shuffle(dbi, seed):
    diag = np.diag(dbi.diagonal_h_matrix)
    d_min = min(diag)
    d_max = max(diag)
    random.seed(seed)
    d_ls = sorted(np.linspace(d_max, d_min, 2**dbi.h.nqubits), key=lambda x:random.random())
    return np.diag(d_ls)

def min_max_sorted_random(dbi, seed):
    diag = np.diag(dbi.diagonal_h_matrix)
    d_min = min(diag)
    d_max = max(diag)
    min_max = np.linspace(d_max, d_min, 2**dbi.h.nqubits)
    random.seed(seed)
    d_ls = sorted([random.choice(list(min_max)) for _ in range(2**dbi.h.nqubits)])
    print(d_ls)
    return np.diag(d_ls)

def run_param_rc(fontsize = 30):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['font.size'] = fontsize
        #set_matplotlib_formats('pdf', 'png')
        plt.rcParams['savefig.dpi'] = 75
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.labelspacing'] = .3
        plt.rcParams['legend.columnspacing']= .3
        plt.rcParams['legend.handletextpad']= .1
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.serif'] = "cm"
        
def initialize_dbi(nqubits, model, param):
    if model == "XXZ":
        hamiltonian = hamiltonians.XXZ(nqubits=nqubits, delta=param)
    if model == "TFIM_qibo":
        hamiltonian = hamiltonians.TFIM(nqubits=nqubits, h=param)
    if model == "TFIM":
        hamiltonian = h_TFIM(nqubits=nqubits, h=param)
    dbi = DoubleBracketIteration(hamiltonian=hamiltonian)
    return dbi

def remove_list_duplicate(ls):
    return list(np.sort(list(set(ls))))

def remove_nearby_elements(ls, threshold):
    result = []
    for number in ls:
        if not result or abs(number - result[-1]) > threshold:
            result.append(number)
    return result

def remove_duplicate_and_nearby_elements(ls, threshold=0.1):
    return remove_nearby_elements(remove_list_duplicate(ls),threshold)

def plot_matrix_inset(axes, parent_axis, matrix):
    axin = parent_axis.inset_axes(axes)
    inset = axin.imshow(matrix, cmap='RdBu')

    divider = make_axes_locatable(axin)
    div_ax = divider.append_axes('right', size="8%", pad=0.05)
    cbar = plt.colorbar(inset, cax = div_ax )
    D = matrix.shape[0]
    axin.set_yticks(range(D))  
    axin.set_xticklabels([1]+['']*(D-2)+[D])
    axin.set_yticklabels([1]+['']*(D-2)+[D])
    axin.set_xticks(range(D))