import hyperopt

from qibo.backends import _check_backend
from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.utils import *

def diagonal_1_pauli_z(i,n):
    """See Eq. 3 in https://arxiv.org/abs/1707.05181"""
    i += 1

    block_size = 2**(n-i)

    plus_block = [1]*block_size
    minus_block = [-1]*block_size
    block = plus_block + minus_block

    return np.array(block * (2**(i-1)))

def diagonal_product_pauli_z(i_list,n):
    """Diagonal matrices remain diagonal following matrix multiplication"""

    diagonals = []
    for i in i_list:
       diagonals.append(diagonal_1_pauli_z(i%n,n))
    return np.prod(np.array(diagonals), axis = 0)    

def dephasing_approximation(h_matrix: np.array, order = 1):
    """finds the approximation long-range Ising decomposition of the diagonal of the hamiltonian `h_matrix`"""
    nqubits = int(np.log2(h_matrix.shape[0]))

    diagonal = np.diag(h_matrix)

    order_1 = [ diagonal.T @ diagonal_1_pauli_z(i,nqubits) / 2**nqubits for i in range(nqubits)]
    order_2 = [ diagonal.T @ diagonal_product_pauli_z([i,j],nqubits) / 2**nqubits for i,j in itertools.product(range(nqubits),range(nqubits))]
    return {1: order_1, 2: order_2}

def nn_dephasing_approximation(h_matrix: np.array, order = 1):
    """finds the approximation nearest neighbor Ising decomposition of the diagonal of the hamiltonian `h_matrix`"""
    nqubits = int(np.log2(h_matrix.shape[0]))

    diagonal = np.diag(h_matrix)

    order_1 = [ diagonal.T @ diagonal_1_pauli_z(i,nqubits) / 2**nqubits for i in range(nqubits)]
    order_2 = [ diagonal.T @ diagonal_product_pauli_z([i,i+1],nqubits) / 2**nqubits for i in range(nqubits)]
    return {1: order_1, 2: order_2}


def gradient_numerical_circuits(
    dbi_object: DoubleBracketIteration,
    d_params: list,
    parameterization: ParameterizationTypes,
    s: float = 1e-2,
    delta: float = 1e-3,
    backend=None,
    loss_0 = None,
    **kwargs,
):
    r"""
    Gradient of the DBI with respect to the parametrization of D. A simple finite difference is used to calculate the gradient.

    Args:
        dbi_object (DoubleBracketIteration): DoubleBracketIteration object.
        d_params (np.array): Parameters for the ansatz (note that the dimension must be 2**nqubits for full ansazt and nqubits for Pauli ansatz).
        s (float): A short flow duration for finding the numerical gradient.
        delta (float): Step size for numerical gradient.
    Returns:
        grad (np.array): Gradient of the D operator.
    """

    nqubits = dbi_object.nqubits
    grad = np.zeros(len(d_params))
    d = params_to_diagonal_operator(
        d_params, nqubits, parameterization=parameterization, **kwargs
    )
    if loss_0 is None:
        loss_0 = dbi_object.loss(s, d)
    for i in range(len(d_params)):
        params_new = deepcopy(d_params)
        params_new[i] += delta
        d_new = MagneticFieldEvolutionOracle(params_new)
        # find the increment of a very small step
        grad[i] = (dbi_object.loss(s, d_new) - loss_0 ) / delta
    return grad

def gradient_descent_circuits(
    dbi_object: DoubleBracketIteration,
    train_epochs: int,
    d_params_init: list,
    times_choose_step: list,
    lr: float = 1e-2,
    step_guess = 0.01
):

    nqubits = dbi_object.nqubits

    d_params_store = []
    s_store = []
    loss_store = []


    eo_d = MagneticFieldEvolutionOracle(d_params_init)
    step_guess,loss,_ = dbi_object.choose_step(d = eo_d, times = times_choose_step)

    d_params_test = d_params_init
    for i in range(train_epochs):
        print(i)
        # find gradient
        grad = gradient_numerical_circuits(
            dbi_object,
            d_params_test,
            ParameterizationTypes.circuits,
            s=step_guess,
            loss_0 = loss
        )
        d_params_test = [d_params_init[j] - grad[j] * lr for j in range(len(grad))]
        d = MagneticFieldEvolutionOracle(d_params_test)
        step_guess, loss,_ = dbi_object.choose_step(d=d,times = times_choose_step,verbose=False)
        print(loss)
        d_params_store.append(d_params_test)
        s_store.append(step_guess)
        loss_store.append(loss)

    min_loss = min(loss_store)
    idx_min = loss_store.index(min_loss)
    d_params_test = d_params_store[idx_min]
    s = s_store[idx_min]

    return d_params_test, s, min_loss


def gradient_descent_circuits_lr(
    dbi_object: DoubleBracketIteration,
    train_epochs: int,
    d_params_init: list,
    lr: float = 1e-2,
    step_guess = 0.01
):

    nqubits = dbi_object.nqubits

    d_params_store = []
    s_store = []
    loss_store = []


    d_init = params_to_diagonal_operator(
                d_params_init,
                nqubits,
                parameterization=ParameterizationTypes.circuits
            )
    s,loss = gci.choose_step(d = d_init, max_evals = 50)
    print(loss)
    d_params_test = d_params_init
    for i in range(train_epochs):
        print(i)
        # find gradient
        grad = gradient_numerical_circuits(
            dbi_object,
            d_params_test,
            ParameterizationTypes.circuits,
            s=0.004,
            loss_0 = loss
        )
        d_params_store_lr = []
        s_store_lr = []
        loss_store_lr = []
        for lr in np.linspace(1,1e3,9):
            d_params_test = [d_params_init[j] - grad[j] * lr for j in range(len(grad))]
            d = params_to_diagonal_operator(
                d_params_test,
                nqubits,
                parameterization=ParameterizationTypes.circuits
            )
            #step_guess, loss = dbi_object.choose_step(d=d,step_min=1e-5,step_max=0.1, max_evals=35,verbose=False)
            print(loss)

            d_params_store_lr.append(d_params_test)
            s_store_lr.append(step_guess)
            loss_store_lr.append(loss)
        # store values
        loss = min(loss_store_lr)

        idx_min = loss_store_lr.index(loss)

        d_params_test = d_params_store_lr[idx_min]  
        d_params_store.append(d_params_test)      
        step_guess = s_store_lr[idx_min]
        s_store.append(step_guess)
        loss_store.append(loss)

        print(loss_store_lr[idx_min])
        # choose the minimum loss from store
    min_loss = min(loss_store)
    idx_min = loss_store.index(min_loss)
    d_params_test = d_params_store[idx_min]
    s = s_store[idx_min]

    return d_params_test, s, min_loss


def get_gd_evolution_oracle(n_local, params):
    if n_local == 1:
        return MagneticFieldEvolutionOracle(params)
    elif n_local == 2:
        return IsingNNEvolutionOracle(params[:int(len(params)/2)], params[int(len(params)/2):])
    else:
        raise_error(ValueError, "Model not supported, change `n_local` to values in [1, 2]")
     
def gradient_numerical(
    loss_function,
    n_local,
    params: list, 
    loss_0,
    s_0,
    delta: float = 1e-3,
):
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params_new = deepcopy(params)
        params_new[i] += delta
        eo_d = get_gd_evolution_oracle(n_local, params_new)
        # find the increment of a very small step
        grad[i] = (loss_function(s_0, eo_d) - loss_0 ) / delta
        
    # normalize gradient
    grad = grad / max(abs(grad))
    return grad


from boostvqe.models.dbi.utils_scheduling import adaptive_binary_search
def choose_gd_params(gci,
                     n_local,
                     params,
                     loss_0,
                     s_0,
                     s_min=1e-4,
                     s_max=2e-2,
                     lr_min=1e-4,
                     lr_max=1,
                     threshold=1e-4,
                     max_eval=30,
                     please_be_adaptive = True,
                     please_be_verbose = False
                     ):
    evaluated_points = {}
    max_eval = int(np.sqrt(max_eval))
    grad = gradient_numerical(gci.loss, n_local, params, loss_0, s_0)

    def loss_func_lr(lr):
        filtered_entries = {k:v for k, v in evaluated_points.items() if k[0]==lr}
        if len(filtered_entries) > 0:
            return min(filtered_entries.values())
        elif lr < 0:
            return float("inf")
        else:
            params_eval = (1 - grad*lr) * deepcopy(params)
            eo_d = get_gd_evolution_oracle(n_local, params_eval)
            # given lr find best possible s_min and loss_min
            if please_be_adaptive:
                best_s, best_loss, evaluated_points_s, exit_criterion = adaptive_binary_search(lambda s: gci.loss(s, eo_d) if s>0 else float("inf"), 
                                                                                               threshold, s_min, s_max, max_eval)
            else:
                best_s, best_loss, evaluated_points_s = gci.choose_step(d = eo_d,s_min=1e-4, s_max=2e-2, max_eval = max_eval)

            for s,l in evaluated_points_s.items():
                if l < 0:
                    evaluated_points[(lr, s)] = l
            if please_be_verbose:
                print(f"For lr = {lr} found optimal s = {best_s} yielding loss = {best_loss} after terminating with {exit_criterion}")
        return best_loss
    
    best_lr, best_loss, _ ,exit_criterion = adaptive_binary_search(loss_func_lr, threshold, lr_min, lr_max, max_eval)
    if please_be_verbose:
        print(f"For lr = {lr} found optimal s = {best_s} yielding loss = {best_loss} after terminating with {exit_criterion}")
    best_params = (1-grad*best_lr)*deepcopy(params)
    eo_d = get_gd_evolution_oracle(n_local, best_params)
    # find best_s
    for (lr, s), loss in evaluated_points.items():
        if lr == best_lr and loss == best_loss:
            best_s = s
    return eo_d, best_s, best_loss, evaluated_points, best_params, best_lr



def evaluate_histogram_data(gci_object, input_field, nmb_shots=10, randomization_amp = 0.2):
    #  this is quite simple - just pass a diagonal SymbolicHamiltonian 
    # and because it will be commuting we can use the member function circuit for compiling
    fields = []
    losses = []
    steps = []
    for j in range(nmb_shots):
        field = [a+b for a,b in zip(np.random.rand(len(input_field))*randomization_amp,input_field)]
        eo_d = MagneticFieldEvolutionOracle(field,name = "D(linear)")
        step,loss, _ = gci_object.choose_step(d = eo_d,max_evals=34,step_min = 0.0051,step_max = 0.03)
        losses.append(loss)
        fields.append(field)
        steps.append(step)

    plt.hist(losses)
    print(np.min(losses))
    f = fields[np.argmin(losses)]
    s = steps[np.argmin(losses)]
    return f,s, np.min(losses), fields, steps, losses

def get_linspace_approximation_field():
    return [-0.25024438, -0.12512219, -0.06256109, -0.03128055, -0.01564027, -0.00782014,
 -0.00391007, -0.00195503, -0.00097752, -0.00048876]

def get_mareks_favorite_field():
    return [1+np.sin(x/3) for x in range(10)]

def get_best_exhaustive_10q_7l():
    return [1.0074343565408288,
 1.1483550802051337,
 1.1152574570741454,
 1.1207635311034556,
 1.0290334907280279,
 1.0582973691183888,
 1.0009000697562873,
 1.0467850494221564,
 1.0291778083015108,
 1.180909625995888]