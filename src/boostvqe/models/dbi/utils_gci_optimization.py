import hyperopt

from qibo.backends import _check_backend
from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.utils import *

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
    lr: float = 1e-2,
    step_guess = 0.01
):

    nqubits = dbi_object.nqubits

    d_params_store = []
    s_store = []
    loss_store = []


    eo_d = MagneticFieldEvolutionOracle(d_params_init)
    step_guess,loss,_ = dbi_object.choose_step(d = eo_d,step_min=0.018,step_max=0.026, max_evals=11)
    print(loss)
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
        step_guess, loss,_ = dbi_object.choose_step(d=d,step_min=0.014,step_max=0.025, max_evals=15,verbose=False)
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
    return [4-sin(x/3) for x in range(10)]

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