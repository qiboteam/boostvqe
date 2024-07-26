import qibo
from qibo import gates, get_backend
from qibo.backends import construct_backend
from qibo.models import Circuit

from boostvqe.training_utils import vqe_loss

def connect_qubits(circuit, jumpsize=1, start_from=0):
    def get_circular_index(n, index):
        circular_index = index % n
        return circular_index
    for q in range(start_from, circuit.nqubits, jumpsize+1):
        ctrl_index = q
        targ_index = get_circular_index(circuit.nqubits, q+jumpsize)
        circuit.add(gates.RBS(q0=ctrl_index, q1=targ_index, theta=0.))
    return circuit


def hdw_efficient(nqubits, nlayers):
    """Build qibo's aavqe example circuit."""

    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.RZ(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(0, nqubits - 1, 2))
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.RZ(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(1, nqubits - 2, 2))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add(gates.RY(q, theta=0) for q in range(nqubits))

    return circuit
            

def hw_preserving(nqubits, nlayers=1):

    if nqubits%2 != 0:
        raise_error(
            ValueError,
            "To use this ansatz please be sure number of qubits is even."
    )
    c = Circuit(nqubits)

    for q in range(int(nqubits/2)):
        c.add(gates.X(q))

    for _ in range(int(nlayers)):
        c = connect_qubits(c, jumpsize=1, start_from=0)
        c = connect_qubits(c, jumpsize=1, start_from=1)
        c = connect_qubits(c, jumpsize=2, start_from=0)
        c = connect_qubits(c, jumpsize=2, start_from=1)
        c = connect_qubits(c, jumpsize=2, start_from=3)

    return c

def su2_preserving(nqubits, nlayers):
    """SU2 invariant circuit."""
    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(1, nqubits, 2):
            c.add(gates.RXX(q0=q, q1=(q+1)%nqubits, theta=0.))
            c.add(gates.RYY(q0=q, q1=(q+1)%nqubits, theta=0.))
            c.add(gates.RZZ(q0=q, q1=(q+1)%nqubits, theta=0.))
        for q in range(0, nqubits, 2):
            c.add(gates.RXX(q0=q, q1=(q+1)%nqubits, theta=0.))
            c.add(gates.RYY(q0=q, q1=(q+1)%nqubits, theta=0.))
            c.add(gates.RZZ(q0=q, q1=(q+1)%nqubits, theta=0.))
    return c

def compute_gradients(parameters, circuit, hamiltonian):
    """
    Compute gradients of circuit's parameters to check the problem trainability.
    The evaluated derivatives are the ones of the expectation of `hamiltonian`
    over the final state get running `circuit.execute` w.r.t. rotational angles.

    """
    tf_backend = construct_backend("tensorflow")
    parameters = tf_backend.tf.Variable(parameters, dtype=tf_backend.tf.float64)

    with tf_backend.tf.GradientTape() as tape:
        circuit.set_parameters(parameters)
        final_state = tf_backend.execute_circuit(circuit).state()
        expectation = tf_backend.calculate_expectation_state(
            tf_backend.cast(hamiltonian.matrix), final_state, normalize=False
        )

    return hamiltonian.backend.cast(tape.gradient(expectation, parameters))


class VQE:
    """This class implements the variational quantum eigensolver algorithm."""

    from qibo import optimizers

    def __init__(self, circuit, hamiltonian):
        """Initialize circuit ansatz and hamiltonian."""
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.backend = hamiltonian.backend

    def minimize(
        self,
        initial_state,
        method="Powell",
        loss_func=None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        compile=False,
        processes=None,
    ):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                See :meth:`qibo.optimizers.optimize` for available optimization
                methods.
            loss (callable): loss function, the default one is :func:`qibo.models.utils.vqe_loss`.
            jac (dict): Method for computing the gradient vector for scipy optimizers.
            hess (dict): Method for computing the hessian matrix for scipy optimizers.
            hessp (callable): Hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds (sequence or Bounds): Bounds on variables for scipy optimizers.
            constraints (dict): Constraints definition for scipy optimizers.
            tol (float): Tolerance of termination for scipy optimizers.
            callback (callable): Called after each iteration for scipy optimizers.
            options (dict): a dictionary with options for the different optimizers.
            compile (bool): whether the TensorFlow graph should be compiled.
            processes (int): number of processes when using the paralle BFGS method.

        Return:
            The final expectation value.
            The corresponding best parameters.
            The optimization result object. For scipy methods it returns
            the ``OptimizeResult``, for ``'cma'`` the ``CMAEvolutionStrategy.result``,
            and for ``'sgd'`` the options used during the optimization.
        """
        if loss_func is None:
            loss_func = vqe_loss
        if compile:
            loss = self.hamiltonian.backend.compile(loss_func)
        else:
            loss = loss_func

        if method == "cma":
            dtype = self.hamiltonian.backend.np.float64
            loss = lambda p, c, h: dtype(loss_func(p, c, h))
        elif method != "sgd":
            loss = lambda p, c, h: self.hamiltonian.backend.to_numpy(loss_func(p, c, h))

        result, parameters, extra = self.optimizers.optimize(
            loss,
            initial_state,
            args=(self.circuit, self.hamiltonian),
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
            compile=compile,
            processes=processes,
            backend=self.hamiltonian.backend,
        )
        self.circuit.set_parameters(parameters)
        return result, parameters, extra

    def energy_fluctuation(self, state):
        """Compute Energy Fluctuation."""
        return self.hamiltonian.energy_fluctuation(state)
