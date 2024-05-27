import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from qibo import gates, hamiltonians
from qibo.backends import NumpyBackend, TensorflowBackend
from qibo.config import raise_error
from qibo.hamiltonians import AbstractHamiltonian
from qibo.symbols import Z


def vqe_loss(params, circuit, hamiltonian, nshots=None, delta=0.5):
    """
    Evaluate the hamiltonian expectation values of the
    circuit final state.

    TODO: fix the following statement.
    IMPORTANT: this works only for Heisemberg hamiltonians XXZ.
    """
    circ = circuit.copy(deep=True)
    circ.set_parameters(params)

    if isinstance(hamiltonian.backend, TensorflowBackend) and nshots is not None:
        expectation_value = _exp_with_tf(circ, hamiltonian, nshots, delta)
    elif nshots is None:
        expectation_value = _exact(circ, hamiltonian)
    else:
        expectation_value = _with_shots(circ, hamiltonian, nshots, delta)
    return expectation_value


def _exact(circ, hamiltonian):
    """Helper function to compute expectation value of Heisemberg hamiltonian."""
    expectation_value = hamiltonian.expectation(
        hamiltonian.backend.execute_circuit(circuit=circ).state()
    )
    return expectation_value


def _with_shots(circ, hamiltonian, nshots, delta=0.5, exec_backend=None):
    """Helper function to compute XXZ expectation value from frequencies."""

    # we may prefer run this on a different backend (e.g. with TF and PSR)
    if exec_backend is None:
        exec_backend = hamiltonian.backend

    hamiltonian = sum(Z(i) * Z(i + 1) for i in range(circ.nqubits - 1))
    hamiltonian += Z(0) * Z(circ.nqubits - 1)
    hamiltonian = hamiltonians.SymbolicHamiltonian(hamiltonian)
    coefficients = [1, 1, delta]
    mgates = ["X", "Y", "Z"]
    expectation_value = 0
    for i, mgate in enumerate(mgates):
        circ1 = circ.copy(deep=True)
        if mgate != "Z":
            circ1.add(gates.M(*range(circ1.nqubits), basis=getattr(gates, mgate)))
        else:
            circ1.add(gates.M(*range(circ1.nqubits)))

        expval_contribution = exec_backend.execute_circuit(
            circuit=circ1, nshots=nshots
        ).expectation_from_samples(hamiltonian)
        expectation_value += coefficients[i] * expval_contribution
    return expectation_value


def _exp_with_tf(circuit, hamiltonian, nshots=None, delta=0.5):
    params = circuit.get_parameters()
    nparams = len(circuit.get_parameters())

    @tf.custom_gradient
    def _expectation(params):
        def grad(upstream):
            print("with psr")
            gradients = []
            for p in range(nparams):
                gradients.append(
                    upstream
                    * parameter_shift(
                        circuit=circuit,
                        hamiltonian=hamiltonian,
                        parameter_index=p,
                        nshots=nshots,
                        delta=delta,
                        exec_backend=NumpyBackend(),
                    )
                )
            return gradients

        if nshots is None:
            expectation_value = _exact(circuit, hamiltonian)
        else:
            expectation_value = _with_shots(circuit, hamiltonian, nshots, delta)
        return expectation_value, grad

    return _expectation(params)


def parameter_shift(
    hamiltonian,
    circuit,
    parameter_index,
    exec_backend,
    nshots=None,
    delta=0.5,
):
    """Parameter Shift Rule."""

    if parameter_index > len(circuit.get_parameters()):
        raise_error(ValueError, """This index is out of bounds.""")

    if not isinstance(hamiltonian, AbstractHamiltonian):
        raise_error(
            TypeError,
            "hamiltonian must be a qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian object",
        )

    gate = circuit.associate_gates_with_parameters()[parameter_index]
    generator_eigenval = gate.generator_eigenvalue()

    s = np.pi / (4 * generator_eigenval)

    original = np.asarray(circuit.get_parameters()).copy()
    shifted = original.copy()

    shifted[parameter_index] += s
    circuit.set_parameters(shifted)

    if nshots is None:
        forward = _exact(circ=circuit, hamiltonian=hamiltonian)
        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)
        backward = _exact(circ=circuit, hamiltonian=hamiltonian)

    else:
        forward = _with_shots(circuit, hamiltonian, nshots, delta, exec_backend)
        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)
        backward = _with_shots(circuit, hamiltonian, nshots, delta, exec_backend)

    circuit.set_parameters(original)
    return float(generator_eigenval * (forward - backward))
