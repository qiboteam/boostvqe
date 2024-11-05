import logging
import os
from enum import Enum

from qibo.hamiltonians.models import HamiltonianTerm, _multikron

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from qibo import gates, hamiltonians
from qibo.backends import NumpyBackend, matrices
from qibo.config import raise_error
from qibo.hamiltonians import AbstractHamiltonian, Hamiltonian, SymbolicHamiltonian
from qibo.hamiltonians.models import _build_spin_model
from qibo.symbols import Z
from qiboml.backends import TensorflowBackend

DEFAULT_DELTA = 0.5
"""Default `delta` value of XXZ Hamiltonian"""
DEFAULT_DELTAS = [0.0, 2.0]
TLFIM_h = [1.0, 2.0]
J1J2_h = [1.0, 0.2]


class Model(Enum):
    XXZ = lambda nqubits: hamiltonians.XXZ(
        nqubits=nqubits, delta=DEFAULT_DELTA, dense=False
    )
    XYZ = lambda nqubits: XYZ(nqubits=nqubits, deltas=[0.5, 0.5], dense=False)
    TFIM = lambda nqubits: hamiltonians.TFIM(nqubits=nqubits, h=nqubits, dense=False)
    TLFIM = lambda nqubits: TLFIM(nqubits=nqubits, h=TLFIM_h, dense=False)
    J1J2 = lambda nqubits: J1J2(nqubits=nqubits, h=J1J2_h, dense=False)


def TLFIM(nqubits, h=TLFIM_h, dense=True, backend=None):
    """Transverse and longitudinal field Ising model with periodic boundary conditions.

    .. math::
        H = - \\sum _{i=0}^N \\left ( Z_i Z_{i + 1} + h_0 X_i + h_1 Z_i \\right ).

    Args:
        nqubits (int): number of quantum bits.
        h (float): value of the transverse field.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")
    if dense:
        condition = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        ham = _build_spin_model(nqubits, matrices.Z, condition)
        for m, matrix in enumerate([matrices.X, matrices.Z]):
            if h[m] != 0:
                condition = lambda i, j: i == j % nqubits
                ham += h[m] * _build_spin_model(nqubits, matrix, condition)
        ham *= -1
        return Hamiltonian(nqubits, ham, backend=backend)

    matrix = -(
        _multikron([matrices.Z, matrices.Z])
        + h[0] * _multikron([matrices.X, matrices.I])
        + h[1] * _multikron([matrices.Z, matrices.I])
    )
    terms = [HamiltonianTerm(matrix, i, i + 1) for i in range(nqubits - 1)]
    terms.append(HamiltonianTerm(matrix, nqubits - 1, 0))
    ham = SymbolicHamiltonian(backend=backend)
    ham.terms = terms
    return ham


def J1J2(nqubits, h=J1J2_h, dense=True, backend=None):
    """Heisenberg J1-J2 model."""
    logging.info("Building the J1-J2 Heisenberg model.")
    if nqubits < 3:
        raise_error(ValueError, "Number of qubits must be larger than two.")
    if dense:
        condition_1 = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        hx = _build_spin_model(nqubits, matrices.X, condition_1)
        hy = _build_spin_model(nqubits, matrices.Y, condition_1)
        hz = _build_spin_model(nqubits, matrices.Z, condition_1)
        condition_2 = lambda i, j: i in {j % nqubits, (j + 2) % nqubits}
        hx2 = _build_spin_model(nqubits, matrices.X, condition_2)
        hy2 = _build_spin_model(nqubits, matrices.Y, condition_2)
        hz2 = _build_spin_model(nqubits, matrices.Z, condition_2)
        matrix = h[0] * (hx + hy + hz) + h[1] * (hx2 + hy2 + hz2)
        return Hamiltonian(nqubits, matrix, backend=backend)

    hx = _multikron([matrices.X, matrices.X])
    hy = _multikron([matrices.Y, matrices.Y])
    hz = _multikron([matrices.Z, matrices.Z])
    matrix_1 = h[0] * (hx + hy + hz)
    matrix_2 = h[1] * (hx + hy + hz)
    terms = [HamiltonianTerm(matrix_1, i, i + 1) for i in range(nqubits - 1)]
    terms.extend([HamiltonianTerm(matrix_2, i, i + 2) for i in range(nqubits - 2)])
    terms.append(HamiltonianTerm(matrix_1, nqubits - 1, 0))
    terms.append(HamiltonianTerm(matrix_2, nqubits - 2, 0))
    terms.append(HamiltonianTerm(matrix_2, nqubits - 1, 1))
    ham = SymbolicHamiltonian(backend=backend)
    ham.terms = terms
    return ham


def XYZ(nqubits, deltas=[0.5, 0.5], dense=True, backend=None):
    """XYZ model with periodic boundary conditions.

    .. math::
        H = \\sum _{i=0}^N \\left ( X_iX_{i + 1} + \\delta_0 Y_iY_{i + 1} + \\delta_1 Z_iZ_{i + 1} \\right ).

    Args:
        nqubits (int): number of quantum bits.
        deltas (list): coefficients for the Z component (default 0.5).
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.

    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")
    if dense:
        condition = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        hx = _build_spin_model(nqubits, matrices.X, condition)
        hy = _build_spin_model(nqubits, matrices.Y, condition)
        hz = _build_spin_model(nqubits, matrices.Z, condition)
        matrix = hx + deltas[0] * hy + deltas[1] * hz
        return Hamiltonian(nqubits, matrix, backend=backend)

    hx = _multikron([matrices.X, matrices.X])
    hy = _multikron([matrices.Y, matrices.Y])
    hz = _multikron([matrices.Z, matrices.Z])
    matrix = (hx + deltas[0] * hy) + deltas[1] * hz
    terms = [HamiltonianTerm(matrix, i, i + 1) for i in range(nqubits - 1)]
    terms.append(HamiltonianTerm(matrix, nqubits - 1, 0))
    ham = SymbolicHamiltonian(backend=backend)
    ham.terms = terms
    return ham


def vqe_loss(params, circuit, hamiltonian, nshots=None):
    """Evaluate the hamiltonian expectation values of the circuit final state."""
    circ = circuit.copy(deep=True)
    circ.set_parameters(params)
    if isinstance(hamiltonian.backend, TensorflowBackend) and nshots is not None:
        expectation_value = _exp_with_tf(
            circuit=circ, hamiltonian=hamiltonian, nshots=nshots
        )
    elif nshots is None:
        expectation_value = _exact(circ, hamiltonian)
    else:
        expectation_value = _with_shots(circ=circ, ham=hamiltonian, nshots=nshots)
    return expectation_value


def _exact(circ, hamiltonian):
    """Helper function to compute expectation value of Heisemberg hamiltonian."""
    expectation_value = hamiltonian.expectation(
        hamiltonian.backend.execute_circuit(circuit=circ).state()
    )
    return expectation_value


def _with_shots(circ, ham, nshots, exec_backend=None):
    """Helper function to compute XXZ expectation value from frequencies."""
    # we may prefer run this on a different backend (e.g. with TF and PSR)
    if exec_backend is None:
        exec_backend = ham.backend

    hamiltonian = sum(Z(i) * Z(i + 1) for i in range(circ.nqubits - 1))
    hamiltonian += Z(0) * Z(circ.nqubits - 1)
    hamiltonian = hamiltonians.SymbolicHamiltonian(hamiltonian)
    hamiltonian1 = sum(Z(i) for i in range(circ.nqubits))
    hamiltonian1 = hamiltonians.SymbolicHamiltonian(hamiltonian1)
    expectation_value = 0
    nqubits = circ.nqubits

    if np.array_equal(
        np.array(ham.matrix), np.array(Model.TFIM(nqubits=nqubits).matrix)
    ):
        # Evaluate the ZZ terms
        circ1 = circ.copy(deep=True)
        circ1.add(gates.M(*range(circ1.nqubits)))
        expval_contribution = exec_backend.execute_circuit(
            circuit=circ1, nshots=nshots
        ).expectation_from_samples(hamiltonian)
        expectation_value -= expval_contribution

        # Evaluate X terms
        circ1 = circ.copy(deep=True)
        circ1.add(gates.M(*range(circ1.nqubits), basis=gates.X))
        expval_contribution = exec_backend.execute_circuit(
            circuit=circ1, nshots=nshots
        ).expectation_from_samples(hamiltonian1)
        expectation_value -= nqubits * expval_contribution

    elif np.array_equal(
        np.array(ham.matrix), np.array(Model.TLFIM(nqubits=nqubits).matrix)
    ):
        # Evaluate the ZZ terms
        circ1 = circ.copy(deep=True)
        circ1.add(gates.M(*range(circ1.nqubits)))
        expval_contribution = exec_backend.execute_circuit(
            circuit=circ1, nshots=nshots
        ).expectation_from_samples(hamiltonian)
        expectation_value -= expval_contribution

        # Evaluate X and Z terms
        for i, gate in enumerate([gates.X, gates.Z]):
            circ1 = circ.copy(deep=True)
            if gate == gates.X:
                circ1.add(gates.M(*range(circ1.nqubits), basis=gate))
            else:
                circ1.add(gates.M(*range(circ1.nqubits)))
            expval_contribution = exec_backend.execute_circuit(
                circuit=circ1, nshots=nshots
            ).expectation_from_samples(hamiltonian1)
            expectation_value -= nqubits * expval_contribution

    else:
        if np.array_equal(
            np.array(ham.matrix), np.array(Model.XXZ(nqubits=nqubits).matrix)
        ):
            coefficients = [1, 1, DEFAULT_DELTA]
        elif np.array_equal(
            np.array(ham.matrix), np.array(Model.XYZ(nqubits=nqubits).matrix)
        ):
            coefficients = [1, *DEFAULT_DELTAS]
        else:
            raise NotImplemented("This option is not valid")
        mgates = ["X", "Y", "Z"]
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


def _exp_with_tf(circuit, hamiltonian, nshots=None):
    params = circuit.get_parameters()
    nparams = len(circuit.get_parameters())

    @tf.custom_gradient
    def _expectation(params):
        def grad(upstream):
            gradients = []
            for p in range(nparams):
                gradients.append(
                    upstream
                    * parameter_shift(
                        circuit=circuit,
                        hamiltonian=hamiltonian,
                        parameter_index=p,
                        nshots=nshots,
                        exec_backend=NumpyBackend(),
                    )
                )
            return gradients

        if nshots is None:
            expectation_value = _exact(circuit, hamiltonian)
        else:
            expectation_value = _with_shots(
                circ=circuit, ham=hamiltonian, nshots=nshots
            )
        return expectation_value, grad

    return _expectation(params)


def parameter_shift(
    hamiltonian,
    circuit,
    parameter_index,
    exec_backend,
    nshots=None,
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
        forward = _with_shots(circuit, hamiltonian, nshots, exec_backend)
        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)
        backward = _with_shots(circuit, hamiltonian, nshots, exec_backend)

    circuit.set_parameters(original)
    return float(generator_eigenval * (forward - backward))
