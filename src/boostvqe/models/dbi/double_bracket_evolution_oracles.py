from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, reduce
from typing import Union

import hyperopt
import matplotlib.pyplot as plt
import numpy as np
from qibo import Circuit, gates, symbols
from qibo.config import raise_error
from qibo.hamiltonians import AbstractHamiltonian, SymbolicHamiltonian

# TODO: remove this global import
from boostvqe.compiling_XXZ import nqubit_XXZ_decomposition


class EvolutionOracleType(Enum):
    """Mode to represent evolution oracle exp{-i t H}."""

    numerical = auto()
    """Matrix exponentiation."""

    hamiltonian_simulation = auto()
    """Use SymbolicHamiltonian and the compiles the circuit."""


@dataclass
class EvolutionOracle:
    """Representation of an EvolutionOracle exp{-i t h }."""

    h: AbstractHamiltonian
    """Hamiltonian to be exponentiated."""
    evolution_oracle_type: EvolutionOracleType
    """Type of evolution oracle."""

    def __post_init__(self):
        if self.steps is None:
            self.steps = 1
        """Number of steps in Trotter-Suzuki discretization."""

    def __call__(self, t_duration: float):
        """Returns either the name or the circuit"""
        return self.circuit(t_duration=t_duration)

    def circuit(self, t_duration: float):
        """This function returns depending on `EvolutionOracleType` string, ndarray or `Circuit`.
        In the hamiltonian_simulation mode we evaluate an appropriate Trotter-Suzuki discretization up to `self.eps_trottersuzuki` threshold.
        """
        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)  # e^{- i t_duration H}
        else:
            dt = t_duration / self.steps
            return reduce(
                Circuit.__add__,
                [self.h.circuit(dt)] * self.steps,  # approx of e^{- i t_duration H}
            )

    def inverted(self, duration: float):
        if self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
            return self.circuit(duration).operator.invert()
        elif self.evolution_oracle_type is EvolutionOracleType.numerical:
            return np.linalg.inv(self.circuit(duration))


@dataclass
class FrameShiftedEvolutionOracle(EvolutionOracle):
    """EvolutionOracle to perform FrameShift.

    The new EvolutionOracle will be the following V base_evolution_oracle Vdag.
    Where V is `before circuit` and Vdag is `after circuit.
    """

    # before_circuit: Union[Circuit, np.ndarray]
    # after_circuit: Union[Circuit, np.ndarray]
    circuit_frame: Union[Circuit, np.ndarray]
    base_evolution_oracle: EvolutionOracle

    @classmethod
    def from_evolution_oracle(
        cls,
        base_evolution_oracle: EvolutionOracle,
        circuit_frame,
    ):
        """Create instance using only new attributes of FreameShiftedEvolutionOracle."""
        return cls(
            base_evolution_oracle=base_evolution_oracle,
            circuit_frame=circuit_frame,
            h=base_evolution_oracle.h,
            evolution_oracle_type=base_evolution_oracle.evolution_oracle_type,
        )

    @property
    def nqubits(self):
        return self.circuit_frame.nqubits

    def circuit(self, duration: float = None):
        """Compute corresponding circuit."""

        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return (
                self.inverse_circuit
                @ self.base_evolution_oracle(duration)
                @ self.circuit_frame
            )
        elif self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
            return (
                self.circuit_frame
                + self.base_evolution_oracle.circuit(duration)
                + self.inverse_circuit
            )
        else:
            raise_error(
                ValueError,
                f"You are using an EvolutionOracle type which is not yet supported.",
            )

    @cached_property
    def inverse_circuit(self):
        if self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
            return self.circuit_frame.invert()
        elif self.evolution_oracle_type is EvolutionOracleType.numerical:
            return np.linalg.inv(self.circuit_frame)

    def get_composed_circuit(self):
        """Collect all frame shift in circuits."""
        c = Circuit(nqubits=self.nqubits)
        fseo = self
        while isinstance(fseo, FrameShiftedEvolutionOracle):
            if (
                self.base_evolution_oracle.evolution_oracle_type
                is EvolutionOracleType.numerical
            ):
                c = c @ fseo.circuit_frame
            elif (
                self.base_evolution_oracle.evolution_oracle_type
                is EvolutionOracleType.hamiltonian_simulation
            ):
                c = c + fseo.circuit_frame
            fseo = fseo.base_evolution_oracle
        return c


@dataclass
class MagneticFieldEvolutionOracle(EvolutionOracle):
    """Evolution oracle with MagneticField."""

    _params: Union[list, np.ndarray]

    @property
    def params(self):
        if isinstance(self._params, list):
            return self._params
        return self._params.tolist()

    @params.setter
    def params(self, params):
        self._params = params

    @classmethod
    def load(
        cls,
        params: list,
        evolution_oracle_type: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
    ):
        nqubits = len(params)
        hamiltonian = SymbolicHamiltonian(
            sum([bi * symbols.Z(j) for j, bi in zip(range(nqubits), params)])
        )
        return cls(
            h=hamiltonian, evolution_oracle_type=evolution_oracle_type, _params=params
        )

    def circuit(self, t):
        """
        Constructs an Magnetic Field model circuit for n qubits, given by:
        .. math::
            H(B) = \\sum_{i=0}^{N-1} B_i Z_i


        Args:
        nqubits (int): Number of qubits.
        t (float): Total evolution time.

        Returns:
        Circuit: The final multi-layer circuit.

        """
        nqubits = len(self.params)
        circuit = Circuit(nqubits=nqubits)
        # TODO: ask Marek
        circuit.add(
            gates.RZ(q_i, 2 * t * b) for q_i, b in zip(range(nqubits), self.params)
        )
        return circuit


@dataclass
class IsingNNEvolutionOracle(EvolutionOracle):
    _params: Union[list, np.ndarray]

    @property
    def params(self):
        if isinstance(self._params, list):
            return self._params
        return self._params.tolist()

    @params.setter
    def params(self, params):
        self._params = params

    @classmethod
    def load(
        cls,
        params: list,
        evolution_oracle_type: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
    ):
        nqubits = len(params) // 2
        b = params[:nqubits]
        j = params[nqubits:]
        hamiltonian = SymbolicHamiltonian(
            sum(
                [bi * symbols.Z(l) for l, bi in zip(range(nqubits), b)]
                + [
                    j[i] * symbols.Z(i) * symbols.Z((i + 1) % nqubits)
                    for i in range(nqubits)
                ]
            )
        )
        return cls(
            h=hamiltonian, evolution_oracle_type=evolution_oracle_type, _params=params
        )

    def circuit(self, t):
        """
        Constructs an XXZ model circuit for n qubits, given by:
        .. math::
            H(B,J) = \\sum_{i=0}^{N-1} \\left( B_i Z_i+ J_i Z_i Z_{i+1} \\right)


        Args:
        nqubits (int): Number of qubits.
        t (float): Total evolution time.

        Returns:
        Circuit: The final multi-layer circuit.

        """
        nqubits = len(self.params) // 2
        circuit = Circuit(nqubits=nqubits)
        # Create lists of even and odd qubit indices
        list_q_i = [num for num in range(nqubits)]
        list_q_ip1 = [num + 1 for num in range(nqubits - 1)]
        list_q_ip1.append(0)

        for q_i, q_ip1, j in zip(list_q_i, list_q_ip1, self.params[nqubits:]):
            circuit.add(gates.CNOT(q_i, q_ip1))
            circuit.add(gates.RZ(q_ip1, 2 * t * j))
            circuit.add(gates.CNOT(q_i, q_ip1))
        circuit.add(
            gates.RZ(q_i, 2 * t * b)
            for q_i, b in zip(range(nqubits), self.params[:nqubits])
        )
        return circuit


@dataclass
class XXZ_EvolutionOracle(EvolutionOracle):
    steps: int = None
    order: int = None
    delta: float = 0.5

    @classmethod
    def from_nqubits(cls, nqubits, delta, **kwargs):
        hamiltonian = SymbolicHamiltonian(
            sum(
                [
                    symbols.X(j) * symbols.X(j + 1)
                    + symbols.Y(j) * symbols.Y(j + 1)
                    + delta * symbols.Z(j) * symbols.Z(j + 1)
                    for j in range(nqubits - 1)
                ]
                + [
                    symbols.X(nqubits - 1) * symbols.X(0)
                    + symbols.Y(nqubits - 1) * symbols.Y(0)
                    + delta * symbols.Z(nqubits - 1) * symbols.Z(0)
                ]
            ),
            nqubits=nqubits,
        )
        return cls(
            h=hamiltonian,
            evolution_oracle_type=EvolutionOracleType.hamiltonian_simulation,
            **kwargs,
        )

    def circuit(self, t_duration, steps=None, order=None):
        if steps is None:
            steps = self.steps
        if order is None:
            order = self.order
            # this is doing correctly e^{- i t H_XXZ}
        return nqubit_XXZ_decomposition(
            nqubits=self.h.nqubits,
            t=t_duration,
            delta=self.delta,
            steps=steps,
            order=order,
        )


@dataclass
class tfim_EvolutionOracle(EvolutionOracle):
    steps: int = None
    B_a: float = None

    def circuit(self, a, t_duration, steps=None, order=None):
        if steps is None:
            steps = self.steps

        circuit = Circuit(self.h.nqubits)  # Initialize the circuit with the number of qubits

        # Add CNOT(a, a+1)
        circuit.add(gates.CNOT(a, a + 1))

        # Time evolution under the transverse field Ising model Hamiltonian
        # exp(-i t (X(a) + B_a * Z(a)))
        dt = t_duration / steps  # Divide the time duration for Trotterization if needed

        for _ in range(steps):
            # Apply time evolution for X(a) + B_a * Z(a)
            circuit += self._time_evolution_step(a, dt)

        # Add second CNOT(a, a+1)
        circuit.add(gates.CNOT(a, a + 1))

        return circuit

    def _time_evolution_step(self, a: int, dt: float, B_a: float):
        """Apply a single Trotter step of the time evolution operator exp(-i dt (X(a) + B_a Z(a)))."""
        step_circuit = Circuit(self.h.nqubits)

        # Time evolution for X(a)
        step_circuit.add(gates.RX(a, theta=-2 * dt))  # Apply exp(-i dt X(a))

        # Time evolution for Z(a)
        step_circuit.add(gates.RZ(a, theta=-2 * dt * B_a))  # Apply exp(-i dt B_a Z(a))

        return step_circuit