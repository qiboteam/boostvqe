from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
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
    numerical = auto()
    """If you will work with exp(is_k J_k) as a numerical matrix"""

    hamiltonian_simulation = auto()
    """If you will use SymbolicHamiltonian"""


@dataclass
class EvolutionOracle:
    h: AbstractHamiltonian
    evolution_oracle_type: EvolutionOracleType

    def __post_init__(self):
        self.steps = 1

    def __call__(self, t_duration: float):
        """Returns either the name or the circuit"""
        return self.circuit(t_duration=t_duration)

    def circuit(self, t_duration: float):
        """This function returns depending on `EvolutionOracleType` string, ndarray or `Circuit`.
        In the hamiltonian_simulation mode we evaluate an appropriate Trotter-Suzuki discretization up to `self.eps_trottersuzuki` threshold.
        """
        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        else:
            dt = t_duration / self.steps
            return reduce(
                Circuit.__add__,
                [deepcopy(self.h).circuit(dt)] * self.steps,
            )


@dataclass
class FrameShiftedEvolutionOracle(EvolutionOracle):
    before_circuit: str
    after_circuit: str
    base_evolution_oracle: EvolutionOracle

    @classmethod
    def from_evolution_oracle(
        cls,
        base_evolution_oracle: EvolutionOracle,
        before_circuit,
        after_circuit,
    ):
        return cls(
            base_evolution_oracle=base_evolution_oracle,
            before_circuit=before_circuit,
            after_circuit=after_circuit,
            h=base_evolution_oracle.h,
            evolution_oracle_type=base_evolution_oracle.evolution_oracle_type,
        )

    @property
    def nqubits(self):
        assert self.before_circuit.nqubits == self.after_circuit.nqubits
        return self.before_circuit.nqubits

    def circuit(self, t_duration: float = None):
        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return (
                self.before_circuit
                @ self.base_evolution_oracle(t_duration)
                @ self.after_circuit
            )
        elif self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
            return (
                self.after_circuit
                + self.base_evolution_oracle.circuit(t_duration)
                + self.before_circuit
            )
        else:
            raise_error(
                ValueError,
                f"You are using an EvolutionOracle type which is not yet supported.",
            )

    def get_composed_circuit(self):
        c = Circuit(nqubits=self.nqubits)
        fseo = self
        while isinstance(fseo, FrameShiftedEvolutionOracle):
            if (
                self.base_evolution_oracle.evolution_oracle_type
                is EvolutionOracleType.numerical
            ):
                c = c @ fseo.after_circuit
            elif (
                self.base_evolution_oracle.evolution_oracle_type
                is EvolutionOracleType.hamiltonian_simulation
            ):
                c = c + fseo.after_circuit
            fseo = fseo.base_evolution_oracle
        return c


@dataclass
class MagneticFieldEvolutionOracle(EvolutionOracle):
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

        #"""
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
        return nqubit_XXZ_decomposition(
            nqubits=self.h.nqubits,
            t=t_duration,
            delta=self.delta,
            steps=steps,
            order=order,
        )
