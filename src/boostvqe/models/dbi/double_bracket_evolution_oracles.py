from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce

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
    hamiltonian: AbstractHamiltonian
    evolution_oracle_type: EvolutionOracleType

    # self.mode_find_number_of_trottersuzuki_steps = True
    # self.eps_trottersuzuki = 0.0001
    # self.please_be_verbose = False
    # self.please_use_prescribed_nmb_ts_steps = False

    def __call__(self, t_duration: float):
        """Returns either the name or the circuit"""
        return self.circuit(t_duration=t_duration)

    def circuit(self, t_duration):
        raise (NotImplementedError)

    # def eval_unitary(self, t_duration):
    #     """This wraps around `circuit` and always returns a unitary"""
    #     if self.evolution_oracle_type is EvolutionOracleType.numerical:
    #         return self.circuit(t_duration)
    #     elif self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
    #         return self.circuit(t_duration).unitary()

    def circuit(self, t_duration: float = None):
        """This function returns depending on `EvolutionOracleType` string, ndarray or `Circuit`.
        In the hamiltonian_simulation mode we evaluate an appropriate Trotter-Suzuki discretization up to `self.eps_trottersuzuki` threshold.
        """
        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        elif self.evolution_oracle_type is EvolutionOracleType.hamiltonian_simulation:
            return self.discretized_evolution_circuit_binary_search(
                t_duration, eps=0.0001
            )
            # else:
            #     dt = t_duration / self.please_use_prescribed_nmb_ts_steps
            #     return reduce(
            #         Circuit.__add__,
            #         [deepcopy(self.h).circuit(dt)]
            #         * self.please_use_prescribed_nmb_ts_steps,
            #     )

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        nmb_trottersuzuki_steps = 1  # this is the smallest size
        nmb_trottersuzki_steps_right = 800  # this is the largest size for binary search
        if eps is None:
            eps = self.eps_trottersuzuki
        target_unitary = self.hamiltonian.exp(t_duration)

        def check_accuracy(n_steps):
            proposed_circuit_unitary = np.linalg.matrix_power(
                deepcopy(self.hamiltonian).circuit(t_duration / n_steps).unitary(),
                n_steps,
            )
            norm_difference = np.linalg.norm(target_unitary - proposed_circuit_unitary)
            return norm_difference < eps

        nmb_trottersuzuki_steps_used = nmb_trottersuzki_steps_right
        while nmb_trottersuzuki_steps <= nmb_trottersuzki_steps_right:
            mid = (
                nmb_trottersuzuki_steps
                + (nmb_trottersuzki_steps_right - nmb_trottersuzuki_steps) // 2
            )
            if check_accuracy(mid):
                nmb_trottersuzuki_steps_used = mid
                nmb_trottersuzki_steps_right = mid - 1
            else:
                nmb_trottersuzuki_steps = mid + 1
        nmb_trottersuzuki_steps = nmb_trottersuzuki_steps_used

        circuit_1_step = deepcopy(
            self.hamiltonian.circuit(t_duration / nmb_trottersuzuki_steps)
        )
        combined_circuit = reduce(
            Circuit.__add__, [circuit_1_step] * nmb_trottersuzuki_steps
        )
        assert (
            np.linalg.norm(combined_circuit.unitary() - target_unitary) < eps
        ), f"{np.linalg.norm(combined_circuit.unitary() - target_unitary)},{eps}, {nmb_trottersuzuki_steps}"
        return combined_circuit


@dataclass
class FrameShiftedEvolutionOracle(EvolutionOracle):
    before_circuit: str
    after_circuit: str
    base_evolution_oracle: EvolutionOracle

    @classmethod
    def from_evolution_oracle(
        cls, base_evolution_oracle: EvolutionOracle, before_circuit, after_circuit
    ):
        return cls(
            base_evolution_oracle=base_evolution_oracle,
            before_circuit=before_circuit,
            after_circuit=after_circuit,
            hamiltonian=base_evolution_oracle.hamiltonian,
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
    b: list

    @property
    def params(self):
        if isinstance(self.b, list):
            return self.b
        return self.b.tolist()

    @classmethod
    def from_b(
        cls,
        b: list,
        evolution_oracle_type: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
    ):
        nqubits = len(b)
        hamiltonian = SymbolicHamiltonian(
            sum([bi * symbols.Z(j) for j, bi in zip(range(nqubits), b)])
        )
        return cls(
            hamiltonian=hamiltonian, evolution_oracle_type=evolution_oracle_type, b=b
        )

    # def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
    #     if self.evolution_oracle_type is EvolutionOracleType.numerical:
    #         return self.h.exp(t_duration)

    #     if self.please_assess_how_many_steps_to_use:
    #         return super().discretized_evolution_circuit_binary_search(
    #             t_duration, eps=eps
    #         )
    #     else:
    #         return self.h.circuit(t_duration)


class IsingNNEvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        b_list,
        j_list,
        name="H_ClassicalIsing(B,J)",
        evolution_oracle_type: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
    ):
        """
        Constructs the evolution oracle for the classical Ising model
        .. math::
            H = \\sum_{i=0}^{N-1} \\left( B_i Z_i+ J_i Z_i Z_{i+1} \\right)
        """

        self.nqubits = len(b_list)
        d = SymbolicHamiltonian(
            sum(
                [b * symbols.Z(j) for j, b in zip(range(self.nqubits), b_list)]
                + [
                    j_list[j] * symbols.Z(j) * symbols.Z((j + 1) % self.nqubits)
                    for j in range(self.nqubits)
                ]
            )
        )
        super().__init__(d, name, evolution_oracle_type)
        self.b_list = b_list
        self.j_list = j_list
        self.please_assess_how_many_steps_to_use = (
            False  # otherwise methods which cast to dense will be used
        )
        self.please_use_coarse_compiling = False

    @property
    def params(self):
        return self.b_list.tolist() + self.j_list.tolist()

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        if self.evolution_oracle_type is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)

        if self.please_assess_how_many_steps_to_use:
            return super().discretized_evolution_circuit_binary_search(
                t_duration, eps=eps
            )
        else:
            return self.h.circuit(t_duration)

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
        if self.please_use_coarse_compiling:
            return super().circuit(t)
        else:
            circuit = Circuit(nqubits=self.nqubits)
            # Create lists of even and odd qubit indices
            list_q_i = [num for num in range(self.nqubits)]
            list_q_ip1 = [num + 1 for num in range(self.nqubits - 1)]
            list_q_ip1.append(0)

            for q_i, q_ip1, j in zip(list_q_i, list_q_ip1, self.j_list):
                circuit.add(gates.CNOT(q_i, q_ip1))
                circuit.add(gates.RZ(q_ip1, 2 * t * j))
                circuit.add(gates.CNOT(q_i, q_ip1))
            circuit.add(
                gates.RZ(q_i, 2 * t * b)
                for q_i, b in zip(range(self.nqubits), self.b_list)
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
            hamiltonian=hamiltonian,
            evolution_oracle_type=EvolutionOracleType.hamiltonian_simulation,
            **kwargs,
        )

    # def __init__(
    #     self,
    #     nqubits,
    #     evolution_oracle_type: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
    #     steps=None,
    #     order=None,
    #     delta=0.5,
    # ):
    #     super().__init__(
    #         XXZ_EvolutionOracle.xxz_symbolic(nqubits, delta=delta),
    #         evolution_oracle_type,
    #     )

    #     if steps is None:
    #         self.steps = 1
    #     else:
    #         self.steps = steps
    #     if order is None:
    #         self.order = 1
    #     else:
    #         self.order = order
    #     self.nqubits = nqubits
    #     self.delta = delta
    #     self.please_assess_how_many_steps_to_use = False

    # def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
    #     if self.please_assess_how_many_steps_to_use:
    #         return super().discretized_evolution_circuit_binary_search(
    #             t_duration, eps=eps
    #         )
    #     else:
    #         return self.h.circuit(t_duration)

    def circuit(self, t_duration, steps=None, order=None):
        if steps is None:
            steps = self.steps
        if order is None:
            order = self.order
        return nqubit_XXZ_decomposition(
            nqubits=self.hamiltonian.nqubits,
            t=t_duration,
            delta=self.delta,
            steps=steps,
            order=order,
        )

    # @staticmethod
    # def xxz_symbolic(nqubits, delta=0.5):
    #     return SymbolicHamiltonian(
    #         sum(
    #             [
    #                 symbols.X(j) * symbols.X(j + 1)
    #                 + symbols.Y(j) * symbols.Y(j + 1)
    #                 + delta * symbols.Z(j) * symbols.Z(j + 1)
    #                 for j in range(nqubits - 1)
    #             ]
    #             + [
    #                 symbols.X(nqubits - 1) * symbols.X(0)
    #                 + symbols.Y(nqubits - 1) * symbols.Y(0)
    #                 + delta * symbols.Z(nqubits - 1) * symbols.Z(0)
    #             ]
    #         ),
    #         nqubits=nqubits,
    #     )
