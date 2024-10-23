from dataclasses import dataclass
from enum import Enum, auto

import hyperopt
import numpy as np
from qibo import *
from qibo import gates, symbols
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian

from boostvqe.models.dbi.double_bracket_evolution_oracles import *


# TODO: change name
class DoubleBracketRotationType(Enum):
    # The dbr types below need a diagonal input matrix $\hat D_k$   :

    group_commutator = auto()
    """First order group commutator approximation."""
    group_commutator_reordered = auto()
    """First order group commutator approximation with reordering of the operators."""
    group_commutator_reduced = auto()
    """First group commutator approximation with a reduction using symmetry."""
    group_commutator_third_order = auto()
    """Third order group commutator approximation."""
    group_commutator_third_order_reduced = auto()
    """Third order group commutator approximation with reordering of the operators."""
    group_commutator_reduced_twice = auto()
    """First order group commutator applied twice."""
    group_commutator_third_order_reduced_twice = auto()
    """Third order group commutator applied twice."""


# @dataclass
class GroupCommutatorIterationWithEvolutionOracles:
    """DBI evolution under group commutator unfolded."""

    def __init__(self, oracle: EvolutionOracle, mode: DoubleBracketRotationType):
        self.oracle = oracle
        self.mode = mode
        self.initial_oracle = deepcopy(oracle)

    @property
    def nqubits(self):
        return self.h.nqubits

    @property
    def h0(self):
        """Starting Hamiltonian."""
        return self.initial_oracle.h

    @property
    def h(self):
        """Hamiltonian evolved."""
        return self.oracle.h

    def __call__(
        self,
        step_duration: float,
        d: EvolutionOracle = None,
        mode: DoubleBracketRotationType = None,
    ):
        """Evolution step of oracle using d and mode."""

        forward = self._forward(duration=step_duration, d=d, mode=mode)

        self.oracle = FrameShiftedEvolutionOracle.from_evolution_oracle(
            base_evolution_oracle=self.oracle,
            circuit_frame=forward,
        )

    def _operators(
        self, duration: float, d: EvolutionOracle, mode: DoubleBracketRotationType
    ):
        s_step = np.sqrt(duration)

        if mode is DoubleBracketRotationType.group_commutator:
            operators = [
                d.circuit(-s_step),
                self.oracle.circuit(-s_step),
                d.circuit(s_step),
                self.oracle.circuit(s_step),
            ]
        elif mode is DoubleBracketRotationType.group_commutator_reordered:
            operators = [
                self.oracle.circuit(s_step),
                d.circuit(-s_step),
                self.oracle.circuit(-s_step),
                d.circuit(s_step),
            ]
        elif mode is DoubleBracketRotationType.group_commutator_reduced:
            operators = [
                d.circuit(-s_step),
                self.oracle.circuit(s_step),
                d.circuit(s_step),
            ]
        elif mode is DoubleBracketRotationType.group_commutator_third_order:
            operators = [
                self.oracle.circuit(-s_step * (np.sqrt(5) - 1) / 2),
                d.circuit(-s_step * (np.sqrt(5) - 1) / 2),
                self.oracle.circuit(s_step),
                d.circuit(s_step * (np.sqrt(5) + 1) / 2),
                self.oracle.circuit(-s_step * (3 - np.sqrt(5)) / 2),
                d.oracle.circuit(-s_step),
            ]
        elif mode is DoubleBracketRotationType.group_commutator_third_order_reduced:
            operators = [
                d.circuit(-s_step * (np.sqrt(5) - 1) / 2),
                self.oracle.circuit(s_step),
                d.circuit(s_step * (np.sqrt(5) + 1) / 2),
                self.oracle.circuit(-s_step * (3 - np.sqrt(5)) / 2),
                d.circuit(-s_step),
            ]
        elif mode is DoubleBracketRotationType.group_commutator_reduced_twice:
            s_step = duration / 2
            # FIXME: this will do /2 and sqrt
            operators = 2 * self._execute(
                s_step, d, DoubleBracketRotationType.group_commutator_reduced
            )
        elif (
            mode is DoubleBracketRotationType.group_commutator_third_order_reduced_twice
        ):
            s_step = duration / 2
            operators = 2 * self._execute(
                s_step, d, DoubleBracketRotationType.group_commutator_third_order
            )

        return operators

    def _forward(
        self, duration: float, d: EvolutionOracle, mode: DoubleBracketRotationType
    ):
        assert self.oracle.evolution_oracle_type == d.evolution_oracle_type
        return self._contract(
            self._operators(duration, d, mode), d.evolution_oracle_type
        )

    @staticmethod
    def _contract(operators: list, mode: EvolutionOracleType.hamiltonian_simulation):
        if mode is EvolutionOracleType.hamiltonian_simulation:
            return reduce(Circuit.__add__, operators[::-1])
        elif mode is EvolutionOracleType.numerical:
            return reduce(np.ndarray.__matmul__, operators)

    def loss(
        self, step_duration: float, d: EvolutionOracle, mode: DoubleBracketRotationType
    ):
        circ = self._forward(step_duration, d, mode) + self.get_composed_circuit()
        return self.h.expectation(circ().state())

    def choose_step(
        self,
        d,
        step_grid=None,
        mode_dbr=None,
    ):
        losses = []
        for s in step_grid:
            losses.append(self.loss(s, d, mode_dbr))
        return step_grid[np.argmin(losses)], np.min(losses), losses

    def get_composed_circuit(self, step_duration=None, eo_d=None):
        """Get the ordered composition of all previous circuits regardless of previous mode_dbr
        settings."""
        if step_duration is None or eo_d is None:
            return self.oracle.get_composed_circuit()
        else:
            return (
                self._forward(step_duration, eo_d) + self.oracle.get_composed_circuit()
            )

    @staticmethod
    def count_gates(circuit, gate_type):
        t = 0
        for g in circuit.queue:
            if isinstance(g, gate_type):
                t = t + 1
        return t

    def count_CNOTs(self, circuit=None):
        if circuit is None:
            circuit = self.get_composed_circuit()
        return self.count_gates(circuit, gates.gates.CNOT)

    def count_CZs(self, circuit=None):
        if circuit is None:
            circuit = self.get_composed_circuit()
        return self.count_gates(circuit, gates.gates.CZ)

    def count_RBS(self, circuit=None):
        if circuit is None:
            circuit = self.get_composed_circuit()
        return self.count_gates(circuit, gates.gates.RBS)

    def get_gate_count_dict(self):
        return dict(
            nmb_cz=self.count_CZs() + 2 * self.count_RBS(),
            nmb_cnot=self.count_CNOTs(),
            nmb_cnot_relative=self.count_CZs() / self.nqubits,
            nmb_cz_relative=self.count_CNOTs() / self.nqubits,
        )
