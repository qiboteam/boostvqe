from dataclasses import dataclass
from enum import Enum, auto

import hyperopt
import numpy as np
from qibo import *
from qibo import gates, symbols
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian

from boostvqe.models.dbi import *
from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.double_bracket_evolution_oracles import *


class DoubleBracketRotationType(Enum):
    # The dbr types below need a diagonal input matrix $\hat D_k$   :

    group_commutator = auto()
    """Use group commutator approximation"""
    group_commutator_reordered = auto()
    """Use group commutator approximation with reordering of the operators"""
    group_commutator_reduced = auto()
    """Use group commutator approximation with a reduction using symmetry"""
    group_commutator_third_order = auto()
    """Higher order approximation    """
    group_commutator_third_order_reduced = auto()
    """Higher order approximation    """
    group_commutator_mix_twice = auto()
    group_commutator_reduced_twice = auto()
    group_commutator_third_order_reduced_twice = auto()


@dataclass
class GroupCommutatorIterationWithEvolutionOracles(DoubleBracketIteration):
    """
    Class which will be later merged into the @super somehow"""

    input_hamiltonian_evolution_oracle: EvolutionOracle
    double_bracket_rotation_type: DoubleBracketRotationType

    def __post_init__(self):
        self.iterated_hamiltonian_evolution_oracle = deepcopy(
            self.input_hamiltonian_evolution_oracle
        )

    @property
    def nqubits(self):
        return self.h.nqubits

    @property
    def h(self):
        return self.input_hamiltonian_evolution_oracle.h

    def __call__(
        self,
        step_duration: float,
        diagonal_association: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        # This will run the appropriate group commutator step
        rs_circ = self.recursion_step_circuit(
            step_duration,
            diagonal_association,
            mode_dbr,
        )
        if (
            self.input_hamiltonian_evolution_oracle.evolution_oracle_type
            is EvolutionOracleType.numerical
        ):
            rs_circ_inv = np.linalg.inv(rs_circ)
        else:
            rs_circ_inv = rs_circ.invert()

        self.iterated_hamiltonian_evolution_oracle = (
            FrameShiftedEvolutionOracle.from_evolution_oracle(
                deepcopy(self.iterated_hamiltonian_evolution_oracle),
                rs_circ_inv,
                rs_circ,
            )
        )

    def group_commutator(
        self,
        step_duration: float,
        eo_1: EvolutionOracle,
        eo_2: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        s_step = np.sqrt(step_duration)
        if eo_2 is None:
            eo_2 = self.iterated_hamiltonian_evolution_oracle

        if mode_dbr is None:
            gc_type = self.double_bracket_rotation_type
        else:
            gc_type = mode_dbr

        if gc_type is DoubleBracketRotationType.group_commutator:
            query_list_forward = [
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(-s_step),
                deepcopy(eo_1).circuit(-s_step),
            ]
            query_list_backward = [
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(-s_step),
                deepcopy(eo_2).circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reordered:
            query_list_forward = [
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(-s_step),
                deepcopy(eo_1).circuit(-s_step),
                deepcopy(eo_2).circuit(s_step),
            ]
            query_list_backward = [
                deepcopy(eo_2).circuit(-s_step),
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reduced:
            query_list_forward = [
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(-s_step),
                deepcopy(eo_1).circuit(-s_step),
            ]
            query_list_backward = [
                deepcopy(eo_1).circuit(s_step),
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_third_order:
            query_list_forward = [
                deepcopy(eo_2).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(eo_1).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(s_step * (np.sqrt(5) + 1) / 2),
                deepcopy(eo_2).circuit(-s_step * (3 - np.sqrt(5)) / 2),
                deepcopy(eo_1).circuit(-s_step),
            ]
            query_list_backward = [Circuit.invert(c) for c in query_list_forward[::-1]]
        elif gc_type is DoubleBracketRotationType.group_commutator_third_order_reduced:
            query_list_forward = [
                deepcopy(eo_1).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(eo_2).circuit(s_step),
                deepcopy(eo_1).circuit(s_step * (np.sqrt(5) + 1) / 2),
                deepcopy(eo_2).circuit(-s_step * (3 - np.sqrt(5)) / 2),
                deepcopy(eo_1).circuit(-s_step),
            ]
            query_list_backward = [Circuit.invert(c) for c in query_list_forward[::-1]]
        elif gc_type is DoubleBracketRotationType.group_commutator_mix_twice:
            s_step = step_duration / 2
            c1 = self.group_commutator(
                s_step, eo_1, eo_2, mode_dbr=DoubleBracketRotationType.group_commutator
            )["forwards"]
            c2 = self.group_commutator(
                s_step,
                eo_1,
                eo_2,
                mode_dbr=DoubleBracketRotationType.group_commutator_reduced,
            )["forwards"]
            return {"forwards": c2 + c1, "backwards": (c2 + c1).invert()}
        elif gc_type is DoubleBracketRotationType.group_commutator_reduced_twice:
            s_step = step_duration / 2
            c1 = self.group_commutator(
                s_step,
                eo_1,
                eo_2,
                mode_dbr=DoubleBracketRotationType.group_commutator_reduced,
            )["forwards"]
            return {"forwards": c1 + c1, "backwards": (c1 + c1).invert()}
        elif (
            gc_type
            is DoubleBracketRotationType.group_commutator_third_order_reduced_twice
        ):
            s_step = step_duration / 2
            c1 = self.group_commutator(
                s_step,
                eo_1,
                eo_2,
                mode_dbr=DoubleBracketRotationType.group_commutator_third_order_reduced,
            )["forwards"]
            return {"forwards": c1 + c1, "backwards": (c1 + c1).invert()}
        else:
            raise_error(
                ValueError,
                "You are in the group commutator query list but your dbr mode is not recognized",
            )

        eo_mode = eo_1.evolution_oracle_type
        if eo_mode is EvolutionOracleType.hamiltonian_simulation:
            return {
                "forwards": reduce(Circuit.__add__, query_list_forward[::-1]),
                "backwards": reduce(Circuit.__add__, query_list_backward[::-1]),
            }
        elif eo_mode is EvolutionOracleType.numerical:
            return {
                "forwards": reduce(np.ndarray.__matmul__, query_list_forward),
                "backwards": reduce(np.ndarray.__matmul__, query_list_backward),
            }
        else:
            raise_error(ValueError, "Your EvolutionOracleType is not recognized")

    def loss(self, step_duration: float, eo_d, mode_dbr):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step_duration (float): iteration step.
            d (np.array): diagonal operator, use canonical by default.
            look_ahead (int): number of iteration steps to compute the loss function;
        """

        circ = self.get_composed_circuit()
        if step_duration is not None:
            circ = self.recursion_step_circuit(step_duration, eo_d, mode_dbr) + circ
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
            return self.iterated_hamiltonian_evolution_oracle.get_composed_circuit()
        else:
            return (
                self.recursion_step_circuit(step_duration, eo_d)
                + self.iterated_hamiltonian_evolution_oracle.get_composed_circuit()
            )

    def recursion_step_circuit(self, step_duration, eo_d, mode_dbr=None):
        # Set rotation type
        if mode_dbr is None:
            mode_dbr = self.double_bracket_rotation_type
        return self.group_commutator(
            step_duration=step_duration, eo_1=eo_d, mode_dbr=mode_dbr
        )["forwards"]

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

    def print_gate_count_report(self):
        counts = self.get_gate_count_dict()
        print(
            f"The boosting circuit used {counts['nmb_cnot']} CNOT gates coming from compiled XXZ evolution and {counts['nmb_cz']} CZ gates from VQE.\n\
For {self.nqubits} qubits this gives n_CNOT/n_qubits = {counts['nmb_cnot_relative']} and n_CZ/n_qubits = {counts['nmb_cz_relative']}"
        )
