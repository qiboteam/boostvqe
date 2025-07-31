from dataclasses import dataclass
from enum import Enum, auto

import hyperopt
import numpy as np
from qibo import *
from qibo import gates, symbols
from qibo.config import raise_error
from qibo.symbols import X, Y, Z
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian

from boostvqe.models.dbqa import *
from boostvqe.models.dbqa.double_bracket_iteration import *
from boostvqe.models.dbqa.evolution_oracles_CZ_gates import *


class DoubleBracketRotationApproximationType(Enum):
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

def group_commutator(
        step_duration: float,
        d: SymbolicHamiltonian,
        h: SymbolicHamiltonian,
        mode_dbr: DoubleBracketRotationApproximationType = None,
        ):
        """This will return the group commutator circuit for the given step duration"""
        
        s_step = np.sqrt(step_duration)

        if mode_dbr is DoubleBracketRotationApproximationType.group_commutator:
            query_list_forward = [
                deepcopy(d).circuit(s_step),
                deepcopy(h).circuit(s_step),
                deepcopy(d).circuit(-s_step),
                deepcopy(h).circuit(-s_step),
            ]
            
        elif mode_dbr is DoubleBracketRotationApproximationType.group_commutator_reordered:
            query_list_forward = [
                deepcopy(h).circuit(s_step),
                deepcopy(d).circuit(-s_step),
                deepcopy(h).circuit(-s_step),
                deepcopy(d).circuit(s_step),
            ]
        elif mode_dbr is DoubleBracketRotationApproximationType.group_commutator_reduced:
            query_list_forward = [
                deepcopy(h).circuit(s_step),
                deepcopy(d).circuit(-s_step),
                deepcopy(h).circuit(-s_step),
            ]
        elif mode_dbr is DoubleBracketRotationApproximationType.group_commutator_third_order:
            query_list_forward = [
                deepcopy(d).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(h).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(d).circuit(s_step),
                deepcopy(h).circuit(s_step * (np.sqrt(5) + 1) / 2),
                deepcopy(d).circuit(-s_step * (3 - np.sqrt(5)) / 2),
                deepcopy(h).circuit(-s_step),
            ]
        elif mode_dbr is DoubleBracketRotationApproximationType.group_commutator_third_order_reduced:
            query_list_forward = [
                deepcopy(h).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(d).circuit(s_step),
                deepcopy(h).circuit(s_step * (np.sqrt(5) + 1) / 2),
                deepcopy(d).circuit(-s_step * (3 - np.sqrt(5)) / 2),
                deepcopy(h).circuit(-s_step),
            ]
        elif mode_dbr is DoubleBracketRotationApproximationType.group_commutator_mix_twice:
            s_step = step_duration / 2
            c1 = group_commutator(
                s_step, h, d, mode_dbr=DoubleBracketRotationApproximationType.group_commutator
            )
            c2 = group_commutator(
                s_step,
                h,
                d,
                mode_dbr=DoubleBracketRotationApproximationType.group_commutator_reduced,
            )
            return c2 + c1
        elif (
            mode_dbr is 
            DoubleBracketRotationApproximationType.group_commutator_reduced_twice
            or           
            mode_dbr is 
            DoubleBracketRotationApproximationType.group_commutator_third_order_reduced_twice
        ):
            s_step = step_duration / 2
            c1 = group_commutator(
                s_step,
                h,
                d,mode_dbr
  ) 
            return  c1 + c1 
        else:
            raise_error(
                ValueError,
                "You are using a DoubleBracketRotationApproximationType which is not yet supported.",)

        if isinstance(h, SymbolicHamiltonian):
            return reduce(Circuit.__add__, query_list_forward)#[::-1])
        elif isinstance(h, Hamiltonian):
            return reduce(np.ndarray.__matmul__, query_list_forward)
        else:
            raise_error(ValueError, "Your EvolutionOracleType is not recognized")

@dataclass
class GroupCommutatorIteration:
    """
    Simulates circuits of double-bracket quantum algorithms."""

    h: SymbolicHamiltonian
    """Input Hamiltonian, principally expecting SymbolicHamiltonian."""
    preparation_circuit: Circuit = None
    double_bracket_rotation_approximation_type: DoubleBracketRotationApproximationType = None

    def __post_init__(self):
        # If the input is an integer, we assume it is the number of qubits
        if self.h is None:
            L = 3       
        if isinstance(self.h, int):            
            L=self.h
            self.h = SymbolicHamiltonian(
                sum([X(i)*X(i+1)+ Y(i)*Y(i+1) + 0.5* Z(i)*Z(i+1) 
                     for i in range(L-1)]))
            
        # If there is no preparation circuit, we create an empty one
        if self.preparation_circuit is None:
            self.preparation_circuit = Circuit(self.h.nqubits)

        # We set the circuit function to be conjugated by the preparation circuit 
        # and the original circuit function is stored as original_circuit
        self.h.original_circuit = deepcopy(self.h.circuit)
        self.h.circuit = lambda t_duration: (
            deepcopy(self.preparation_circuit)
            + self.h.original_circuit(t_duration)
            + self.preparation_circuit.invert()
            )
        
        # If the approximation type is not set, we default to the reduced group commutator
        if self.double_bracket_rotation_approximation_type is None:
            double_bracket_rotation_type = DoubleBracketRotationApproximationType.group_commutator_reduced

    @property
    def nqubits(self):
        return self.h.nqubits

    def __call__(
        self,
        step_duration: float,
        d: SymbolicHamiltonian = None,
        mode_dbr: DoubleBracketRotationApproximationType = None,
    ):
        if mode_dbr is None:
            mode_dbr = self.double_bracket_rotation_approximation_type
        if d is None:
            d = SymbolicHamiltonian(sum([Z(i) for i in range(self.nqubits)]))
        """This will run the appropriate group commutator step"""

        self.preparation_circuit = deepcopy(
            group_commutator(step_duration**2, d, self.h, mode_dbr)+self.preparation_circuit
        )

    
    def loss(self, step_duration: float = None, eo_d=None, mode_dbr=None):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step_duration (float, optional): iteration step. If None, no extra step is added.
            eo_d (EvolutionOracle, optional): diagonal operator. Defaults to self.input_hamiltonian_evolution_oracle if not provided.
            mode_dbr (DoubleBracketRotationApproximationType, optional): DBR mode. Defaults to self.double_bracket_rotation_type if not provided.
        """
        if mode_dbr is None:
            mode_dbr = self.double_bracket_rotation_approximation_type
        if step_duration is not None: 
            circ = (
             group_commutator(step_duration**2, eo_d, self.h, mode_dbr) 
            + self.preparation_circuit)   
        else:
            circ = self.preparation_circuit 
        return self.h.expectation(circ().state())

    def choose_step(
        self,
        d,
        step_grid=None,
        mode_dbr=None,
    ):
        if isinstance(step_grid, int):
            step_grid = np.linspace(0.01, 1.0, step_grid)
        elif step_grid is None:
            step_grid = np.linspace(0.0001, 0.1, 10)
        losses = [ self.loss(s, d, mode_dbr) for s in step_grid]
        return step_grid[np.argmin(losses)], np.min(losses), losses
    
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
