from enum import Enum, auto

import hyperopt
import numpy as np

from qibo import *
from qibo import symbols, gates
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from boostvqe.models.dbi import *
from boostvqe.models.dbi.double_bracket import *
from boostvqe.models.dbi.double_bracket_evolution_oracles import *


class DoubleBracketRotationType(Enum):
    # The dbr types below need a diagonal input matrix $\hat D_k$   :

    single_commutator = auto()
    """Use single commutator."""

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


class GroupCommutatorIterationWithEvolutionOracles(DoubleBracketIteration):
    """
    Class which will be later merged into the @super somehow"""

    def __init__(
        self,
        input_hamiltonian_evolution_oracle: EvolutionOracle,
        mode_double_bracket_rotation: DoubleBracketRotationType = DoubleBracketRotationType.group_commutator,
        h_ref = None
    ):
        if mode_double_bracket_rotation is DoubleBracketRotationType.single_commutator:
            mode_double_bracket_rotation_old = (
                DoubleBracketGeneratorType.single_commutator
            )
        else:
            mode_double_bracket_rotation_old = (
                DoubleBracketGeneratorType.group_commutator
            )
        super().__init__(
            input_hamiltonian_evolution_oracle.h, mode_double_bracket_rotation_old
        )
        if h_ref is not None:
            self.h_ref = h_ref  
        else:
            self.h_ref = deepcopy(input_hamiltonian_evolution_oracle.h)
        self.input_hamiltonian_evolution_oracle = input_hamiltonian_evolution_oracle
        
        self.mode_double_bracket_rotation = mode_double_bracket_rotation

        self.gci_unitary = []
        self.gci_unitary_dagger = []
        self.iterated_hamiltonian_evolution_oracle = deepcopy(
            self.input_hamiltonian_evolution_oracle
        )
        self.please_evaluate_matrices = False
        

    def __call__(
        self,
        step_duration: float,
        diagonal_association: EvolutionOracle,
        mode_dbr: DoubleBracketRotationType = None,
    ):

        # Set rotation type
        if mode_dbr is None:
            mode_dbr = self.mode_double_bracket_rotation

        if mode_dbr is DoubleBracketRotationType.single_commutator:
            raise_error(
                ValueError,
                "single_commutator DBR mode doesn't make sense with EvolutionOracle",
            )

        # This will run the appropriate group commutator step
        double_bracket_rotation_step = self.group_commutator(
            step_duration, diagonal_association, mode_dbr=mode_dbr
        )

        before_circuit = double_bracket_rotation_step["backwards"]
        after_circuit = double_bracket_rotation_step["forwards"]

        self.iterated_hamiltonian_evolution_oracle = FrameShiftedEvolutionOracle(
            deepcopy(self.iterated_hamiltonian_evolution_oracle),
            str(step_duration),
            before_circuit,
            after_circuit,
        )
        if self.please_evaluate_matrices:
            if (
                self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
                is EvolutionOracleType.numerical
            ):
                self.h.matrix = before_circuit @ self.h.matrix @ after_circuit

            elif (
                self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
                is EvolutionOracleType.hamiltonian_simulation
            ):

                self.h.matrix = (
                    before_circuit.unitary() @ self.h.matrix @ after_circuit.unitary()
                )

            elif (
                self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
                is EvolutionOracleType.text_strings
            ):
                raise_error(NotImplementedError)
            else:
                super().__call__(step_duration, diagonal_association.h.dense.matrix)

    def eval_gcr_unitary(
        self,
        step_duration: float,
        eo1: EvolutionOracle,
        eo2: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        u = self.group_commutator(step_duration, eo1, eo2, mode_dbr=mode_dbr)[
            "forwards"
        ]
        if eo1.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            return u.unitary()
        elif eo1.mode_evolution_oracle is EvolutionOracleType.numerical:
            return u

    def group_commutator(
        self,
        t_step: float,
        eo1: EvolutionOracle,
        eo2: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        s_step = np.sqrt(t_step)
        if eo2 is None:
            eo2 = self.iterated_hamiltonian_evolution_oracle

        assert eo1.mode_evolution_oracle.value is eo2.mode_evolution_oracle.value

        if mode_dbr is None:
            gc_type = self.mode_double_bracket_rotation
        else:
            gc_type = mode_dbr

        if gc_type is DoubleBracketRotationType.single_commutator:
            raise_error(
                ValueError,
                "You are trying to get the group commutator query list but your dbr mode is single_commutator and not an approximation by means of a product formula!",
            )

        if gc_type is DoubleBracketRotationType.group_commutator:
            query_list_forward = [
                eo2.circuit(-s_step),
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]          
            query_list_backward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
                eo2.circuit(s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reordered:
            query_list_forward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
                eo2.circuit(s_step),
            ]
            query_list_backward = [
                eo2.circuit(-s_step),
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reduced:
            query_list_forward = [
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]
            query_list_backward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_third_order:
            query_list_forward = [
                eo2.circuit(-s_step * (np.sqrt(5) - 1) / 2),
                eo1.circuit(-s_step * (np.sqrt(5) - 1) / 2),
                eo2.circuit(s_step),
                eo1.circuit(s_step * (np.sqrt(5) + 1) / 2),
                eo2.circuit(-s_step * (3 - np.sqrt(5)) / 2),
                eo1.circuit(-s_step)            
            ]
            query_list_backward = [ Circuit.invert(c) for c in query_list_forward[::-1]]
        elif gc_type is DoubleBracketRotationType.group_commutator_third_order_reduced:
            query_list_forward = [
                deepcopy(eo1).circuit(-s_step * (np.sqrt(5) - 1) / 2),
                deepcopy(eo2).circuit(s_step),
                deepcopy(eo1).circuit(s_step * (np.sqrt(5) + 1) / 2),
                deepcopy(eo2).circuit(-s_step * (3 - np.sqrt(5)) / 2),
                deepcopy(eo1).circuit(-s_step)            
            ]
            query_list_backward = [ Circuit.invert(c) for c in query_list_forward[::-1]]
        else:
            raise_error(
                ValueError,
                "You are in the group commutator query list but your dbr mode is not recognized",
            )

        eo_mode = eo1.mode_evolution_oracle

        if eo_mode is EvolutionOracleType.text_strings:
            return {
                "forwards": reduce(str.__add__, query_list_forward),
                "backwards": reduce(str.__add__, query_list_backward),
            }
        elif eo_mode is EvolutionOracleType.hamiltonian_simulation:
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

    def loss(self, step: float = None, eo_d = None):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step (float): iteration step.
            d (np.array): diagonal operator, use canonical by default.
            look_ahead (int): number of iteration steps to compute the loss function;
        """
        if step is None:
            circ = self.get_composed_circuit()  
        else:
            circ = self.group_commutator(step, eo_d)["forwards"] + \
            self.get_composed_circuit()  
        return self.h_ref.expectation(
            circ().state()
        )
    
    def choose_step( self,
        step_min: float = 1e-3,
        step_max: float = .03,
        s_guess = 1e-5,
        max_evals: int = 3,
        times = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
        d = None):        

        if times is None:
            times = np.linspace(step_min,step_max,max_evals)

        losses = []
        circ_boost = self.iterated_hamiltonian_evolution_oracle.get_composed_circuit()
        for s in times:
            losses.append(self.loss(s,d))
        if verbose:
            print(losses)
        return times[np.argmin(losses)], np.min(losses), losses
    def get_composed_circuit(self,s = None, eo_d = None):
        if s is None or eo_d is None:
            return self.iterated_hamiltonian_evolution_oracle.get_composed_circuit()    
        else:
            return self.recursion_step_circuit(s,eo_d)+self.iterated_hamiltonian_evolution_oracle.get_composed_circuit()
    
    def recursion_step_circuit(self,s, eo_d ):
        return self.group_commutator(s, eo_d)["forwards"]

    @staticmethod
    def count_gates(circuit, gate_type):
        t=0     
        for g in circuit.queue:
            if isinstance(g, gate_type):
                t = t +1
        return t
    
    def count_CNOTs(self,circuit = None):
        if circuit is None:
            circuit = self.get_composed_circuit( )
        return self.count_gates( circuit, gates.gates.CNOT )
        
    def count_CZs(self,circuit = None):
        if circuit is None:
            circuit = self.get_composed_circuit( )
        return self.count_gates( circuit, gates.gates.CZ )

    def print_gate_count_report(self):
        nmb_cz = self.count_CZs()
        nmb_cnot = self.count_CNOTs()
        print(f"The boosting circuit used {nmb_cnot} CNOT gates coming from compiled XXZ evolution and {nmb_cz} CZ gates from VQE.\n\
For {self.nqubits} qubits this gives n_CNOT/n_qubits = {nmb_cnot/self.nqubits} and n_CZ/n_qubits = {nmb_cz/self.nqubits}")
