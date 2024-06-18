from enum import Enum, auto

import hyperopt
import numpy as np

from qibo import Circuit, symbols
from qibo.config import raise_error
from qibo.hamiltonians import AbstractHamiltonian, SymbolicHamiltonian


from boostvqe.compiling_XXZ import *
from copy import deepcopy

from functools import reduce

class EvolutionOracleType(Enum):
    text_strings = auto()
    """If you only want to get a sequence of names of the oracle"""

    numerical = auto()
    """If you will work with exp(is_k J_k) as a numerical matrix"""

    hamiltonian_simulation = auto()
    """If you will use SymbolicHamiltonian"""


class EvolutionOracle:
    def __init__(
        self,
        h_generator: AbstractHamiltonian,
        name,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings,
    ):
        if (
            mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation
            and type(h_generator) is not SymbolicHamiltonian
        ):
            raise_error(
                TypeError,
                "If the evolution oracle mode will be to make Trotter-Suzuki decompositions then you must use the SymbolicHamiltonian generator",
            )
        if h_generator is None and name is None:
            raise_error(
                NotImplementedError,
                "You have to specify either a matrix and then work in the numerical mode, or SymbolicHamiltonian and work in hamiltonian_simulation mode or at least a name and work with text_strings to list DBI query lists",
            )
        
        if (
            mode_evolution_oracle is EvolutionOracleType.numerical
            and type(h_generator) is SymbolicHamiltonian
        ):
            self.h = h_generator.dense
        else:
            self.h = h_generator
        self.nqubits = self.h.nqubits
        self.name = name
        self.mode_evolution_oracle = mode_evolution_oracle
        self.mode_find_number_of_trottersuzuki_steps = True
        self.eps_trottersuzuki = 0.0001
        self.please_be_verbose = False
        self.please_use_prescribed_nmb_ts_steps = False

    def __call__(self, t_duration: float):
        """Returns either the name or the circuit"""
        return self.circuit(t_duration=t_duration)

    def eval_unitary(self, t_duration):
        """This wraps around `circuit` and always returns a unitary"""
        if self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.circuit(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            return self.circuit(t_duration).unitary()

    def circuit(self, t_duration: float = None):
        """This function returns depending on `EvolutionOracleType` string, ndarray or `Circuit`.
        In the hamiltonian_simulation mode we evaluate an appropriate Trotter-Suzuki discretization up to `self.eps_trottersuzuki` threshold.
        """
        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + str(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if self.please_use_prescribed_nmb_ts_steps is False:
                return self.discretized_evolution_circuit_binary_search(
                    t_duration, eps=self.eps_trottersuzuki
                    )
            else:
                dt = t_duration / self.please_use_prescribed_nmb_ts_steps
                return  reduce(
                    Circuit.__add__, 
                    [deepcopy(self.h).circuit(dt)] * self.please_use_prescribed_nmb_ts_steps
                    )

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        nmb_trottersuzuki_steps = 1  # this is the smallest size
        nmb_trottersuzki_steps_right = 800  # this is the largest size for binary search
        if eps is None:
            eps = self.eps_trottersuzuki
        target_unitary = self.h.exp(t_duration)

        def check_accuracy(n_steps):
            proposed_circuit_unitary = np.linalg.matrix_power(
                deepcopy(self.h).circuit(t_duration / n_steps).unitary(),
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

        circuit_1_step = deepcopy(self.h.circuit(t_duration / nmb_trottersuzuki_steps))
        combined_circuit = reduce(
            Circuit.__add__, [circuit_1_step] * nmb_trottersuzuki_steps
        )
        assert (
            np.linalg.norm(combined_circuit.unitary() - target_unitary) < eps
        ), f"{np.linalg.norm(combined_circuit.unitary() - target_unitary)},{eps}, {nmb_trottersuzuki_steps}"
        return combined_circuit



class FrameShiftedEvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        base_evolution_oracle: EvolutionOracle,
        name,
        before_circuit,
        after_circuit,
    ):

        assert isinstance(before_circuit, type(after_circuit))

        self.h = base_evolution_oracle.h
        self.base_evolution_oracle = base_evolution_oracle
        self.name = name + "(" + base_evolution_oracle.name + ")"
        self.mode_evolution_oracle = base_evolution_oracle.mode_evolution_oracle
        self.before_circuit = before_circuit
        self.after_circuit = after_circuit
        self.nqubits = base_evolution_oracle.nqubits
    def circuit(self, t_duration: float = None):

        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + "(" + str(t_duration) + ")"
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return (
                self.before_circuit
                @ self.base_evolution_oracle(t_duration)
                @ self.after_circuit
            )
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:

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
            if self.mode_evolution_oracle is EvolutionOracleType.numerical:
                c =  c @fseo.after_circuit 
            elif (
                self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation
            ):
                c =  c +fseo.after_circuit 
            fseo = fseo.base_evolution_oracle
        return c
    
class VQERotatedEvolutionOracle(FrameShiftedEvolutionOracle):
    def __init__(
        self,
        base_evolution_oracle: EvolutionOracle,        
        vqe,
        name = "VQE Rotated EO",
    ):
        super().__init__(base_evolution_oracle, before_circuit=vqe.circuit.invert(), 
                                                        after_circuit=vqe.circuit,name="shifting by vqe")
        self.vqe = vqe

class MagneticFieldEvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        b_list,
        name = "B Field",
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation   
    ):
        self.nqubits = len(b_list)
        d = SymbolicHamiltonian( sum([b*symbols.Z(j) for j,b in zip(range(self.nqubits),b_list)]))
        super().__init__(            
        d,
        name,
        mode_evolution_oracle
        )      
        self.b_list = b_list
        self.please_assess_how_many_steps_to_use = False #otherwise methods which cast to dense will be used

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        if self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        
        if self.please_assess_how_many_steps_to_use:
            return super().discretized_evolution_circuit_binary_search(t_duration,eps=eps)
        else:
            return self.h.circuit(t_duration)

class IsingNNEvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        b_list,
        j_list,
        name = "B Field",
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation   
    ):
        self.nqubits = len(b_list)
        d = SymbolicHamiltonian( 
            sum([b*symbols.Z(j) for j,b in zip(range(self.nqubits),b_list)]+
                [j_list[j] * symbols.Z(j)*symbols.Z((j+1) % self.nqubits) for j in range(self.nqubits)]
                ))
        super().__init__(            
        d,
        name,
        mode_evolution_oracle
        )      
        self.b_list = b_list
        self.please_assess_how_many_steps_to_use = False #otherwise methods which cast to dense will be used

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        if self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        
        if self.please_assess_how_many_steps_to_use:
            return super().discretized_evolution_circuit_binary_search(t_duration,eps=eps)
        else:
            return self.h.circuit(t_duration)
class XXZ_EvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        nqubits,
        name = "XXZ",
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.hamiltonian_simulation,
        steps=None,
        order=None,
        delta = 0.5        
    ):
        super().__init__(            
        XXZ_EvolutionOracle.xxz_symbolic(nqubits, delta = delta),
        name,
        mode_evolution_oracle
        )

        if steps is None:
            self.steps = 1
        else:
            self.steps = steps
        if order is None:
            self.order = 1
        else:
            self.order = order
        self.nqubits = nqubits
        self.delta = delta
        self.please_assess_how_many_steps_to_use = False

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        if self.please_assess_how_many_steps_to_use:
            return super().discretized_evolution_circuit_binary_search(t_duration,eps=eps)
        else:
            return self.h.circuit(t_duration)
    def circuit(self, t_duration,steps=None,order=None):
        if steps is None:
            steps = self.steps
        if order is None:
            order = self.order
        return nqubit_XXZ_decomposition(
            nqubits=self.h.nqubits,t=t_duration,delta=self.delta,steps=steps,order=order)
    
    @staticmethod
    def xxz_symbolic(nqubits, delta = 0.5):
        return SymbolicHamiltonian(
                            sum([ symbols.X(j)*symbols.X(j+1) + symbols.Y(j)*symbols.Y(j+1) +delta*symbols.Z(j)*symbols.Z(j+1) for j in range(nqubits-1)] 
                                + [ symbols.X(nqubits-1)*symbols.X(0) + symbols.Y(nqubits-1)*symbols.Y(0) +delta*symbols.Z(nqubits-1)*symbols.Z(0)]),
                                nqubits= nqubits
                            )





