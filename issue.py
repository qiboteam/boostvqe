from qibo.symbols import *
import qibo
from qibo.transpiler.unitary_decompositions import two_qubit_decomposition
L = 7
H_def = sum([ Z(x)*Z(x+1) +X(x)*X(x+1) +Y(x)*Y(x+1) +0.5*X(x) for x in range(L-1)])
H_sym = qibo.hamiltonians.SymbolicHamiltonian(H_def)
circ = H_sym.circuit(.1)
for gate in circ.queue:
    gate_decomposition = two_qubit_decomposition(
        *gate.qubits, gate.matrix()
    )
    for gate_elem in gate_decomposition:
        print(gate_elem)