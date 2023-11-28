from qibo import gates
from qibo.models import Circuit

def build_circuit(nqubits, nlayers):
    """Build qibo's aavqe example circuit."""

    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(0, nqubits - 1, 2))
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(1, nqubits - 2, 2))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
    circuit.add(gates.M(*range(nqubits)))

    return circuit