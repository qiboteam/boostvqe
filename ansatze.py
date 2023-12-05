import qibo
from qibo import gates
from qibo.backends import construct_backend
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


def compute_gradients(parameters, circuit, hamiltonian):
    """
    Compute gradients of circuit's parameters to check the problem trainability.
    The evaluated derivatives are the ones of the expectation of `hamiltonian`
    over the final state get running `circuit.execute` w.r.t. rotational angles.

    NOTE: set "tensorflow" backend to make this function work.
    """
    backend = hamiltonian.backend 
    parameters = backend.tf.Variable(parameters, dtype=backend.tf.complex128)

    with backend.tf.GradientTape() as tape:
        circuit.set_parameters(parameters)
        expectation = hamiltonian.expectation(backend.execute_circuit(circuit).state())
        
    return tape.gradient(expectation, parameters)