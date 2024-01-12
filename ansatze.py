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

    """
    tf_backend = construct_backend("tensorflow")
    parameters = tf_backend.tf.Variable(parameters, dtype=tf_backend.tf.complex128)

    with tf_backend.tf.GradientTape() as tape:
        circuit.set_parameters(parameters)
        final_state = tf_backend.execute_circuit(circuit).state()
        expectation = tf_backend.calculate_expectation_state(
            tf_backend.cast(hamiltonian.matrix), final_state, normalize=False
        )

    return hamiltonian.backend.cast(tape.gradient(expectation, parameters))


def loss(params, circuit, hamiltonian):
    """Loss function."""
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(
        final_state
    )

def callbacks(
    params,
    vqe,
    loss_list,
    loss_fluctuation,
    params_history,
    tracker=0,
):
    """
    Callback function that updates the energy, the energy fluctuations and
    the parameters lists.
    """
    energy, energy_fluctuation = loss(params, vqe.circuit, vqe.hamiltonian)
    loss_list.append(energy)
    loss_fluctuation.append(energy_fluctuation)
    params_history.append(params)
    tracker += 1