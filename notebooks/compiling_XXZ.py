import numpy as np
from qibo import Circuit,gates

def add_gates(circuit,list_q_i,list_q_ip1,t,delta,steps):
    """
    Adds gates for an XXZ model time evolution step.
    
    Parameters:
    circuit (Circuit): The quantum circuit.
    list_q_i (list): Control qubit indices.
    list_q_ip1 (list): Target qubit indices.
    t (float): Total evolution time.
    delta (float): Coefficient for the Z component.
    steps (int): Number of time steps.

    Returns:
    Circuit: Updated circuit.
    """

    dt = t/steps
    alpha = -dt
    beta = -dt
    gamma = -delta*dt

    circuit.add(gates.RZ(q_ip1, -np.pi/2) for q_ip1 in list_q_ip1)
    circuit.add(gates.CNOT(q_ip1, q_i) for q_i,q_ip1 in zip(list_q_i,list_q_ip1))
    circuit.add(gates.RZ(q_i, -2*gamma + np.pi/2) for q_i in list_q_i)
    circuit.add(gates.RY(q_ip1, -np.pi/2+2*alpha) for q_ip1 in list_q_ip1)
    circuit.add(gates.CNOT(q_i, q_ip1) for q_i,q_ip1 in zip(list_q_i,list_q_ip1))
    circuit.add(gates.RY(q_ip1, -2*beta + np.pi/2) for q_ip1 in list_q_ip1)
    circuit.add(gates.CNOT(q_ip1, q_i) for q_i,q_ip1 in zip(list_q_i,list_q_ip1))
    circuit.add(gates.RZ(q_i, np.pi/2) for q_i in list_q_i)
    return circuit

def nqubit_XXZ_decomposition(nqubits,t,delta,steps):
    """
    Constructs an XXZ model circuit for n qubits, given by:
    .. math::
        H = \\sum_{i=0}^{N-1} \\left( X_i X_{i+1} + Y_i Y_{i+1} + \\delta Z_i Z_{i+1} \\right)
    
    This function decomposes the time evolution operator of the XXZ model
    into a sequence of quantum gates applied to a quantum circuit.
    
    Parameters:
    nqubits (int): Number of qubits.
    t (float): Total evolution time.
    delta (float): Coefficient for the Z component (default 0.5).
    steps (int): Number of time steps (default 1).

    Returns:
    Circuit: The final multi-layer circuit.

    Example:
        .. testcode::

            from compiling_XXZ_utils import *
            # Create circuit to decompose 6 qubits XXZ with 3 decomposition steps
            circ = nqubit_XXZ_decomposition(nqubits=6,t=0.01,delta=0.5,steps=3)
            print(circ.draw())
    """
    circuit = Circuit(nqubits = nqubits)

    # Create lists of even and odd qubit indices
    even_numbers = [num for num in range(nqubits) if num % 2 == 0]
    odd_numbers = [num for num in range(nqubits) if num % 2 == 1]

    if nqubits%2 == 0:
        # Handle even number of qubits
        even_numbers_end_0 = [num for num in range(1,nqubits) if num % 2 == 0]
        even_numbers_end_0.append(0)
        
        circuit = add_gates(circuit,even_numbers,odd_numbers,t,delta,steps)
        circuit = add_gates(circuit,odd_numbers,even_numbers_end_0,t,delta,steps)

    elif nqubits%2 == 1:
        # Handle odd number of qubits (since XXZ model has periodic boundary conditions)
        circuit = add_gates(circuit,even_numbers[:-1],odd_numbers,t,delta,steps)
        circuit = add_gates(circuit,odd_numbers,even_numbers[1:],t,delta,steps)
        circuit = add_gates(circuit,[nqubits-1],[0],t,delta,steps)

    # Create a multi-layer circuit with the time evolution steps
    multi_layer = Circuit(nqubits = nqubits)
    for step in range(steps):
        multi_layer += circuit
    return multi_layer
