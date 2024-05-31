
from qibo.models.variational import VQE
# from boostvqe.ansatze import build_circuit
from qibo import hamiltonians, Circuit,gates
from pathlib import Path
import numpy as np
import scipy


### CNOT + single qubit rotation decomposition of each 2-qubit XXZ evolution for n-qubit XXZ Trotterisation

def single_XXZ_decomposition(nqubits,q_i,q_ip1,t,delta):
#     This is a code that decomposes e^{-it(XX + YY + delta*ZZ)}
    alpha = -t
    beta = -t
    gamma = -delta*t
    circuit = Circuit(nqubits = nqubits)
    circuit.add(gates.RZ(q_ip1, -np.pi/2))
    circuit.add(gates.CNOT(q_ip1, q_i))
    circuit.add(gates.RZ(q_i, -2*gamma + np.pi/2))
    circuit.add(gates.RY(q_ip1, -np.pi/2+2*alpha))
    circuit.add(gates.CNOT(q_i, q_ip1))
    circuit.add(gates.RY(q_ip1, -2*beta + np.pi/2))
    circuit.add(gates.CNOT(q_ip1, q_i))
    circuit.add(gates.RZ(q_i, np.pi/2))
    return circuit


### Test for 2-qubit XXZ

# In[74]:


delta = 3
t = 1
# Multiplying 0.5 to h_xxz for making it h_XXZ = (XX + YY + delta*ZZ)
# Without this multiplication, we get h_XXZ = 2(XX + YY + delta*ZZ) due to the periodic BC of hamiltonians.XXZ
h_xxz = 0.5*hamiltonians.XXZ(nqubits=2, delta = delta)
circ = single_XXZ_decomposition(nqubits = 2,q_i = 0,q_ip1 = 1,t = t,delta = delta)


# In[75]:


u = h_xxz.exp(t)
v = circ.unitary()
print(v)
print(u)
np.linalg.norm(u-(1+1j)*v/np.sqrt(2))


# ## n-qubit function (even qubits for now)

# In[155]:


def nqubit_XXZ_decomposition(nqubits,t,delta,steps):
    dt = t/steps
    alpha = -dt
    beta = -dt
    gamma = -delta*dt
    circuit = Circuit(nqubits = nqubits)
    even_qubits = np.arange(0,nqubits,2)
    odd_qubits = np.arange(1,nqubits,2)
    if nqubits%2 == 0:
#         commented out RZ rotations that cancel out with each other
#         circuit.add(gates.RZ(q_i+1, -np.pi/2) for q_i in even_qubits)
        circuit.add(gates.CNOT(q_i+1, q_i) for q_i in even_qubits)
        circuit.add(gates.RZ(q_i, -2*gamma + np.pi/2) for q_i in even_qubits)
        circuit.add(gates.RY(q_i+1, -np.pi/2+2*alpha) for q_i in even_qubits)
        circuit.add(gates.CNOT(q_i, q_i+1) for q_i in even_qubits)
        circuit.add(gates.RY(q_i+1, -2*beta + np.pi/2) for q_i in even_qubits)
        circuit.add(gates.CNOT(q_i+1, q_i) for q_i in even_qubits)
        
        circuit.add(gates.CNOT(even_qubits[n],odd_qubits[n-1]) for n in range(nqubits//2))
        circuit.add(gates.RZ(q_i+1, -2*gamma + np.pi/2) for q_i in even_qubits)
        circuit.add(gates.RY(q_i, -np.pi/2+2*alpha) for q_i in even_qubits)
        circuit.add(gates.CNOT(odd_qubits[n-1],even_qubits[n]) for n in range(nqubits//2))
        circuit.add(gates.RY(q_i, -2*beta + np.pi/2) for q_i in even_qubits)
        circuit.add(gates.CNOT(even_qubits[n],odd_qubits[n-1]) for n in range(nqubits//2))
#         circuit.add(gates.RZ(q_i+1, np.pi/2) for q_i in even_qubits)
        
        multi_layer = Circuit(nqubits = nqubits)
        multi_layer.add(gates.RZ(q_i+1, -np.pi/2) for q_i in even_qubits)
        for step in range(steps):
            multi_layer += circuit
        multi_layer.add(gates.RZ(q_i+1, np.pi/2) for q_i in even_qubits)
        return multi_layer
    
    elif nqubits%2 == 1:
        circuit.add(gates.RZ(q_i, -np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i, q_i-1) for q_i in odd_qubits)
        circuit.add(gates.RZ(q_i-1, -2*gamma + np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.RY(q_i, -np.pi/2+2*alpha) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i-1, q_i) for q_i in odd_qubits)
        circuit.add(gates.RY(q_i, -2*beta + np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i, q_i-1) for q_i in odd_qubits)
        circuit.add(gates.RZ(q_i-1, np.pi/2) for q_i in odd_qubits)
        
        circuit.add(gates.RZ(q_i+1, -np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i+1, q_i) for q_i in odd_qubits)
        circuit.add(gates.RZ(q_i, -2*gamma + np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.RY(q_i+1, -np.pi/2+2*alpha) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i, q_i+1) for q_i in odd_qubits)
        circuit.add(gates.RY(q_i+1, -2*beta + np.pi/2) for q_i in odd_qubits)
        circuit.add(gates.CNOT(q_i+1, q_i) for q_i in odd_qubits)
        circuit.add(gates.RZ(q_i, np.pi/2) for q_i in odd_qubits)
        
        last = nqubits-1
        circuit.add(gates.RZ(0, -np.pi/2))
        circuit.add(gates.CNOT(0, last))
        circuit.add(gates.RZ(last, -2*gamma + np.pi/2))
        circuit.add(gates.RY(0, -np.pi/2+2*alpha))
        circuit.add(gates.CNOT(last, 0))
        circuit.add(gates.RY(0, -2*beta + np.pi/2))
        circuit.add(gates.CNOT(0, last))
        circuit.add(gates.RZ(last, np.pi/2))
        
        multi_layer = Circuit(nqubits = nqubits)
        for step in range(steps):
            multi_layer += circuit
        return multi_layer


# In[157]:


t = 0.01
steps = 1
delta=0.5
nqubits=11

h_xxz = hamiltonians.XXZ(nqubits=nqubits, delta = delta)
u = h_xxz.exp(t)
circ = nqubit_XXZ_decomposition(nqubits=nqubits,t=t,delta=delta,steps=steps)
v = circ.unitary()
print(np.linalg.norm(u-np.exp(nqubits*steps*1j*np.pi/4)*v))
print(circ.draw())


# In[158]:


print(circ.summary())


# In[ ]:




