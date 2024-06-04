import json
import time
from pathlib import Path

import numpy as np
import qibo
from qibo import hamiltonians, set_backend
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)

from boostvqe.ansatze import VQE, build_circuit
from boostvqe.utils import apply_dbi_steps, rotate_h_with_vqe

qibo.set_backend("numpy")

# set the path string which define the results
path = "../results/vqe_data/with_params/10q3l/sgd_10q_3l_42/"

# set the target epoch to which apply DBQA and the number of steps
target_epoch = 20
dbi_steps = 1

# upload system configuration and parameters for all the training
with open(path + "optimization_results.json") as file:
    config = json.load(file)

losses = dict(np.load(path + "energies.npz"))["0"]
params = np.load(path + f"parameters/params_ite{target_epoch}.npy")


# build circuit, hamiltonian and VQE
hamiltonian = hamiltonians.XXZ(nqubits=config["nqubits"], delta=0.5)
circuit = build_circuit(config["nqubits"], config["nlayers"], "numpy")
vqe = VQE(circuit, hamiltonian)
zero_state = hamiltonian.backend.zero_state(config["nqubits"])
target_energy = np.min(hamiltonian.eigenvalues())


# set target parameters into the VQE
vqe.circuit.set_parameters(params)
vqe_state = vqe.circuit().state()

ene1 = hamiltonian.expectation(vqe_state)

# DBQA stuff
t0 = time.time()
print("Rotating with VQE")
new_hamiltonian_matrix = rotate_h_with_vqe(hamiltonian=hamiltonian, vqe=vqe)
new_hamiltonian = hamiltonians.Hamiltonian(
    config["nqubits"], matrix=new_hamiltonian_matrix
)
print(time.time() - t0)
dbi = DoubleBracketIteration(
    hamiltonian=new_hamiltonian,
    mode=DoubleBracketGeneratorType.single_commutator,
)

zero_state_t = np.transpose([zero_state])
energy_h0 = float(dbi.h.expectation(np.array(zero_state_t)))
fluctuations_h0 = float(dbi.h.energy_fluctuation(zero_state_t))

print("Applying DBI steps")
(
    _,
    dbi_energies,
    dbi_fluctuations,
    _,
    _,
    _,
) = apply_dbi_steps(
    dbi=dbi,
    nsteps=dbi_steps,
)
print(time.time() - t0)
print(
    f"\nReached accuracy before DBI at iter {target_epoch}: {np.abs(target_energy - ene1)}"
)
print(
    f"Reached accuracy after DBI at iter {target_epoch}: {np.abs(target_energy - dbi_energies[-1])}"
)
print(
    f"Reached accuracy in the end of VQE long training: {np.abs(target_energy - losses[-1])}"
)
