import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from qibo import set_backend
from qibo.quantum_info import entangling_capability

from boostvqe.ansatze import build_circuit

set_backend("numpy")

N = 8
S = 500

nqubits = np.arange(2, N, 1)
nlayers = np.arange(2, N, 1)

meyer_wallach_ent_power = np.zeros((N - 2, N - 2))

for n in nqubits:
    for l in nlayers:
        print(f"Testing n: {n}, l: {l}")
        c = build_circuit(int(n), int(l))
        meyer_wallach_ent_power[n - 2][l - 2] = entangling_capability(c, S)

sns.heatmap(
    meyer_wallach_ent_power,
    cmap="coolwarm",
    annot=meyer_wallach_ent_power,
    xticklabels=nlayers,
    yticklabels=nqubits,
)
plt.title("Meyer-Wallach measure")
plt.xlabel("Layers")
plt.ylabel("Qubits")
plt.savefig("entangling_power.png")
