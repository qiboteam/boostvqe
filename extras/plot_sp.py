import os

import numpy as np
import matplotlib.pyplot as plt

NQUBITS = np.arange(2, 10, 2)
# NLAYERS = np.arange(1, 101, 5)

init_var, final_var = [], []

for q in NQUBITS:
    init_varq, final_varq = [], []
    for f in os.listdir("sp_data"):
        if f"{q}q" in f:
            if "init" in f:
                init_varq.append(np.var(np.abs(np.load("sp_data/" + f))))
            elif "final" in f:
                final_varq.append(np.var(np.abs(np.load("sp_data/" + f))))
    
    init_var.append(np.mean(init_varq))
    final_var.append(np.mean(final_varq))


plt.figure(figsize=(6, 6*6/8))
plt.plot(NQUBITS, init_var, marker=".", markersize=12, ls="--", lw=1, color="red", label="Before DBI")
plt.plot(NQUBITS, final_var, marker=".", markersize=12, ls="--", lw=1, color="royalblue", label="After DBI")
plt.legend()
plt.xlabel("layers")
plt.ylabel("VAR(grads)")
plt.yscale("log")
plt.savefig("state_preparation.png", dpi=500)

