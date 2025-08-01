import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NL = 100
NQ = 15

qubits = np.arange(2, NQ, 2)
layers = np.arange(2, NL, 5)

print(layers)

colors = sns.color_palette("inferno", n_colors=len(qubits)).as_hex()
exp_track = []

def plot_variance(qubits, layers, params=np.arange(4,10,1)):
    plt.figure(figsize=(6,6*6/8))

    for i, q in enumerate(qubits): 
        gradients = []
        for l in layers:
            # upload gradients of the model with l layers and q qubits
            grads = np.load(f"gradients/grads_l{l}_q{q}.npy")
            # compute the variance for each parameter
            y = np.var(np.abs(grads), axis=0)
            # to be resilient to fluctuations, mean over n parameter indexes
            gradients.append(np.mean(y[params]))

            if l == 17:
                exp_track.append(gradients[-1])

            
        plt.plot(
            layers, 
            gradients, 
            label=fr"$N_q$ = {q}", 
            color=colors[i], 
            marker=".", 
            markersize=5,
            lw=1,
        )
        plt.xticks(range(2, NL, 8))

    plt.ylabel("Var")
    plt.yscale("log")
    plt.legend(ncols=3)
    plt.xlabel("Layers")
    plt.savefig("bp_diagnostic.png", dpi=1200)

    print(exp_track)

    plt.figure(figsize=(6, 6*6/8))
    plt.plot(qubits, exp_track, color="black", marker=".", markersize=12, ls="--", lw=1)
    plt.xlabel("qubits")
    plt.ylabel("VAR[grad]")
    plt.yscale("log")
    plt.savefig("bp_exp.png", dpi=1200)

plot_variance(qubits, layers, np.arange(1,17,1))