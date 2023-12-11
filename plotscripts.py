import numpy as np
import matplotlib.pyplot as plt

RED = "#f54242"
YELLOW = "#edd51a"
GREEN = "#2db350"
PURPLE = "#587ADB"
BLUE = "#4287f5"

def plot_matrix(matrix, title="", save=True, width=0.5):
    """
    Visualize hamiltonian in a heatmap form.
    
    Args:
        matrix (np.ndarray): target matrix to be represented in heatmap form.
        title (str): figure title.
        save (bool): if ``True``, the figure is saved as `./plots/matrix_title.pdf`.
        width (float): ratio of the LaTeX manuscript which will be occupied by 
            the figure. This argument is useful to standardize the image and font sizes. 
    """
    fig, ax = plt.subplots(figsize=(10 * width, 10 * width))
    ax.set_title(title)
    try:
        im = ax.imshow(np.absolute(matrix), cmap="inferno")
    except TypeError:
        im = ax.imshow(np.absolute(matrix.get()), cmap="inferno")
    fig.colorbar(im, ax=ax)
    if save:
        plt.savefig(f"plots/matrix_{title}.pdf", bbox_inches="tight")
        

def plot_loss(loss_history, title="", save=True, width=0.5):
    """
    Plot loss function history.
    
    Args:
        loss_history (list or np.ndarray): loss function history.
        title (str): figure title.
        save (bool): if ``True``, the figure is saved as `./plots/matrix_title.pdf`.
        width (float): ratio of the LaTeX manuscript which will be occupied by 
            the figure. This argument is useful to standardize the image and font sizes. 
    """
    plt.figure(figsize=(10 * width, 10 * width * 6/8))
    plt.title(title)
    plt.plot(loss_history, lw=1.5, color=BLUE)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if save:
        plt.savefig(f"plots/loss_{title}.pdf", bbox_inches="tight")