import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_images_separately(images: np.array,
        figure_name: str,
        n_images: int) -> None:
    fig: plt.Figure = plt.figure()
    idx: int
    for idx in range(1, n_images + 1):
        ax: plt.Axes = fig.add_subplot(1, n_images, idx)
        ax.matshow(images[idx - 1], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.savefig(figure_name)

