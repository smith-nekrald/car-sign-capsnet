import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_images_separately(images: np.array,
        figure_name: str,
        n_images: int) -> None:
    fig: plt.Figure; axes: plt.Axes
    fig, axes = plt.subplots(1, n_images)
    idx: int
    for idx in range(0, n_images):
        axes[idx].matshow(images[idx], cmap=matplotlib.cm.binary)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.axis('off')
    plt.savefig(figure_name, bbox_inches='tight',pad_inches = 0)

