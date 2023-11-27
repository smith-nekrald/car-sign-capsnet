""" Visualization API. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_images_separately(images: np.array, figure_name: str, n_images: int) -> None:
    """ Plots a sequence of images on the same figure, and saves result to a file. 

    Args:
        images: The array with images.
        figure_name: The path to save figure.
        n_images: Number of images.
    """
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

