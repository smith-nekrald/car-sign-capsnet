import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_images_separately(images, figure_name, n_images):
    fig = plt.figure()
    for idx in range(1, n_images + 1):
        ax = fig.add_subplot(1, n_images, idx)
        ax.matshow(images[idx - 1], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.savefig(figure_name)

