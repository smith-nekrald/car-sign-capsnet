import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_images_separately(images, figure_name):
    fig = plt.figure()
    for j in range(1, 7):
        ax = fig.add_subplot(1, 6, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.savefig(figure_name)

