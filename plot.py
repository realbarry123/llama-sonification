import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_history(history):
    fig, ax = plt.subplots()
    bar_plot = plt.bar(range(0, len(history[0])), history[0], width=0.8, bottom=None)

    def update(frame):
        for i, b in enumerate(bar_plot):
            b.set_height(frame[i])

    game = animation.FuncAnimation(fig, update, frames=history)
    plt.show()

def scatterplot(data):
    fig, ax = plt.subplots()
    bar_plot = plt.scatter(range(0, len(data)), data, marker=".")
    plt.show()

def histogram(data, bin_size=5): 
    flat = torch.flatten(data).numpy()
    plt.hist(
        flat, 
        bins=np.arange(flat.min(), flat.max(), bin_size)
    )
    plt.show()
