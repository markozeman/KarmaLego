import matplotlib.pyplot as plt
import numpy as np
from entities import entity_list


def plot_entity(entity):
    colors = ['r', 'c', 'y', 'k', 'g', 'b', 'm']
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    for i, label in enumerate(labels):
        for od, do in entity[label]:
            plt.hlines(i + 1, od, do, colors=colors[i])

    plt.yticks(np.arange(7) + 1, labels=labels)
    plt.xticks(np.arange(21))
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()


if __name__ == "__main__":
    entity = entity_list[0]
    plot_entity(entity)

