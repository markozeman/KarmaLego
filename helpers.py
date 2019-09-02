import numpy as np
import matplotlib.pyplot as plt


def plot_entity(entity):
    """
    Plot entity's time intervals of events.

    :param entity: dict - key: state, value: list of time intervals of specific event
    :return: None
    """
    # colors = ['r', 'c', 'y', 'k', 'g', 'b', 'm']
    # labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    colors = ['r', 'g', 'b', 'k']
    labels = ['D', 'C', 'B', 'A']

    for i, label in enumerate(labels):
        for od, do in entity[label]:
            plt.hlines(i + 1, od, do, colors=colors[i])

    plt.yticks(np.arange(len(labels)) + 1, labels=labels)
    plt.xticks(np.arange(21))
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()


