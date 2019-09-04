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


def lexicographic_sorting(entity):
    """
    Lexicographically order entity. Sorted function orders tuples first by the first element, then by the second, etc.

    :param entity: dict - key: state, value: list of time intervals of specific event
    :return: list with lexicographically ordered time intervals
    """
    return sorted([(*ti, state) for state, time_intervals in entity.items() for ti in time_intervals])


def temporal_relations(ti_1, ti_2, epsilon, max_distance):
    """
    Find out the temporal relation between time intervals ti_1 and ti_2 among 7 Allen's temporal relations.
    It is assumed that ti_1 is lexicographically before or equal to ti_2 (ti_1 <= ti_2).

    :param ti_1: first time interval (A.start, A.end)
    :param ti_2: second time interval (B.start, B.end)
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :return: string - one of 7 possible temporal relations
    """
    A_start, A_end = ti_1
    B_start, B_end = ti_2
    if epsilon < B_start - A_end < max_distance:    # before
        return '<'
    elif abs(B_start - A_end) <= epsilon:   # meets
        return 'm'
    elif B_start - A_start > epsilon and A_end - B_start > epsilon and B_end - A_end > epsilon:     # overlaps
        return 'o'
    elif B_start - A_start > epsilon and A_end - B_end > epsilon:   # contains
        return 'c'
    elif B_start - A_start > epsilon and abs(B_end - A_end) <= epsilon:     # finish by
        return 'f'
    elif abs(B_start - A_start) <= epsilon and abs(B_end - A_end) <= epsilon:   # equal
        return '='
    elif abs(B_start - A_start) <= epsilon and B_end - A_end > epsilon:     # starts
        return 's'
    else:
        print('Wrong temporal relation!')
        return '?'




def vertical_support_symbol(entity_list, symbol, min_ver_supp):
    """
    Check if symbol is present in at least min_ver_supp proportion of entities.

    :param entity_list: list of all entities
    :param symbol: symbol existing in at least one entity
    :param min_ver_supp: proportion (value between 0-1) defining threshold for accepting TIRP
    :return: boolean - True if given symbol has at least min_ver_supp support, otherwise False
             integer - value of symbol support
    """
    pass




