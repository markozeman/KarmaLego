import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from entities import entity_list


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
        return None


def vertical_support_symbol(entity_list, symbol, min_ver_supp):
    """
    Check if symbol is present in at least min_ver_supp proportion of entities.

    :param entity_list: list of all entities
    :param symbol: string of symbol existing in at least one entity
    :param min_ver_supp: proportion (value between 0-1) defining threshold for accepting TIRP
    :return: boolean - True if given symbol has at least min_ver_supp support, otherwise False
             integer - value of symbol support
    """
    support_count = 0
    for entity in entity_list:
        if symbol in list(entity.keys()) and len(entity[symbol]) != 0:
            support_count += 1

    support = support_count / len(entity_list)
    return support >= min_ver_supp, support


def find_match_recursively(symbol_occurrences, curr_number, arr_number, indices):
    """
    Find if TIRP symbols match entity symbols with recursive function.

    :param symbol_occurrences: 2D list with TIRP symbols occurrences in entity symbols
    :param curr_number: current index in entity symbols list
    :param arr_number: index of list in symbol_occurrences
    :param indices: list of indices of symbols matching, length at the end of the function = len(symbol_occurrences)
    :return: indices after recursion or None if it is not a match
    """
    if arr_number >= len(symbol_occurrences):
        return indices
    curr_array = symbol_occurrences[arr_number]
    for x in curr_array:
        if x > curr_number:
            indices.append(x)
            res = find_match_recursively(symbol_occurrences, x, arr_number + 1, indices)
            if res is not None:
                return res
    return None


def find_all_possible_matches(symbol_occurrences):
    """
     Find if TIRP symbols match entity symbols. Find all possible solutions.

    :param symbol_occurrences: 2D list with TIRP symbols occurrences in entity symbols
    :return: 2D list of indices or None if no match is found
    """
    all_options = list(itertools.product(*symbol_occurrences))
    possible_options = list(filter(filter_for_matches, all_options))
    return possible_options if len(possible_options) != 0 else None


def filter_for_matches(lst):
    """
    Filter for finding only possible matches.

    :param lst: list of numbers (indices)
    :return: True if numbers in lst are strictly increasing, otherwise False
    """
    for item_1, item_2 in zip(lst, lst[1:]):
        if item_1 >= item_2:
            return False
    return True


def check_symbols_lexicographically(entity_symbols, tirp_symbols, single_or_all='all'):
    """
    Check if symbols in entity and TIRP are lexicographically equivalent. That means that symbols in entity
    are in the same lexicographical order as in TIRP, but they do not have to be consecutive.

    :param entity_symbols: list of lexicographically ordered entity symbols
    :param tirp_symbols: list of lexicographically ordered TIRP symbols
    :param single_or_all: string with possible values 'all' or 'single'
    :return: 2D list - list of all possible indices in entity list where symbols match the TIRP symbols (if 'all')
             2D list with only one element - indices in entity list where symbols match the TIRP symbols (if 'single')
    """
    symbol_occurrences = [list(filter(lambda i: entity_symbols[i] == tirp_sym, range(len(entity_symbols)))) for tirp_sym in tirp_symbols]
    if single_or_all == 'all':
        return find_all_possible_matches(symbol_occurrences)
    elif single_or_all == 'single':
        return [find_match_recursively(symbol_occurrences, -1, 0, [])]





if __name__ == "__main__":
    entity_symbols = ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'a', 'k', 'b', 'b', 'l', 'd', 'd']
    tirp_symbols = ['a', 'b', 'd', 'b', 'b', 'd']

    i = check_symbols_lexicographically(entity_symbols, tirp_symbols, 'single')
    print(i)


    print(vertical_support_symbol(entity_list, 'C', 0.1))





