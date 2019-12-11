import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import json
import pickle
from transition_table import *
from copy import deepcopy
from collections import defaultdict, Counter


def write2json(filename, data):
    """
    Write data to JSON file with name filename.

    :param filename: name of the file we write to
    :param data: data to write
    :return: None
    """
    with open(filename, 'w') as f:
        json.dump(data, f)


def read_json(filename):
    """
    Read JSON file with name filename.

    :param filename: name of the file we read from
    :return: content of the file
    """
    with open(filename, "r") as content:
        return json.loads(content.read())


def save_pickle(filename, data):
    """
    Save Python object with pickle.

    :param filename: name of the file we write to
    :param data: data to write
    :return: None
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """
    Load pickle from disk.

    :param filename: name of the file we read from
    :return: content of the file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def plot_entity(entity, ver_supp=None, number_of_all_tirps=None, number_of_patients=None):
    """
    Plot entity's time intervals of events. Show vertical support if ver_supp is not None.

    :param entity: dict - key: state, value: list of time intervals of specific event
    :param ver_supp: optional parameter for vertical support
    :param number_of_all_tirps: integer number of all TIRPs above minimum support
    :param number_of_patients: integer number of all patients in entity list used for calculating showed TIRPs
    :return: None
    """
    # make fullscreen window and clear figure
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.clf()

    colors = ['r', 'c', 'y', 'k', 'b', 'g', 'm']
    labels = list(entity.keys())
    padding = 0.25
    min_num = np.inf
    max_num = -np.inf
    for i, label in enumerate(labels):
        for od, do in entity[label]:
            if od == do:    # draw time event point
                plt.plot(od, i + 1, color=colors[i % len(colors)], marker='x', markersize=10)
            else:   # draw time interval line
                plt.hlines(i + 1, od, do, colors=colors[i % len(colors)])

            plt.vlines(od, 1 - padding, len(labels) + padding, colors='lightgray', linestyles='dotted')
            plt.vlines(do, 1 - padding, len(labels) + padding, colors='lightgray', linestyles='dotted')

            # set min and max number
            if od < min_num:
                min_num = od
            if do > max_num:
                max_num = do

    if None not in (ver_supp, number_of_all_tirps, number_of_patients):
        global tirp_indexxx
        plt.title('TIRP: %d / %d\n\nVertical support: %.2f  (%d patients)'
                  % (tirp_indexxx + 1, number_of_all_tirps, ver_supp, round(number_of_patients * ver_supp)))

    plt.yticks(np.arange(len(labels)) + 1, labels=labels)
    plt.xticks(np.arange(min_num - 2, max_num + 3))
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()


def lexicographic_sorting(entity):
    """
    Lexicographically order entity. Sorted function orders tuples first by the first element, then by the second, etc.
    If time interval is given backwards e.g. (8, 5) it is transformed to (5, 8).

    :param entity: dict - key: state, value: list of time intervals of specific event
    :return: list with lexicographically ordered time intervals
    """
    return sorted([(*ti, state) if ti[0] <= ti[1] else (ti[1], ti[0], state) for state, time_intervals in entity.items() for ti in time_intervals])


def temporal_relations(ti_1, ti_2, epsilon, max_distance):
    """
    Find out the temporal relation between time intervals ti_1 and ti_2 among 7 Allen's temporal relations.
    It is assumed that ti_1 is lexicographically before or equal to ti_2 (ti_1 <= ti_2).

    :param ti_1: first time interval (A.start, A.end)
    :param ti_2: second time interval (B.start, B.end)
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :return: string - one of 7 possible temporal relations or None if relation is unknown
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
        # print('Other temporal relation!')
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
    Filter for finding only possible matches depending on list strictly increasing.

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


def are_TIRPs_equal(tirp_1, tirp_2):
    """
    Check if two TIRPs are exactly the same.

    :param tirp_1: first TIRP
    :param tirp_2: second TIRP
    :return: boolean - True if TIRPS are the same, else False
    """
    return tirp_1.symbols == tirp_2.symbols and tirp_1.relations == tirp_2.relations


def find_all_possible_extensions(all_paths, path, BrC, curr_rel_index, decrement_index, TIRP_relations):
    """
    Recursively find all possible relations extensions of new (last) column of TIRP based on transition table.

    :param all_paths: 2D list - all possible extensions (this list is returned)
    :param path: list - one possible relations extension of last column
    :param BrC: current relation in new (last) column
    :param curr_rel_index: index of current relation in TIRP_relations list
    :param decrement_index: number defining how much curr_rel_index is decremented each time
    :param TIRP_relations: list of relations of the current TIRP
    :return: 2D list - each element is a list representing one of possible extensions of last column of TIRP
             (inner lists don't contain relation between last 2 symbols i.e. the most lower relation in half matrix)
    """
    if curr_rel_index < 0:
        all_paths.append(deepcopy(path))
        return

    ArB = TIRP_relations[curr_rel_index]
    poss_relations = transition_table[(ArB, BrC)]

    for poss_rel in poss_relations:
        path.append(poss_rel)
        decrement_index -= 1
        find_all_possible_extensions(all_paths, path, poss_rel, curr_rel_index - decrement_index - 1, decrement_index, TIRP_relations)
        decrement_index += 1
        del path[-1]    # delete last element from path list

    return all_paths


def visualize_tirp(tirp, entity_list, epsilon, max_distance, number_of_all_tirps):
    """
    Visualize TIRP as one of existing examples from entity_list.

    :param tirp: TIRP to visualize
    :param entity_list: list of all entities
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :param number_of_all_tirps: integer number of all TIRPs above minimum support
    :return: None
    """
    all_options = []
    for index in tirp.entity_indices_supporting:
        patient_dict = entity_list[index]
        lexi_ordered = lexicographic_sorting(patient_dict)
        entity_ti = list(map(lambda x: (x[0], x[1]), lexi_ordered))
        entity_symbols = list(map(lambda x: x[2], lexi_ordered))
        symbols_match_TIRP = check_symbols_lexicographically(entity_symbols, tirp.symbols)

        for tup in symbols_match_TIRP:
            matching_intervals = list(np.array(entity_ti)[list(tup)])

            # check if given tirp has the same relations as intervals in matching_intervals
            curr_interval_index = 1
            relation_index = 0
            all_relations_match = True
            while curr_interval_index < len(matching_intervals):
                second_interval = matching_intervals[curr_interval_index]
                for i in range(curr_interval_index):
                    first_interval = matching_intervals[i]
                    relation = temporal_relations(first_interval, second_interval, epsilon, max_distance)
                    if relation != tirp.relations[relation_index]:
                        all_relations_match = False
                        break
                    relation_index += 1
                if not all_relations_match:
                    break
                curr_interval_index += 1

            if all_relations_match:
                all_options.append(matching_intervals)

    labels = tirp.symbols
    time_differences = list(map(min_max_difference, all_options))

    # select option with the smallest time difference between first and last time point
    time_intervals = all_options[time_differences.index(min(time_differences))]

    # change to entity dict and plot it
    entity = {}
    for label, ti in zip(labels, time_intervals):
        if label in entity:
            entity[label].append(ti)
        else:
            entity[label] = [ti]

    plot_entity(entity, tirp.vertical_support, number_of_all_tirps, len(entity_list))


def min_max_difference(l):
    """
    Find difference between maximum and minimum of 2D list l.
    First nested element is minimum, last nested element is maximum.

    :param l: 2D list of numbers
    :return: max - min
    """
    first, *_, last = np.concatenate(l)
    return last - first


def on_key_pressed(event, sorted_tirps, entity_list, epsilon, max_distance, number_of_all_tirps):
    """
    Function that reacts on event of key pressed on the plot.
    Pressing right arrow key moves plot forward to next TIRP.
    Pressing left arrow key moves plot backwards to previous TIRP.

    :param event: Python mpl event
    :param sorted_tirps: list of all TIRPs ordered by vertical support in decreasing order
    :param entity_list: list of all entities
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :param number_of_all_tirps: integer number of all TIRPs above minimum support
    :return:
    """
    global tirp_indexxx
    if event.key == 'right' and tirp_indexxx < len(sorted_tirps) - 1:
        tirp_indexxx += 1
    elif event.key == 'left' and tirp_indexxx > 0:
        tirp_indexxx -= 1

    try:
        visualize_tirp(sorted_tirps[tirp_indexxx], entity_list, epsilon, max_distance, number_of_all_tirps)
    except RecursionError:
        print("\nMaximum recursion depth exceeded. Figure exited. Run the program again.")
        plt.close()


def visualize_tirps_from_file(tree, entity_list, epsilon, max_distance, order_by, search_drug=None, search_type=None):
    """
    Visualize TIRPs from tree stored in 'filename' file based on vertical support in decreasing order or TIRP size.

    :param tree: tree of TIRPs
    :param entity_list: list of all entities
    :param epsilon: maximum amount of time between two events that we consider it as the same time
    :param max_distance: maximum distance between two time intervals that means first one still influences the second
    :param order_by: string telling the way to sort TIRP nodes from KarmaLego tree ('vertical support' or 'TIRP size')
    :param search_drug: string representing drug to search for in a tree; if None the whole tree is visualized
    :param search_type: string representing type of drug search:
                        'subtree' (collect all the nodes from search_drug down) or
                        'included' (collect all nodes that have search_drug included in the TIRP);
                        if None the whole tree is visualized
    :return: None
    """
    all_nodes = tree.find_tree_nodes([])    # used if search_drug or search_type are None

    if None not in (search_drug, search_type):
        if search_type == 'subtree':
            match_found = False
            for child in tree.children:
                if child.data == search_drug:
                    match_found = True
                    all_nodes = child.find_tree_nodes([])
                    break
            if not match_found:
                raise NameError('This search drug does not exist in this tree!')
        elif search_type == 'included':
            all_nodes = list(filter(lambda node: search_drug in node.symbols, all_nodes))
            if len(all_nodes) == 0:
                raise NameError('This search drug does not exist in this tree!')
        else:
            raise NameError('Unknown search type!')

    if order_by == 'vertical support':
        sorted_tirps = sorted(all_nodes, reverse=True)
    elif order_by == 'TIRP size':
        sorted_tirps = sorted(all_nodes, key=lambda tirp: (len(tirp.symbols), tirp.vertical_support), reverse=True)
    else:
        raise NameError('Unknown way of ordering TIRPs!')

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event',
                           lambda event: on_key_pressed(event, sorted_tirps, entity_list, epsilon, max_distance, len(all_nodes)))

    global tirp_indexxx
    tirp_indexxx = 0

    if sorted_tirps:
        visualize_tirp(sorted_tirps[tirp_indexxx], entity_list, epsilon, max_distance, len(all_nodes))


def ordered_diagnoses4clustering():
    """
    Read data from 3 files and based on that return patient diagnoses in the right order.

    :return: list of strings, each string has diagnoses of one patient (divided by | sign)
    """
    d = get_patient_diagnoses('csv/all_admissions.csv')
    all_patient_ids = np.array(load_pickle('data/pickle/patient_IDs_admissions.pickle'))
    randomly_sampled_indices = np.array(load_pickle('data/pickle/sampled_indices.pickle'))
    patient_ids = list(all_patient_ids[randomly_sampled_indices])
    ordered_diagnoses = [' | '.join(d[pat_id]) for pat_id in patient_ids]
    return ordered_diagnoses


def all_ordered_diagnoses4clustering():
    """
    Read data from 2 files and based on that return patient diagnoses in the right order.

    :return: list of strings, each string has diagnoses of one patient (divided by | sign)
    """
    d = get_patient_diagnoses('csv/all_admissions.csv')
    all_patient_ids = list(load_pickle('data/pickle/patient_IDs_prescriptions.pickle'))
    ordered_diagnoses = [' | '.join(d[pat_id]) for pat_id in all_patient_ids]
    return ordered_diagnoses


def get_patient_diagnoses(filename):
    """
    Read file and make dictionary - key: patient_id, value: list of diagnoses

    :param filename: name of the file to read
    :return: dictionary connecting patients and diagnoses
    """
    res = pd.read_csv(filename, sep='\t', index_col=0)
    unique_tuples = list(set(zip(res.patient_id, res.diagnosis)))
    d = defaultdict(list)
    for id, diag in unique_tuples:
        d[id].append(diag)
    return d


def remove_electrolytes(entity_list, filename):
    """
    Remove electrolytes from given entity_list.

    :param entity_list: list of all entities
    :param filename: name of the file to read, where electrolytes are saved
    :return: None
    """
    res = pd.read_csv(filename, sep='\t', index_col=0)
    electrolytes = list(res.Electrolytes)

    for entity in entity_list:
        for key in electrolytes:
            try:
                del entity[key]
            except KeyError:  # if key doesn't exist in the dictionary just continue
                pass


if __name__ == "__main__":
    entity_symbols = ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'a', 'k', 'b', 'b', 'l', 'd', 'd']
    tirp_symbols = ['a', 'b', 'd', 'b', 'b', 'd']

    # print(check_symbols_lexicographically(entity_symbols, tirp_symbols))
    # print(vertical_support_symbol(entity_list, 'C', 0.1))
