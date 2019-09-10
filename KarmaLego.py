"""
Implementation of KarmaLego algorithm based on the article: 
Robert Moskovitch, Yuval Shahar: 
Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining

Link: https://pdfs.semanticscholar.org/a800/83f16631756d0865e13f679c2d5084df03ae.pdf
"""

from entities import entity_list
from copy import deepcopy
from help_functions import *


class TreeNode:
    """
    Implementation of tree structure of nodes representing enumerated tree of TIRPs in KarmaLego.
    """

    def __init__(self, data=None):
        """
        Initialize TreeNode instance with given data and set its children to empty list.

        :param data: data for a given TreeNode
        """
        self.data = data
        self.children = []

    def add_child(self, child):
        """
        Add child TreeNode to the children list.

        :param child: child node to append
        :return: None
        """
        self.children.append(child)

    def print_tree(self):
        """
        Recursive method for printing tree of TIRPs in pre-order Depth First Search.

        :return: None
        """
        if isinstance(self.data, TIRP):
            print(self.data)
        for child in self.children:
            child.print_tree()


class TIRP:
    """
    Representation of Time Interval Relation Pattern (TIRP) with two lists.
    Implementation of basic methods to work with TIRPs.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp, symbols=None, relations=None, k=1, vertical_support=None,
                 indices_supporting=None, parent_indices_supporting=None):
        """
        Initialize TIRP instance with default or given values.

        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: proportion (value between 0-1) defining threshold for accepting TIRP
        :param min_ver_supp: maximum distance between two time intervals that means first one still influences the second
        :param symbols: list of symbols presenting entity in lexicographic order (labels for upper triangular matrix)
        :param relations: list of Allen's temporal relations, presenting upper triangular matrix (half matrix),
                          relations' order is by columns from left to right and from up to down in the half matrix
        :param k: level of the TIRP in the enumeration tree
        :param vertical_support: value of TIRP support (between 0-1)
        :param indices_supporting: list of indices of entity list that support this TIRP
        :param parent_indices_supporting: list of indices of entity list that support parent of this TIRP
        """
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp
        self.symbols = [] if symbols is None else symbols
        self.relations = [] if relations is None else relations
        self.k = k
        self.vertical_support = vertical_support
        self.entity_indices_supporting = None if indices_supporting is None else indices_supporting
        self.parent_entity_indices_supporting = None if parent_indices_supporting is None else parent_indices_supporting

    def __repr__(self):
        return self.print() + '\nVertical support: ' + str(self.vertical_support)

    def extend(self, new_symbol, new_relations):
        """
        Extend TIRP with a new symbol and new relations. Check if sizes of lists are ok after extending.

        :param new_symbol: string representing new symbol to add
        :param new_relations: list of new relations to add
        :return: None
        """
        self.symbols.append(new_symbol)
        self.relations.extend(new_relations)
        if not self.check_size():
            print('Extension of TIRP is wrong!')

    def check_size(self):
        """
        Check if length of list relations is right regarding length of list symbols.

        :return: boolean - if size of symbols and relations lists match
        """
        return (len(self.symbols) ** 2 - len(self.symbols)) / 2 == len(self.relations)

    def print(self):
        """
        Pretty print TIRP as upper triangular matrix.

        :return: empty string because __repr__ method is using print() method
        """
        if len(self.relations) == 0:
            return

        print('\n\n  ‖', '   '.join(self.symbols[1:]))
        print('=' * (4 * len(self.symbols) - 1))

        start_index = 0
        increment = 2
        for row_id in range(len(self.symbols) - 1):
            print(self.symbols[row_id], '‖ ', end='')

            row_increment = row_id + 1
            index = start_index
            for column_id in range(len(self.symbols) - 1):
                if column_id < row_id:    # print spaces
                    print('    ', end='')
                else:   # print relation
                    print(self.relations[index], end='   ')
                    index += row_increment
                    row_increment += 1

            start_index += increment
            increment += 1

            if row_id != len(self.symbols) - 2:
                print()
                print('-' * (4 * len(self.symbols) - 1))

        return ""

    def is_above_vertical_support(self, entity_list):
        """
        Check if this TIRP is present in at least min_ver_supp proportion of entities.

        :param entity_list: list of all entities
        :return: boolean - True if given TIRP has at least min_ver_supp support, otherwise False
        """
        if not self.check_size():
            print('TIRP symbols and relations lists do not have compatible size!')
            return None

        # check only entities from entity list that supported parent (smaller) TIRP
        if self.parent_entity_indices_supporting is not None:
            entity_list_reduced = list(np.array(entity_list)[self.parent_entity_indices_supporting])
        else:
            entity_list_reduced = entity_list

        supporting_indices = []
        for index, entity in enumerate(entity_list_reduced):
            lexi_sorted = lexicographic_sorting(entity)
            entity_ti = list(map(lambda s: (s[0], s[1]), lexi_sorted))
            entity_symbols = list(map(lambda s: s[2], lexi_sorted))
            if len(self.symbols) <= len(entity_symbols):
                matching_indices = check_symbols_lexicographically(entity_symbols, self.symbols, 'all')
                if matching_indices is not None and matching_indices != [None]:     # lexicographic match found, check relations in last column of TIRP
                    last_column_relations = self.relations[-(len(self.symbols) - 1):]

                    for matching_option in matching_indices:
                        *entity_symbols_ti, last_symbol_ti = list(np.array(entity_ti)[list(matching_option)])
                        relations_match = True

                        for rel, symbol_ti in zip(last_column_relations, entity_symbols_ti):
                            if rel != temporal_relations(symbol_ti, last_symbol_ti, self.epsilon, self.max_distance):
                                relations_match = False
                                break

                        if relations_match:
                            supporting_indices.append(index)

        supporting_indices = list(set(supporting_indices))     # make list unique because it can have more matches in one entity
        self.vertical_support = len(supporting_indices) / len(entity_list)

        if self.parent_entity_indices_supporting is not None:
            self.entity_indices_supporting = list(np.array(self.parent_entity_indices_supporting)[supporting_indices])
        else:
            self.entity_indices_supporting = supporting_indices

        return self.vertical_support >= self.min_ver_supp


class Karma:
    """
    Implementation of Karma part of KarmaLego algorithm.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp):
        """
        Initialize Karma instance and set needed parameters.

        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: proportion (value between 0-1) defining threshold for accepting TIRP
        :param min_ver_supp: maximum distance between two time intervals that means first one still influences the second
        """
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    def run(self):
        """
        Run Karma part of algorithm.

        :return: tree of up to 2-sized frequent TIRPs
        """
        all_symbols = list(set(sum([list(entity.keys()) for entity in entity_list], [])))

        tree = TreeNode('root')
        for symbol in all_symbols:
            tree.add_child(TreeNode(symbol))

        all_TIRPs_k2 = []
        for entity_index, entity in enumerate(entity_list):
            ordered_ti = lexicographic_sorting(entity)

            # iterate through all ordered pairs
            for i in range(len(ordered_ti)):
                for j in range(i + 1, len(ordered_ti)):
                    start_1, end_1, symbol_1 = ordered_ti[i]
                    start_2, end_2, symbol_2 = ordered_ti[j]

                    # check temporal relation between 2 time intervals
                    temporal_relation = temporal_relations((start_1, end_1), (start_2, end_2), self.epsilon, self.max_distance)

                    # make a TIRP, save it in list and count occurrences of it through loops
                    tirp = TIRP(self.epsilon, self.max_distance, self.min_ver_supp, symbols=[symbol_1, symbol_2],
                                relations=[temporal_relation], k=2, indices_supporting=[entity_index])

                    same_tirp_exist = False
                    for t in all_TIRPs_k2:
                        if are_TIRPs_equal(t, tirp):
                            same_tirp_exist = True
                            t.entity_indices_supporting.append(entity_index)
                            break
                    if not same_tirp_exist:
                        all_TIRPs_k2.append(tirp)

        for tirp in all_TIRPs_k2:
            # save vertical support to TIRP instances and
            tirp.entity_indices_supporting = list(set(tirp.entity_indices_supporting))
            tirp.vertical_support = len(tirp.entity_indices_supporting) / len(entity_list)

            # prune TIRPs that don't have at least min_ver_supp: assign TIRPs with enough support to the tree
            if tirp.vertical_support >= self.min_ver_supp:
                for child_k1 in tree.children:
                    if child_k1.data == tirp.symbols[0]:    # tree should grow from this node symbol at K=1
                        child_k1.add_child(TreeNode(tirp))
                        break

        return tree


class Lego:
    """
    Implementation of Lego part of KarmaLego algorithm.
    """

    def __init__(self):
        pass


class KarmaLego:
    """
    Implementation of KarmaLego algorithm.
    """

    def __init__(self):
        pass



if __name__ == "__main__":
    # entity = entity_list[0]
    # plot_entity(entity)

    epsilon = 0
    max_distance = 100
    min_ver_supp = 0.1

    tirp = TIRP(epsilon, max_distance, min_ver_supp)
    # tirp.symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # tirp.relations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']

    # e1
    tirp.symbols = ['A', 'B', 'C', 'B', 'D']
    tirp.relations = ['o', 'm', 'o', '<', '<', 'o', '<', '<', '<', '<']

    # e2
    # tirp.symbols = ['B', 'C', 'A', 'B', 'C', 'D']
    # tirp.relations = ['m', '<', 'o', '<', '<', '<', '<', '<', '<', '=', '<', '<', '<', '=', '=']

    # tirp.print()

    # print(tirp.is_above_vertical_support(entity_list))
    # print(tirp.vertical_support)

    karma = Karma(epsilon, max_distance, min_ver_supp)
    tree = karma.run()

    tree.print_tree()


    # todo
    # make a 2D list l = [[ver_supp_1, TIRP_1], [ver_supp_2, TIRP_2], [ver_supp_3, TIRP_3]]
    # sorted(l, key=lambda x: x[0], reverse=True)
    # print sorted list

    # make transition table

