"""
Implementation of KarmaLego algorithm based on the article: 
Robert Moskovitch, Yuval Shahar: 
Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining

Link: https://pdfs.semanticscholar.org/a800/83f16631756d0865e13f679c2d5084df03ae.pdf
"""

from entities import entity_list
from help_functions import *


class TIRP:
    """
    Representation of Time Interval Relation Pattern (TIRP) with two lists.
    Implementation of basic methods to work with TIRPs.
    """

    def __init__(self, symbols=None, relations=None, k=1, vertical_support=None,
                 indices_supporting=None, parent_indices_supporting=None):
        """
        Initializes TIRP instance with default or given values.

        :param symbols: list of symbols presenting entity in lexicographic order (labels for upper triangular matrix)
        :param relations: list of Allen's temporal relations, presenting upper triangular matrix (half matrix),
                          relations' order is by columns from left to right and from up to down in the half matrix
        :param k: level of the TIRP in the enumeration tree
        :param vertical_support: value of TIRP support (between 0-1)
        :param indices_supporting: list of indices of entity list that support this TIRP
        :param parent_indices_supporting: list of indices of entity list that support parent of this TIRP
        """
        self.symbols = [] if symbols is None else symbols
        self.relations = [] if relations is None else relations
        self.k = k
        self.epsilon = 0    # maximum amount of time between two events that we consider it as the same time
        self.max_distance = 100    # maximum distance between two time intervals that first still influences the second
        self.vertical_support = vertical_support
        self.entity_indices_supporting = None if indices_supporting is None else indices_supporting
        self.parent_entity_indices_supporting = None if parent_indices_supporting is None else parent_indices_supporting

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

        :return: None
        """
        if len(self.relations) == 0:
            return

        print('\n  ‖', '   '.join(self.symbols[1:]))
        print('=' * (4 * len(self.symbols) - 1))

        start_index = 0
        increment = 2
        for row_id in range(len(self.symbols) - 1):
            print(self.symbols[row_id], '‖ ', end='')

            row_increment = row_id + 1
            index = start_index
            for column_id in range(len(self.symbols) - 1):
                if column_id < row_id:      # print spaces
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
        print('\n')

    def is_above_vertical_support(self, entity_list, min_ver_supp):
        """
        Check if this TIRP is present in at least min_ver_supp proportion of entities.

        :param entity_list: list of all entities
        :param min_ver_supp: proportion (value between 0-1) defining threshold for accepting TIRP
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
                matching_indices = check_symbols_lexicographically(entity_symbols, self.symbols)
                if matching_indices is not None:     # lexicographic match found, check relations in last column of TIRP
                    *entity_symbols_ti, last_symbol_ti = list(np.array(entity_ti)[matching_indices])
                    last_column_relations = self.relations[-(len(self.symbols) - 1):]
                    relations_match = True

                    for rel, symbol_ti in zip(last_column_relations, entity_symbols_ti):
                        if rel != temporal_relations(symbol_ti, last_symbol_ti, self.epsilon, self.max_distance):
                            relations_match = False
                            break

                    if relations_match:
                        supporting_indices.append(index)

        self.vertical_support = len(supporting_indices) / len(entity_list)

        if self.parent_entity_indices_supporting is not None:
            self.entity_indices_supporting = list(np.array(self.parent_entity_indices_supporting)[supporting_indices])
        else:
            self.entity_indices_supporting = supporting_indices

        return self.vertical_support >= min_ver_supp



class Karma:
    """
    Implementation of Karma part of KarmaLego algorithm.
    """

    def __init__(self):
        pass


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

    tirp = TIRP()
    # tirp.symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # tirp.relations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']

    # e1
    tirp.symbols = ['A', 'B', 'C', 'B', 'D']
    tirp.relations = ['o', 'm', 'o', '<', '<', 'o', '<', '<', '<', '<']

    # e2
    # tirp.symbols = ['B', 'C', 'A', 'B', 'C', 'D']
    # tirp.relations = ['m', '<', 'o', '<', '<', '<', '<', '<', '<', '=', '<', '<', '<', '=', '=']

    tirp.print()

    print(tirp.is_above_vertical_support(entity_list, 0.1))
    print(tirp.vertical_support)
