"""
Implementation of KarmaLego algorithm based on the article: 
Robert Moskovitch, Yuval Shahar: 
Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining

Link: https://pdfs.semanticscholar.org/a800/83f16631756d0865e13f679c2d5084df03ae.pdf
"""

from entities import entity_list
from helpers import *


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
        self.vertical_support = vertical_support
        self.entity_indices_supporting = indices_supporting
        self.parent_entity_indices_supporting = parent_indices_supporting

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

    def vertical_support_TIRP(self, entity_list, TIRP, min_ver_supp):
        """
        Check if TIRP is present in at least min_ver_supp proportion of entities.

        :param entity_list: list of all entities
        :param TIRP: time interval relation pattern
        :param min_ver_supp: proportion (value between 0-1) defining threshold for accepting TIRP
        :return: boolean - True if given TIRP has at least min_ver_supp support, otherwise False
        """
        # check only entities from entity list that supported parent (smaller) TIRP
        entity_list_reduced = list(np.array(entity_list)[self.parent_entity_indices_supporting])

        # stej ustrezne indexe
        for index, entity in enumerate(entity_list_reduced):
            # preglej če ima entity leksikografsko urejene simbole od TIRPa in če relacije v zadnjem stolpcu ustrezajo
            # če ustrezajo shrani index
            pass


        # set integer value of TIRP vertical support


        # set entity list indices supporting



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

    tirp.symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    tirp.relations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']
    tirp.print()



