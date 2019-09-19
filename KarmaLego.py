"""
Implementation of KarmaLego algorithm based on the article: 
Robert Moskovitch, Yuval Shahar: 
Temporal Patterns Discovery from Multivariate Time Series via Temporal Abstraction and Time-Interval Mining

Link: https://pdfs.semanticscholar.org/a800/83f16631756d0865e13f679c2d5084df03ae.pdf
"""
from transition_table import *
from entities import entity_list
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

    def find_tree_nodes(self, all_nodes):
        """
        Recursive method for finding tree nodes of TIRPs in pre-order Depth First Search.

        :param all_nodes: list of nodes (empty at start)
        :return: list of all TIRP nodes that are below self in a tree structure (including self)
        """
        if isinstance(self.data, TIRP):
            all_nodes.append(self.data)
        for child in self.children:
            child.find_tree_nodes(all_nodes)
        return all_nodes

    def print(self):
        """
        Print all nodes below and including self, sorted by TIRP vertical support in decreasing order

        :return: None
        """
        all_nodes = self.find_tree_nodes([])
        for node in sorted(all_nodes, reverse=True):
            print(node, end='')
        print('\n\nAll TIRP nodes: ', len(all_nodes))


class TIRP:
    """
    Representation of Time Interval Relation Pattern (TIRP) with two lists.
    Implementation of basic methods to work with TIRPs.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp, symbols=None, relations=None, k=1, vertical_support=None,
                 indices_supporting=None, parent_indices_supporting=None, indices_of_last_symbol_in_entities=None):
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
        :param indices_of_last_symbol_in_entities: list of indices of last element in symbols list in lexicographically ordered entities
                                                   (len(indices_of_last_symbol_in_entities) = len(indices_supporting))
        """
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp
        self.symbols = [] if symbols is None else symbols
        self.relations = [] if relations is None else relations
        self.k = k
        self.vertical_support = vertical_support
        self.entity_indices_supporting = indices_supporting
        self.parent_entity_indices_supporting = parent_indices_supporting
        self.indices_of_last_symbol_in_entities = [] if indices_of_last_symbol_in_entities is None else indices_of_last_symbol_in_entities

    def __repr__(self):
        """
        Method defining how TIRP class instance is printed to standard output.

        :return: string that is printed
        """
        return self.print() + '\nVertical support: ' + str(self.vertical_support) + '\n'

    def __lt__(self, other):
        """
        Method defining how 2 TIRPs are compared when sorting list of TIRPs.

        :param other: the second TIRP that is compared to self
        :return: boolean - True if self is less than other (in sense of their vertical support)
        """
        return self.vertical_support < other.vertical_support

    def __eq__(self, other):
        """
        Method defining equality of 2 TIRPs.

        :param other: the second TIRP that is compared to self
        :return: boolean - True if TIRPS are equal, False otherwise
        """
        return are_TIRPs_equal(self, other)

    def __hash__(self):
        return hash((sum([(i + 1) * hash(s) for i, s in enumerate(self.symbols)]), (sum([(i + 1) * hash(s) for i, s in enumerate(self.relations)]))))

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
            raise AttributeError

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
        Set some parameters of self instance.

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
                if matching_indices is not None and matching_indices != [None]:     # lexicographic match found, check all relations of TIRP
                    for matching_option in matching_indices:
                        all_relations_match = True

                        relation_index = 0
                        for column_count, entity_index in enumerate(matching_option[1:]):
                            for row_count in range(column_count + 1):
                                ti_1 = entity_ti[matching_option[row_count]]
                                ti_2 = entity_ti[entity_index]

                                if self.relations[relation_index] != temporal_relations(ti_1, ti_2, self.epsilon, self.max_distance):
                                    all_relations_match = False
                                    break

                                relation_index += 1

                            if not all_relations_match:
                                break

                        if all_relations_match:
                            supporting_indices.append(index)
                            self.indices_of_last_symbol_in_entities.append(list(matching_option)[-1])

        if self.parent_entity_indices_supporting is not None:
            self.entity_indices_supporting = list(np.array(self.parent_entity_indices_supporting)[supporting_indices])
        else:
            self.entity_indices_supporting = supporting_indices

        self.vertical_support = len(list(set(self.entity_indices_supporting))) / len(entity_list)

        # make 2 lists the same size by uniqueness of entity_indices_supporting
        if len(self.entity_indices_supporting) != 0:
            sym_index_ent_index_zipped_unique = list(set(list(zip(self.indices_of_last_symbol_in_entities, self.entity_indices_supporting))))
            self.indices_of_last_symbol_in_entities = list(np.array(sym_index_ent_index_zipped_unique)[:, 0])
            self.entity_indices_supporting = list(np.array(sym_index_ent_index_zipped_unique)[:, 1])

        return self.vertical_support >= self.min_ver_supp


class KarmaLego:
    """
    Implementation of KarmaLego algorithm.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp):
        """
        Initialize KarmaLego instance and set needed parameters.

        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: proportion (value between 0-1) defining threshold for accepting TIRP
        :param min_ver_supp: maximum distance between 2 time intervals that means first one still influences the second
        """
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.min_ver_supp = min_ver_supp

    def run(self):
        """
        Run KarmaLego algorithm.

        :return: tree of all frequent TIRPs
        """
        karma = Karma(self.epsilon, self.max_distance, self.min_ver_supp)
        tree = karma.run()

        lego = Lego(tree, self.epsilon, self.max_distance, self.min_ver_supp)
        return lego.run_lego(tree)


class Karma(KarmaLego):
    """
    Implementation of Karma part of KarmaLego algorithm.
    """

    def __init__(self, epsilon, max_distance, min_ver_supp):
        """
        Initialize Karma instance and set needed parameters.

        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: proportion (value between 0-1) defining threshold for accepting TIRP
        :param min_ver_supp: maximum distance between 2 time intervals that means first one still influences the second
        """
        super().__init__(epsilon, max_distance, min_ver_supp)

    def run(self):
        """
        Run Karma part of algorithm.

        :return: tree of up to 2-sized frequent TIRPs
        """
        all_symbols = list(set(sum([list(entity.keys()) for entity in entity_list], [])))

        frequent_symbols = []
        tree = TreeNode('root')
        for symbol in all_symbols:
            # min_ver_supp is in this case the same for accepting symbols and TIRPs
            has_enough_support, _ = vertical_support_symbol(entity_list, symbol, self.min_ver_supp)
            if has_enough_support:
                tree.add_child(TreeNode(symbol))
                frequent_symbols.append(symbol)

        all_TIRPs_k2 = []
        for entity_index, entity in enumerate(entity_list):
            ordered_ti = lexicographic_sorting(entity)

            # iterate through all ordered pairs
            for i in range(len(ordered_ti)):
                for j in range(i + 1, len(ordered_ti)):
                    start_1, end_1, symbol_1 = ordered_ti[i]
                    start_2, end_2, symbol_2 = ordered_ti[j]

                    # check if both symbols are frequent
                    if symbol_1 in frequent_symbols and symbol_2 in frequent_symbols:

                        # check temporal relation between 2 time intervals
                        temporal_relation = temporal_relations((start_1, end_1), (start_2, end_2), self.epsilon, self.max_distance)

                        # make a TIRP, save it in list and count occurrences of it through loops
                        tirp = TIRP(self.epsilon, self.max_distance, self.min_ver_supp, symbols=[symbol_1, symbol_2],
                                    relations=[temporal_relation], k=2, indices_supporting=[entity_index],
                                    indices_of_last_symbol_in_entities=[j])

                        same_tirp_exist = False
                        for t in all_TIRPs_k2:
                            if are_TIRPs_equal(t, tirp):
                                same_tirp_exist = True
                                t.entity_indices_supporting.append(entity_index)
                                t.indices_of_last_symbol_in_entities.append(j)
                                break
                        if not same_tirp_exist:
                            all_TIRPs_k2.append(tirp)

        for tirp in all_TIRPs_k2:
            if len(tirp.entity_indices_supporting) != 0:
                # make 2 lists the same size by uniqueness of entity_indices_supporting
                sym_index_ent_index_zipped_unique = list(set(list(zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting))))
                tirp.indices_of_last_symbol_in_entities = list(np.array(sym_index_ent_index_zipped_unique)[:, 0])
                tirp.entity_indices_supporting = list(np.array(sym_index_ent_index_zipped_unique)[:, 1])

            # save vertical support to TIRP instances
            tirp.vertical_support = len(list(set(tirp.entity_indices_supporting))) / len(entity_list)

            # prune TIRPs that don't have at least min_ver_supp: assign TIRPs with enough support to the tree
            if tirp.vertical_support >= self.min_ver_supp:
                for child_k1 in tree.children:
                    if child_k1.data == tirp.symbols[0]:    # tree should grow from this node symbol at K=1
                        child_k1.add_child(TreeNode(tirp))
                        break

        return tree


class Lego(KarmaLego):
    """
    Implementation of Lego part of KarmaLego algorithm.
    """

    def __init__(self, tree, epsilon, max_distance, min_ver_supp):
        """
        Initialize Lego instance and set needed parameters.

        :param tree: tree structure that is an output of Karma part
        :param epsilon: maximum amount of time between two events that we consider it as the same time
        :param max_distance: proportion (value between 0-1) defining threshold for accepting TIRP
        :param min_ver_supp: maximum distance between 2 time intervals that means first one still influences the second
        """
        self.tree = tree
        super().__init__(epsilon, max_distance, min_ver_supp)

    def run_lego(self, node):
        """
        Run Lego part of algorithm (recursive method).

        :param node: current node of a tree (root node given as a start)
        :return: tree of all frequent TIRPs
        """
        if isinstance(node.data, TIRP):     # True when K > 1
            # node.print()

            # find all possible extensions of current TIRP node
            all_extensions = list(set(self.all_extensions(node.data)))

            # for each extension check if it's above min_ver_supp
            ok_extensions = list(filter(lambda extension: extension.is_above_vertical_support(entity_list), all_extensions))

            for ext in ok_extensions:
                # add extended TIRP 'ext' to the current node children
                node.add_child(TreeNode(ext))

        for child in node.children:
            self.run_lego(child)

        return node

    def all_extensions(self, tirp):
        """
        Find all possible extensions of the current TIRP node.

        :param tirp: current TIRP node
        :return: list of all possible extended TIRPs
        """
        curr_num_of_symbols = len(tirp.symbols)
        all_possible_TIRPs = []
        for sym_index, ent_index in zip(tirp.indices_of_last_symbol_in_entities, tirp.entity_indices_supporting):
            lexi_ordered_entity = lexicographic_sorting(entity_list[ent_index])

            if curr_num_of_symbols < len(lexi_ordered_entity):
                for after_sym_index_ti in lexi_ordered_entity[sym_index + 1:]:
                    *new_ti, new_symbol = after_sym_index_ti

                    rel_between_last_2 = temporal_relations(lexi_ordered_entity[sym_index][:2], new_ti,
                                                            self.epsilon, self.max_distance)

                    curr_rel_index = len(tirp.relations) - 1
                    decrement_index = curr_num_of_symbols - 1

                    all_paths = find_all_possible_extensions([], [], rel_between_last_2, curr_rel_index, decrement_index, tirp.relations)

                    for path in all_paths:
                        new_relations = [rel_between_last_2, *path]
                        new_relations.reverse()

                        tirp_copy = deepcopy(tirp)
                        tirp_copy.extend(new_symbol, new_relations)

                        # set needed parameters for new TIRP tirp_copy
                        tirp_copy.k = tirp.k + 1
                        tirp_copy.vertical_support = None     # don't know yet
                        tirp_copy.parent_entity_indices_supporting = tirp.entity_indices_supporting
                        tirp_copy.entity_indices_supporting = []      # don't know yet
                        tirp_copy.indices_of_last_symbol_in_entities = []       # don't know yet

                        all_possible_TIRPs.append(tirp_copy)

        return all_possible_TIRPs


if __name__ == "__main__":
    # entity = entity_list[1]
    # plot_entity(entity)

    epsilon = 0
    max_distance = 100
    min_ver_supp = 0.1

    # tirp = TIRP(epsilon, max_distance, min_ver_supp)
    # tirp.symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # tirp.relations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']

    # e1
    # tirp.symbols = ['A', 'B', 'C', 'B', 'D']
    # tirp.relations = ['o', 'm', 'o', '<', '<', 'o', '<', '<', '<', '<']

    # e2
    # tirp.symbols = ['B', 'C', 'A', 'B', 'C', 'D']
    # tirp.relations = ['m', '<', 'o', '<', '<', '<', '<', '<', '<', '=', '<', '<', '<', '=', '=']

    # tirp.print()

    # print(tirp.is_above_vertical_support(entity_list))
    # print(tirp.vertical_support)

    tree = KarmaLego(epsilon, max_distance, min_ver_supp).run()
    tree.print()


    # todo
    # test on small dataset if number of TIRPs is 82/83
    # test some TIRPS on small and big entity_list
    # comment __hash__ method in TIRP
    # fix plot_entity in help_functions.py to be more general

