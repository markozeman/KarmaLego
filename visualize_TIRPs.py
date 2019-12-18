from KarmaLego import *


if __name__ == "__main__":
    epsilon = 0
    max_distance = 100

    # possible options of use: 'artificial', 'pneumonia', 'pneumonia groups', '10%', 'all'
    use = 'pneumonia groups'

    # possible options of order_by: 'vertical support', 'TIRP size'
    order_by = 'TIRP size'

    # possible_options of search_drug: every drug that is present in the tree
    search_drug = None
    # search_drug = 'Insulin'

    # possible_options of search_type: 'subtree', 'included'
    search_type = 'included'

    tree_filename = ''
    entity_list = []
    if use == 'artificial':
        tree_filename = 'data/pickle/artificial_entities_tree.pickle'
        from entities import entity_list  # use artificial entities from entities.py
    elif use == 'pneumonia':
        tree_filename = 'data/pickle/pneumonia_tree_without_electrolytes_min_supp_0_05.pickle'    # without electrolytes
        # tree_filename = 'data/pickle/pneumonia_tree.pickle'
        entity_list = read_json('data/json/pneumonia_entity_list.json')
    elif use == 'pneumonia groups':     # according to length of stay
        # 0: 0-7 days, 1: 8-15 days, 2: 16-30 days, 3: >30 days
        group_index = 0    # choose which group to show based on LoS
        tree_filename = 'data/pickle/pneumonia_tree_group_%d.pickle' % group_index
        entity_list = read_json('data/json/pneumonia_entity_list_group_%d.json' % group_index)
    elif use == '10%':
        # use 10% of all admissions data
        tree_filename = 'data/pickle/10percent_all_admissions_tree_without_electrolytes_min_supp_0_1.pickle'   # without electrolytes
        # tree_filename = 'data/pickle/10percent_all_admissions_tree.pickle'
        entity_list = read_json('data/json/10percent_all_admissions_entity_list.json')
    elif use == 'all':
        # all data (calculations for each TIRP take some seconds)
        tree_filename = 'data/pickle/tree.pickle'
        entity_list = read_json('data/json/entity_list.json')

    tree = load_pickle(tree_filename)

    # pickle with supports: [0.3, 0.3, 0.1, 0.3]
    # number of patients: [287, 135, 2813, 558]
    # number of TIRPs: [12, 68, 22, 233]
    ### use next 5 lines to visualize cluster TIRPs; comment this code later
    # tree_filename = 'data/pickle/cluster_trees_min_supp_0_3_and_0_1.pickle'
    # tree = load_pickle(tree_filename)
    #
    # idex = 3    # index of cluster to visualize
    # entity_list = list(np.array(entity_list)[tree[idex][0]])
    # tree = tree[idex][1]

    if tree_filename:
        visualize_tirps_from_file(tree, entity_list, epsilon, max_distance, order_by, search_drug, search_type)
