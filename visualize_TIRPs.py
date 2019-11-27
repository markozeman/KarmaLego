from KarmaLego import *


if __name__ == "__main__":
    epsilon = 0
    max_distance = 100

    # possible options of use: 'artificial', 'pneumonia', '10%', 'all'
    use = 'pneumonia'

    tree_filename = ''
    entity_list = []
    if use == 'artificial':
        tree_filename = 'data/artificial_entities_tree.pickle'
        from entities import entity_list  # use artificial entities from entities.py
    elif use == 'pneumonia':
        tree_filename = 'data/pneumonia_tree_without_electrolytes_min_supp_0_05.pickle'    # without electrolytes
        # tree_filename = 'data/pneumonia_tree.pickle'
        entity_list = read_json('data/pneumonia_entity_list.json')
    elif use == '10%':
        # use 10% of all admissions data
        tree_filename = 'data/10percent_all_admissions_tree_without_electrolytes_min_supp_0_1.pickle'   # without electrolytes
        # tree_filename = 'data/10percent_all_admissions_tree.pickle'
        entity_list = read_json('data/10percent_all_admissions_entity_list.json')
    elif use == 'all':
        # all data (calculations for each TIRP take some seconds)
        tree_filename = 'data/tree.pickle'
        entity_list = read_json('data/entity_list.json')

    tree = load_pickle(tree_filename)

    # pickle with supports: [0.3, 0.3, 0.1, 0.3]
    # number of patients: [287, 135, 2813, 558]
    # number of TIRPs: [12, 68, 22, 233]
    ### use next 5 lines to visualize cluster TIRPs; comment this code later
    # tree_filename = 'data/cluster_trees_min_supp_0_3_and_0_1.pickle'
    # tree = load_pickle(tree_filename)
    #
    # idex = 1    # index of cluster to visualize
    # entity_list = list(np.array(entity_list)[tree[idex][0]])
    # tree = tree[idex][1]

    if tree_filename:
        visualize_tirps_from_file(tree, entity_list, epsilon, max_distance)
