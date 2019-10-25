from KarmaLego import *


if __name__ == "__main__":
    epsilon = 0
    max_distance = 100

    # possible options of use: 'artificial', 'pneumonia', '10%', 'all'
    use = '10%'

    tree_filename = ''
    entity_list = []
    if use == 'artificial':
        tree_filename = 'data/artificial_entities_tree.pickle'
        from entities import entity_list  # use artificial entities from entities.py
    elif use == 'pneumonia':
        tree_filename = 'data/pneumonia_tree.pickle'
        entity_list = read_json('data/pneumonia_entity_list.json')
    elif use == '10%':
        # use 10% of all admissions data
        tree_filename = 'data/10percent_all_admissions_tree.pickle'
        entity_list = read_json('data/10percent_all_admissions_entity_list.json')
    elif use == 'all':
        # all data (calculations for each TIRP take some seconds)
        tree_filename = 'data/tree.pickle'
        entity_list = read_json('data/entity_list.json')

    if tree_filename:
        visualize_tirps_from_file(tree_filename, entity_list, epsilon, max_distance)
