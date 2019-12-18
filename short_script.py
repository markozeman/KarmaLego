import pandas as pd
from collections import defaultdict
from help_functions import *
from entities import entity_list_4


if __name__ == "__main__":
    '''
    # read another pickle
    trees = load_pickle('data/pickle/cluster_trees_min_supp_0_3.pickle')
    print(trees)
    print('\n\n\n')

    # create new pickle with supports [0.3, 0.3, 0.1, 0.3]
    new_trees = trees
    new_indices, new_tree = load_pickle('data/pickle/cluster_trees_min_supp_0_1.pickle')[0]
    new_trees[2] = (new_indices, new_tree)
    print(new_trees)

    # save_pickle('data/pickle/cluster_trees_min_supp_0_3_and_0_1.pickle', new_trees)
    '''

    # check diagnoses for each of 4 clusters (hierarchical clustering with 10% of data)
    diagnoses_clustered = load_pickle('data/pickle/diagnoses_clustered.pickle')
    for id, diag_cl in enumerate(diagnoses_clustered):
        print(str(len(diag_cl)) + ' diagnoses from cluster #' + str(id))
        print(diag_cl, '\n')

    # check indices supporting and KarmaLego trees for each of 4 clusters (hierarchical clustering with 10% of all data)
    cluster_trees = load_pickle('data/pickle/cluster_trees_min_supp_0_3_and_0_1.pickle')
    print('\n\n\n\n\n', cluster_trees)

