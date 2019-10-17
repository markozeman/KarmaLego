from KarmaLego import *
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as HierarchicalClustering


def prepare_matrix(tree_filename, num_of_patients):
    """
    Prepare matrix of data where each row is a vector representing one patient.
    Shape: (number of patients, number of TIRPs)

    :param tree_filename: name of the file where tree of TIRPs is saved
    :param num_of_patients: number of patients in entity list, from which tree of TIRPs was constructed
    :return: 2D np.array
    """
    tree = load_pickle(tree_filename)
    all_nodes = tree.find_tree_nodes([])

    mat = np.zeros((num_of_patients, len(all_nodes)))
    for i, tirp in enumerate(all_nodes):
        tirp_size = len(tirp.symbols)
        mat[tirp.entity_indices_supporting, i] = 1 / tirp_size
    return mat


def hierarchical_clustering_dendrogram(mat, metric, linkage):
    """
    Run hierarchical clustering on data matrix mat. Show truncated dendrogram.

    :param mat: matrix of input data, shape: (number of patients, number of TIRPs)
    :param metric: distance metric to use
    :param linkage: type of linkage to use
    :return: None
    """
    Z = HierarchicalClustering(mat, method=linkage, metric=metric)
    dendrogram(Z, p=5, truncate_mode='level')
    plt.xlabel('index of patient in entity list')
    plt.ylabel('distance')
    plt.title('Trucated dendrogram using %s linkage and %s metric' % (linkage, metric))
    plt.show()


def hierarchical_clustering(mat, k, metric, linkage):
    """
    Run hierarchical clustering on data matrix mat and find k clusters using specified metric and linkage.

    :param mat: matrix of input data, shape: (number of patients, number of TIRPs)
    :param k: number of clusters
    :param metric: distance metric to use
    :param linkage: type of linkage to use
    :return: list of labels (len = number of patients), each label is a cluster that patient belongs to
    """
    return AgglomerativeClustering(n_clusters=k, affinity=metric, linkage=linkage).fit_predict(mat)


def k_means(mat, k):
    """
    Run k-means algorithm on data matrix mat and find k clusters.

    :param mat: matrix of input data, shape: (number of patients, number of TIRPs)
    :param k: number of clusters
    :return: list of labels (len = number of patients), each label is a cluster that patient belongs to
    """
    return KMeans(n_clusters=k, max_iter=1000).fit_predict(mat)


def pca(mat, num_of_components):
    """
    Run PCA decomposition on data matrix.

    :param mat: matrix of input data, shape: (number of patients, number of TIRPs)
    :param num_of_components: number of components wanted after PCA
    :return: 2D np.array of shape (number of patients, num_of_components)
    """
    return PCA(n_components=num_of_components).fit_transform(mat)


def visualize_clusters_in_2D(mat, labels, algorithm_name):
    """
    Visualize clusters after performing PCA to get 2D points.

    :param mat: matrix of input data, shape: (number of patients, number of TIRPs)
    :param labels: list of labels (len = number of patients), each label is a cluster that patient belongs to
    :param algorithm_name: name of the clustering algorithm
    :return: None
    """
    pca_mat = pca(mat, 2)
    plt.scatter(pca_mat[:, 0], pca_mat[:, 1], c=labels, s=5, cmap='rainbow')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Each point represents one patient after performing ' + algorithm_name)
    plt.show()


if __name__ == "__main__":
    use_MIMIC = True
    if use_MIMIC:
        tree_filename = 'data/pneumonia_tree.pickle'
        num_of_patients = len(read_json('data/pneumonia_entity_list.json'))
    else:
        tree_filename = 'data/artificial_entities_tree.pickle'
        num_of_patients = 4

    mat = prepare_matrix(tree_filename, num_of_patients)

    # choose clustering algorithm: 'hierarchical' or 'k-means'
    algorithm = 'hierarchical'
    k = 3

    if algorithm == 'k-means':
        labels = k_means(mat, k)
    elif algorithm == 'hierarchical':
        metric = 'euclidean'
        linkage = 'average'

        labels = hierarchical_clustering(mat, k, metric, linkage)
        hierarchical_clustering_dendrogram(mat, metric, linkage)

    print(Counter(labels))
    visualize_clusters_in_2D(mat, labels, algorithm)


