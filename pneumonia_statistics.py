from KarmaLego import *
from MIMIC_prescriptions2KarmaLego import time_difference
from help_functions import read_json, write2json
from clustering import prepare_matrix, visualize_clusters_in_2D


def length_of_stay(filename, show_plot=True):
    """
    Show histogram of admission length of stay for pneumonia patients.

    :param filename: name of csv file where pneumonia admissions data is saved
    :param show_plot: boolean that indicates whether plot should be shown
    :return: list of pneumonia patients length of stay
    """
    pneumonia_data = pd.read_csv(filename, sep='\t', index_col=0)
    patient_ids = sorted(list(set(pneumonia_data.patient_id)))

    length_of_stays = []
    for pat_id in patient_ids:
        admittance = pneumonia_data[pneumonia_data.patient_id == pat_id].admittime.iloc[0]
        discharge = pneumonia_data[pneumonia_data.patient_id == pat_id].dischtime.iloc[0]
        los = time_difference(admittance, [discharge], unit='day')[0]
        if los > 0:
            length_of_stays.append(los)

    if show_plot:
        plt.hist(length_of_stays, bins=80)
        plt.title('Pneumonia patients length of stay (MIMIC database)')
        plt.xlabel('length of stay (days)')
        plt.ylabel('number of occurrences')
        plt.show()

    return length_of_stays


def clusters_based_on_LoS():
    """
    Using pneumonia patients to show 4 clusters based on patient's length of stay.

    :return: None
    """
    algorithm = 'LoS'
    tree_filename = '../data/pickle/pneumonia_tree_without_electrolytes_min_supp_0_05.pickle'
    entity_list = read_json('../data/json/pneumonia_entity_list.json')
    mat = prepare_matrix(tree_filename, len(entity_list))

    length_of_stays = length_of_stay('../csv/pneumonia_admissions.csv', show_plot=False)
    borders = (7, 15, 30)
    groups = make_groups(length_of_stays, borders)

    labels = np.zeros(len(entity_list), dtype='int')
    labels[groups[0]] = 0
    labels[groups[1]] = 1
    labels[groups[2]] = 1
    labels[groups[3]] = 1

    visualize_clusters_in_2D(mat, labels, algorithm, None, show_annotations=False)


def drugs_according2groups(filename, max_drugs_plotted):
    """
    Show 4 histograms (one for each group) of the 'max_drugs_plotted' most common appearances of drugs in their TIRPs.

    :param filename: name of json file where counter data are saved
    :param max_drugs_plotted: maximum number of drugs that are plotted in one histogram
    :return: None
    """
    drug_appearances = read_json(filename)
    plot_indices = [1, 2, 5, 6]
    LoS = ['1-7 days', '8-15 days', '16-30 days', 'more than 30 days']

    plt.figure()
    for i in range(4):
        drugs = drug_appearances[str(i)]

        plt.subplot(3, 2, plot_indices[i])
        plt.bar(list(drugs.keys())[:max_drugs_plotted], list(drugs.values())[:max_drugs_plotted])
        plt.xticks(rotation=45)
        plt.ylabel('number of occurrences')
        plt.title('Most common drugs in cluster #%d --- %s' % (i, LoS[i]))
    plt.show()


def make_groups(length_of_stays, borders):
    """
    Make groups of pneumonia patients based on their length of stay.

    :param length_of_stays: list of pneumonia patients length of stay
    :param borders: tuple of integers representing borders for cutting patient entities based on LoS
    :return: list of arrays, each array will have indices of pneumonia patients that belong to that group regarding LoS
    """
    length_of_stays = np.array(length_of_stays)
    groups = [np.where(length_of_stays <= borders[0])[0]]   # first group
    for od, do in zip(borders, borders[1:]):
        groups.append(np.where(np.logical_and(od < length_of_stays, length_of_stays <= do))[0])
    groups.append(np.where(borders[len(borders) - 1] < length_of_stays)[0])     # last group
    return groups


def write_groups2file(filename_basis, groups, pneumonia_entity_list):
    """
    Write entity list groups of pneumonia patients in separate files.

    :param filename_basis: name of the file basis where each group will be saved
    :param groups: list of arrays, each array will have indices of pneumonia patients that belong to that group regarding LoS
    :param pneumonia_entity_list: entity list of patients with pneumonia diagnosis
    :return: None
    """
    for i, indices in enumerate(groups):
        group_entities = list(np.array(pneumonia_entity_list)[indices])
        write2json(filename_basis + str(i) + '.json', group_entities)


if __name__ == '__main__':
    length_of_stays = length_of_stay('../csv/pneumonia_admissions.csv')
    borders = (7, 15, 30)
    groups = make_groups(length_of_stays, borders)
    pneumonia_entity_list = read_json('../data/json/pneumonia_entity_list.json')
    # write_groups2file('../data/pickle/pneumonia_entity_list_group_', groups, pneumonia_entity_list)

    drugs_according2groups('../data/json/pneumonia_groups_drug_appearances.json', 8)

    clusters_based_on_LoS()
