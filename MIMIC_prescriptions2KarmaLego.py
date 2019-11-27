import pandas as pd
import intervals as I
from help_functions import *
from functools import reduce
from collections import Counter
from dateutil.parser import parse as date_parser


def time_difference(start, end_list, unit='second'):
    """
    Calculate time differences between start and end_list items in specified unit.

    :param start: start datetime (type = pandas.core.series.Series)
    :param end_list: list of end datetimes (type = pandas.core.series.Series)
    :param unit: what is the unit of the specified time difference (default = 'second')
    :return: list of time differences from start to end_list given some unit
    """
    start = date_parser(start)
    end_list = list(map(lambda d: date_parser(d), end_list))
    l = []
    for end in end_list:
        timedelta = (end - start).total_seconds()
        factor = 1
        if unit == 'minute':
            factor = 60
        elif unit == 'hour':
            factor = 3600
        elif unit == 'day':
            factor = 3600 * 24
        elif unit == 'year':
            factor = 3600 * 24 * 365
        l.append(round(timedelta / factor, 1))
    return l


def remove_overlapping_intervals(ti_list):
    """
    Remove or combine overlapping intervals into list of non-overlapping intervals. Examples:
    [(0.0, 2.0), (0.0, 4.0)] ----> [(0, 4)]
    [(0.0, 2.0), (1.0, 2.0), (1.0, 6.0)] ----> [(0, 6)]
    [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0)] ----> [(0, 2)]

    :param ti_list: list of time intervals, each presented as tuple
    :return: list of time intervals without overlapping
    """
    ti_list = list(map(lambda ti: I.closed(*ti), ti_list))      # tuples converted to intervals
    new_intervals = reduce(lambda x1, x2: x1.union(x2), ti_list)     # non-overlapping intervals made with union
    return [(interval.lower, interval.upper) for interval in new_intervals]    # put intervals back to list of tuples


def prepare_prescriptions_data():
    """
    Prepare data to be suitable as input for KarmaLego algorithm.

    :return: entity list which is suitable for KarmaLego
    """
    results = pd.read_csv('csv/pneumonia_admissions.csv', sep='\t', index_col=0)   # results from database (e.g. prescription table)
    results = results.query('enddate >= startdate')   # remove rows where enddate is before startdate

    # patients = Counter(list(results.patient_id))
    # drugs = Counter(list(results.drug))
    # print(patients)
    # print(drugs)

    print('Number of all patients: ', len(results.groupby('patient_id').groups.items()))

    entity_list = []
    patient_IDs = []
    for patient_id, row_occurrences in results.groupby('patient_id').groups.items():
        patient_df = results.loc[row_occurrences]

        # patient's oldest startdate, datetime of the first prescription for the patient
        reference_timepoint = min(patient_df.startdate)

        ent = {}
        for drug, drugs_row_occurrences in patient_df.groupby('drug').groups.items():
            drug_df = results.loc[drugs_row_occurrences]

            for s_time, e_time in zip(time_difference(reference_timepoint, drug_df.startdate, unit='day'),
                                      time_difference(reference_timepoint, drug_df.enddate, unit='day')):
                ti = (s_time, e_time)
                if drug in ent:
                    ent[drug].append(ti)
                else:
                    ent[drug] = [ti]

            ent[drug] = remove_overlapping_intervals(ent[drug])

        entity_list.append(ent)
        patient_IDs.append(patient_id)

    # save_pickle('data/pickle/patient_IDs_prescriptions.pickle', patient_IDs)

    return entity_list


if __name__ == "__main__":
    pass

    entity_list = prepare_prescriptions_data()

    # write2json('all_admissions_entity_list.json', entity_list)
