import scrapy
import pickle
import matplotlib.pyplot as plt
from collections import Counter


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write2file(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)


def read_lines_of_file(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


def prepare_links():
    diseases = load_pickle('diagnoses_clustered.pickle')[0]   # change index 0-3 (to get diseases for each cluster)
    return ['http://icd9.chrisendres.com/index.php?srchtype=diseases&srchtext=' + dis + '&Submit=Search&action=search'
            for dis in diseases]


def show_statistics():
    filenames = ['icd_%s.txt' % i for i in range(4)]
    plt.figure()
    for i, f_name in enumerate(filenames):
        lines = read_lines_of_file(f_name)
        first_digits = Counter(map(lambda x: x[0], filter(lambda s: s[0].isnumeric(), lines)))

        plt.subplot(2, 2, i + 1)
        plt.bar(list(range(10)), [first_digits[str(i)] for i in range(10)], tick_label=list(map(str, range(10))))
        plt.xlabel('first digit of disease in ICD9')
        plt.ylabel('number of occurrences')
        plt.title('Cluster #' + str(i))
    plt.show()


class ICD9Spider(scrapy.Spider):
    name = "icd9_spider"
    start_urls = prepare_links()

    counter = 0
    last_diagnosis_id = len(start_urls) - 1
    ICDs = []   # ! the order is not the same as for diagnoses ! (because parse is a callback function)

    def parse(self, response):
        diseases_list = response.css('.dlvl')
        if len(diseases_list) > 0:
            first_hit = diseases_list[0].css('div ::text').get()
            icd9 = first_hit.split(' ')[0]
        else:
            icd9 = '/'

        self.ICDs.append(icd9)

        if self.counter == self.last_diagnosis_id:
            pass
            # write2file('icd_2.txt', self.ICDs)   # change filename index number 0-3 (to get diseases for each cluster)

        self.counter += 1


if __name__ == '__main__':
    show_statistics()
