"""
List of dictionaries, each representing one entity (e.g. patient).
Keys of each entity are symbols and value of specific symbol is a list of tuples,
where each tuple represents one time interval of this symbol.
"""


entity_list = [
    {
        'A': [(2, 8)],
        'B': [(6, 10), (12, 16)],
        'C': [(8, 14)],
        'D': [(17, 20)]
    },
    {
        'A': [(8, 12)],
        'B': [(3, 7), (14, 17)],
        'C': [(7, 10), (14, 17)],
        'D': [(14, 17)]
    }
]

'''
    {
        'A': [(4, 14)],
        'B': [(12, 16)],
        'C': [(2, 4), (8, 10), (18, 20)],
        'D': [(5, 10)],
        'E': [(1, 4), (11, 14), (17, 19)],
        'F': [(7, 8), (14, 15)],
        'G': [(2, 8), (13, 18)]
    }
'''

