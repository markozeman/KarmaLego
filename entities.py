"""
List of dictionaries, each representing one entity (e.g. patient).
Keys of each entity are symbols and value of specific symbol is a list of tuples,
where each tuple represents one time interval of this symbol.
"""

# 4 entities
entity_list_4 = [
    {
        'A': [(2, 6), (12, 16)],
        'B': [(4, 13)],
        'C': [(4, 9), (12, 16)],
        'D': [(6, 19)],
        'E': [(8, 11), (14, 19)]
    },
    {
        'A': [(4, 8)],
        'B': [(2, 6)],
        'C': [(7, 14), (16, 19)],
        'D': [(5, 11)],
        'E': [(9, 16)]
    },
    {
        'A': [(3, 8)],
        'B': [(6, 10)],
        'C': [(6, 10), (12, 15)],
        'E': [(3, 12), (15, 18)]
    },
    {
        'B': [(3, 8), (12, 17)],
        'C': [(5, 10)],
        'D': [(5, 10), (14, 19)]
    }
]


# 2 entities
entity_list_2 = [
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


# 1 entity
entity_list_1 = [
    {
        'A': [(4, 14)],
        'B': [(12, 16)],
        'C': [(2, 4), (8, 10), (18, 20)],
        'D': [(5, 10)],
        'E': [(1, 4), (11, 14), (17, 19)],
        'F': [(7, 8), (14, 15)],
        'G': [(2, 8), (13, 18)]
    }
]


entity_list = entity_list_4

