import random
from itertools import chain, combinations
from functools import reduce


# Generates n queries with num_predicates amount of predicates
# A is given as a quick way to know which classes there are
def generate_queries(num_predicates, n, A):
    classes = A.keys()
    queries = []
    for i in range(n):
        pred_left = num_predicates
        query = []
        while pred_left > 0:
            sub_predicate_size = random.randint(1, min(3, pred_left))
            sub_predicate = random.sample(classes, sub_predicate_size)
            query.append(sub_predicate)
            pred_left -= sub_predicate_size
        queries.append(query)
    return queries


# Parses query into list of lists
# WARNING: destructive operation, run detect_NF first to know what type of query this is
def parse_query(string):
    query_list = []
    temp = []
    for character in string.split(' '):
        if character == '(':
            temp = []
        elif character == ')':
            query_list.append(temp)
        elif character == '&' or character == '|':
            pass
        else:
            temp.append(character)
    return query_list


# Detects whether query is DNF or CNF (does not check if it is in proper NF in the first place)
def detect_NF(query):
    if ') | (' in query:
        return "DNF"
    elif ') & (' in query:
        return "CNF"
    else:
        return "Query not in proper form"


# returns the powerset
def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


# creates a range while keeping the original structure of the query
# eg [[0, 1], [2], [3, 4, 5], [6]]
def terror_list(query):
    return reduce(lambda a, b: a + [list(range(a[-1][-1]+1, 1+len(b)+a[-1][-1]))],
           query[1:], [list(range(len(query[0])))])

def flat_list(lst):
    return [item for sublist in lst for item in sublist]