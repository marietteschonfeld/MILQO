import random

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