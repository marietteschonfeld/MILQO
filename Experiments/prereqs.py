from Query_tools import generate_queries
from data_loader import data_loader
import ast


def query_generation(lb, ub, stepsize, num):
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    A, _, _, _ = data_loader(filename)
    queries = []
    for num_predicates in range(lb, ub, stepsize):
        queries.append(generate_queries(num_predicates, num, A))
    queries = [item for sublist in queries for item in sublist]
    with open("queries.txt", 'w') as fp:
        for item in queries:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')

queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

stepsize = 2
lb, ub, num = 2, 17, 5
query_generation(lb, ub, stepsize, num)
