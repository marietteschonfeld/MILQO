from Query_tools import generate_queries
from data_loader import data_loader
import ast


def query_generation(num_preds, num):
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    A, _, _, _ = data_loader(filename)
    queries = []
    for num_predicates in num_preds:
        queries.append(generate_queries(num_predicates, num, A))
    queries = [item for sublist in queries for item in sublist]
    with open("queries_8.txt", 'w') as fp:
        for item in queries:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')

num_preds = [8]
num = 3
query_generation(num_preds, num)

queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

