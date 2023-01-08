import ast
from ExecuteQueryPlan import *
import pandas as pd
import time

from sklearn.model_selection import train_test_split

seed = 2022
data = pd.read_csv("train_extend.csv")
df = data.drop(['Unnamed: 0'], axis=1)
train, test = train_test_split(df, test_size=0.4, random_state=seed)
train = pd.DataFrame(train, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)

assignments = []
with open('assignments.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        assignments.append(ast.literal_eval(x))

orderings = []
with open('orderings.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        orderings.append(ast.literal_eval(x))

queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

evals = []
times = []
trials = 1
num = 0
for NF in ["CNF", "DNF"]:
    for num_query, query in enumerate(queries):
        print("Query {} out of {}".format(num, len(queries)))
        print(query)
        print(NF)
        print(assignments[num])
        print(orderings[num])
        times.append([])
        for trial in range(trials):
            start = time.time()
            (eval, classification) = execute_query_plan(query, NF, test['comment_text'], assignments[num], orderings[num])
            end = time.time()
            times[-1].append((end-start)*1000)
        evals.append(eval)
        num += 1


with open("evals.txt", 'w') as fp:
    for item in evals:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("times.txt", 'w') as fp:
    for item in times:
        # write each item on a new line
        fp.write("%s\n" % item)


greedy_assignments = []
with open('greedy_assignments.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        greedy_assignments.append(ast.literal_eval(x))

greedy_orderings = []
with open('greedy_orderings.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        greedy_orderings.append(ast.literal_eval(x))

greedy_evals = []
greedy_times = []
print("Greedy")
num = 0
for NF in ["CNF", "DNF"]:
    for num_query, query in enumerate(queries):
        print("Query {} out of {}".format(num, len(queries)))
        print(query)
        print(NF)
        print(assignments[num])
        print(orderings[num])
        greedy_times.append([])
        for trial in range(trials):
            start = time.time()
            (eval, classification) = execute_query_plan(query, NF, test['comment_text'], greedy_assignments[num], greedy_orderings[num])
            end = time.time()
            greedy_times[-1].append((end-start)*1000)
        greedy_evals.append(eval)
        num+=1


with open("greedy_evals.txt", 'w') as fp:
    for item in greedy_evals:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("greedy_times.txt", 'w') as fp:
    for item in greedy_times:
        # write each item on a new line
        fp.write("%s\n" % item)

