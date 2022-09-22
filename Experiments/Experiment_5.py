import ast
from ExecuteQueryPlan import *
import pandas as pd
from data_loader import *
import time

# filename = "NLP_modelDB_f1_score.csv"
# sel_filename = "NLP_selectivity.csv"
# A, C, mem, sel = data_loader(filename, sel_filename)

data = pd.read_csv("train_extend.csv")

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


def filter_data(query, NF, data):
    if NF == "DNF":
        group_dfs = []
        for group in query:
            sub_group_df = (data[group[0]] == 1)
            for pred in group[1:]:
                sub_group_df = sub_group_df & (data[pred] == 1)
            group_dfs.append(sub_group_df)
        res = group_dfs[0]
        for group in group_dfs[1:]:
            res = res | group
        return res
    else:
        group_dfs = []
        for group in query:
            sub_group_df = data[group[0]]==1
            for pred in group[1:]:
                sub_group_df = sub_group_df | (data[pred] == 1)
            group_dfs.append(sub_group_df)
        res = group_dfs[0]
        for group in group_dfs[1:]:
            res = res & group
        return res

queries = [
    ([['toxic', 'severe_toxic'], ['negative', 'neutral']], "CNF"),
    ([['insult', 'negative'], ['obscene', 'positive']], "DNF"),
    ([['insult', 'negative'], ['threat', 'neutral'], ['identity_hate', 'negative']], "DNF"),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['negative']], 'CNF'),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['positive', 'neutral']], "CNF")
]

evals = []
times = []
print(assignments)
print(orderings)
trials = 10
for num, (query, NF) in enumerate(queries):
    times.append([])
    for trial in range(trials):
        start = time.time()
        (eval, classification) = execute_query_plan(query, NF, data['comment_text'], assignments[num], orderings[num])
        end = time.time()
        times[-1].append((end-start)*1000)
    evals.append(eval)

print(evals)
print(times)
with open("evals.txt", 'w') as fp:
    for item in evals:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("times.txt", 'w') as fp:
    for item in times:
        # write each item on a new line
        fp.write("%s\n" % item)