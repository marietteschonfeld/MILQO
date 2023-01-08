import ast

import pandas as pd

from Models.MO_optimization import set_MOO_method
from Models.OrderOpt import OrderOpt
from Query_tools import generate_queries
from data_loader import *
from Models.ModelOpt import *
import time
import gurobipy as grb

# modelDB = "../../ssh_landing/NLP_modelDB.csv"
# modelDB = pd.read_csv(modelDB)
# DB_f1 = modelDB[modelDB.score_type=='f1_score']
# DB_f1 = DB_f1.drop(['score_type'], axis=1)
# DB_f1.to_csv("../../Data/NLP_modelDB_f1_score.csv")

filename = "../../Data/NLP_modelDB_f1_score.csv"
sel_filename = "../../Data/NLP_selectivity.csv"
A, C, mem, sel = data_loader(filename, sel_filename)

# def query_generation(num_preds, num):
#     queries = []
#     for num_predicates in num_preds:
#         queries.append(generate_queries(num_predicates, num, A))
#     queries = [item for sublist in queries for item in sublist]
#     with open("queries.txt", 'w') as fp:
#         for item in queries:
#             # write each item on a new line
#             fp.write("%s\n" % item)
#     print('Done')
#
# num_preds = [2,4,6,8]
# num = 5
# query_generation(num_preds, num)

queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

MOO_method = "archimedean_goal_method"
objectives = [['memory_norm', 'cost_norm', 'acc_loss_norm']]
goals = {"memory_norm": 0, "cost_norm": 0, "acc_loss_norm": 0}

new_eq = {'eq45': True, 'accuracy': True, 'eq14': False, 'eq16': True, 'memory': True}
assignments = []
orderings = []
scores = []
print("MOO query plans")
# for NF in ["CNF", "DNF"]:
#     for query in queries:
#         print(query, NF)
#         with grb.Env() as env:
#             model = OrderOpt(A=A, C=C, D=mem, Sel=sel, goal='cost', bound=0,
#                              predicates=query, NF=NF, new_equations=new_eq, env=grb.Env())
#             set_MOO_method(model, MOO_method, objectives=objectives, goals=goals)
#             model.optimize(timeout=30*60)
#         assignment, ordering = model.get_query_plan()
#         assignments.append(assignment)
#         orderings.append(ordering)
#         scores.extend([model.opt_accuracy, model.opt_memory, model.opt_cost, model.model.Runtime*1000])
#         print("acc, cost, mem: ", model.opt_accuracy, model.opt_cost, model.opt_memory)
#
# with open("assignments.txt", 'w') as fp:
#     for item in assignments:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#
# with open("orderings.txt", 'w') as fp:
#     for item in orderings:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#
# with open("scores.txt", 'w') as fp:
#     for item in scores:
#         # write each item on a new line
#         fp.write("%s\n" % item)

greedy_assignments = []
greedy_orderings = []
greedy_scores = []
print("Greedy query plans")
for NF in ["CNF", "DNF"]:
    for query in queries:
        print(query, NF)
        with grb.Env() as env:
            model = ModelOpt(A=A, C=C, D=mem, Sel=sel, goal='', bound=0,
                             predicates=query, NF=NF, new_equations=new_eq, env=grb.Env())
            set_MOO_method(model, "greedy_method", objectives=[], goals=[])
            model.optimize(timeout=30*60)
        assignment, ordering = model.get_query_plan()
        greedy_assignments.append(assignment)
        print(assignment)
        greedy_orderings.append(ordering)
        greedy_scores.extend([model.opt_accuracy, model.opt_memory, model.opt_cost, model.model.Runtime*1000])
        print("acc, cost, mem: ", model.opt_accuracy, model.opt_cost, model.opt_memory)

with open("greedy_assignments.txt", 'w') as fp:
    for item in greedy_assignments:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("greedy_orderings.txt", 'w') as fp:
    for item in greedy_orderings:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("greedy_scores.txt", 'w') as fp:
    for item in greedy_scores:
        # write each item on a new line
        fp.write("%s\n" % item)
