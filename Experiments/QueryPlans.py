from Data.ExecuteQueryPlan import *
import pandas as pd

from Models.MO_optimization import set_MOO_method
from Models.OrderOpt import OrderOpt
from data_loader import *
from Models.ModelOpt import *
import time

modelDB = "../ssh_landing/MIMOP/NLP_modelDB.csv"
modelDB = pd.read_csv(modelDB)
DB_f1 = modelDB[modelDB.score_type=='f1_score']
DB_f1 = DB_f1.drop(['score_type'], axis=1)
DB_f1.to_csv("../Data/NLP_modelDB_f1_score.csv")

filename = "../Data/NLP_modelDB_f1_score.csv"
sel_filename = "../Data/NLP_selectivity.csv"
A, C, mem, sel = data_loader(filename, sel_filename)
print(A)

queries = [
    ([['toxic', 'severe_toxic'], ['negative', 'neutral']], "CNF"),
    ([['insult', 'negative'], ['obscene', 'positive']], "DNF"),
    ([['insult', 'negative'], ['threat', 'neutral'], ['identity_hate', 'negative']], "DNF"),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['negative']], 'CNF'),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['positive', 'neutral']], "CNF")
]

MOO_method = "archimedean_goal_method"
objectives = [['memory_norm', 'cost_norm'], 'acc_loss_norm']
weights = {'memory_norm': 1/8,
           'cost_norm': 2/8,
           'acc_loss_norm': 5/8}
goals = {"memory_norm": 0, "cost_norm": 0, "acc_loss_norm": 0}

new_eq = {'eq45': True, 'accuracy': True, 'eq13': True, 'eq14': True, 'eq16': True}
assignments = []
orderings = []
scores = []
for (query, NF) in queries:
    model = OrderOpt(A=A, C=C, D=mem, Sel=sel, goal='cost', bound=0, predicates=query, NF=NF, new_equations=new_eq)
    set_MOO_method(model, MOO_method, objectives=objectives, weights=weights, goals=goals)
    assignment, ordering = model.get_query_plan()
    assignments.append(assignment)
    orderings.append(ordering)
    scores.extend([model.opt_accuracy, model.opt_memory, model.opt_cost, model.model.Runtime*1000])
    print("acc, cost, mem: ", model.opt_accuracy, model.opt_cost, model.opt_memory)

with open("assignments.txt", 'w') as fp:
    for item in assignments:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("orderings.txt", 'w') as fp:
    for item in orderings:
        # write each item on a new line
        fp.write("%s\n" % item)

with open("scores.txt", 'w') as fp:
    for item in scores:
        # write each item on a new line
        fp.write("%s\n" % item)
