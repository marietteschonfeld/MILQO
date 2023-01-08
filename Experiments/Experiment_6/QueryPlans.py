from Models.MO_optimization import set_MOO_method
from Models.OrderOpt import OrderOpt
from data_loader import *
from Models.ModelOpt import *
import gurobipy as grb

modelDB = "../../ssh_landing/NLP_modelDB.csv"
modelDB = pd.read_csv(modelDB)
DB_f1 = modelDB[modelDB.score_type=='f1_score']
DB_f1 = DB_f1.drop(['score_type'], axis=1)
DB_f1.to_csv("../../Data/NLP_modelDB_f1_score.csv")

filename = "../../Data/NLP_modelDB_f1_score.csv"
sel_filename = "../../Data/NLP_selectivity.csv"
A, C, mem, sel = data_loader(filename, sel_filename)

query = [["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
NF = "CNF"
new_eq = {'eq45': True, 'accuracy': True, 'eq14': True, 'eq16': True, 'memory': True}
MOO_method = "archimedean_goal_method"
ub = 47 # read from file
objectives = [['memory_norm', 'cost_norm'], 'acc_loss_norm']
weights = {"acc_loss_norm": 0.75, "cost_norm": 0.125, "memory_norm": 0.125}

# calculate greedy solution for accuracy as goal point
with grb.Env() as env:
    model = OrderOpt(A=A, C=C, D=mem, Sel=sel, goal='accuracy', bound=0, predicates=query, NF=NF, new_equations=new_eq, env=env)
    greedy_weights = {"acc_loss_norm": 1, "cost_norm": 0, "memory_norm": 0}
    set_MOO_method(model, "weighted_sum", objectives=objectives, weights=greedy_weights)
    model.optimize()
    goals = {"memory_norm": model.model.getVarByName("memory_norm").x,
             "cost_norm": model.model.getVarByName("cost_norm").x,
             "acc_loss_norm": model.model.getVarByName("acc_loss_norm").x}

with grb.Env() as env:
    model = OrderOpt(A=A, C=C, D=mem, Sel=sel, goal='cost', bound=0, predicates=query, NF=NF, new_equations=new_eq, env=env)
    model.model.addConstr(model.model.getVarByName("total_cost") <= ub)
    set_MOO_method(model, MOO_method, objectives=objectives, weights=weights, goals=goals)
    model.optimize()
    assignment, ordering = model.get_query_plan()
    score = [model.opt_accuracy, model.opt_memory, model.opt_cost, model.model.Runtime*1000]
    print("acc, cost, mem: ", model.opt_accuracy, model.opt_cost, model.opt_memory)

with open("assignment.txt", 'w') as fp:
    fp.write("%s\n" % assignment)

with open("ordering.txt", 'w') as fp:
    fp.write("%s\n" % ordering)

with open("score.txt", 'w') as fp:
    for item in score:
        # write each item on a new line
        fp.write("%s\n" % item)
