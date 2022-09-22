import time

import pandas as pd
import ast

# Experiment 2
# Investigate what is the quickest MOO method
from Models.MO_optimization import calculate_weights, set_MOO_method
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader

filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
A, C, memory, sel = data_loader()

# Step 1: load queries
queries = []
with open('queries_8.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

ablation_MAES = {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True}

df_columns = ["Query_num", "query", "model_type", "NF", "preference_type", "num_pred", "MOO_method",
              "trial", "opt_acc", "opt_cost", "opt_mem", "comp_time", "work", "output_flag"]

preference_types = [
    ['memory_norm', 'cost_norm', 'acc_loss_norm'],
    [['memory_norm', 'cost_norm', 'acc_loss_norm']],
    ['memory_norm', ['cost_norm', 'acc_loss_norm']],
    [['memory_norm', 'cost_norm'], 'acc_loss_norm']
]

MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
           'weighted_min_max', 'exponential_weighted_criterion',
           'weighted_product', 'goal_method', 'bounded_objective',
               'archimedean_goal_method', 'goal_attainment_method']

p = 3
goals = {"acc_loss_norm":0,
         "cost_norm":0.1,
         "memory_norm":0.2}
lbs = {"acc_loss_norm": 0,
       "cost_norm": 0.1,
       "memory_norm": 0.2}
ubs = {"acc_loss_norm": 0.4,
       "cost_norm": 0.6,
       "memory_norm": 0.8}
weights = {"acc_loss_norm": 5/8,
           "cost_norm": 2/8,
           "memory_norm": 1/8}

df = []
trials=20
NFs = ['CNF', 'DNF']
for trial in range(trials):
    print("trial", trial)
    for query_num, query in enumerate(queries):
        print("{}/{} queries, ".format(query_num+1, len(queries)), query)
        for NF in NFs:
            print(NF)
            for method in MOO_methods:
                print(method)
                for preference_type in preference_types:
                    item = [query_num, str(query), "model_opt", NF, str(preference_type),
                            len(flat_list(query)), method, trial]
                    model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=0,
                                     predicates=query, NF=NF, new_equations=ablation_MAES)
                    model = set_MOO_method(model, method=method, objectives=preference_type, weights=weights,
                                           p=p, goals=goals, lbs=lbs, ubs=ubs)
                    model.optimize(timeout=5*60)
                    item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory])
                    item.append(1000*model.model.Runtime)
                    item.append(model.model.Work)
                    item.append(model.output_flag)
                    model.model.dispose()
                    df.append(item)

                    item = [query_num, str(query), "order_opt", NF, str(preference_type),
                            len(flat_list(query)), method, trial]
                    model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='cost', bound=0,
                                     predicates=query, NF=NF, new_equations=ablation_MAES)
                    model = set_MOO_method(model, method=method, objectives=preference_type, weights=weights,
                                   p=p, goals=goals, lbs=lbs, ubs=ubs)
                    model.optimize(timeout=5*60)
                    item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory])
                    item.append(1000*model.model.Runtime)
                    item.append(model.model.Work)
                    item.append(model.output_flag)
                    model.model.dispose()
                    df.append(item)

        DF = pd.DataFrame(df, columns=df_columns)
        DF.to_csv("experiment_3-1.csv")

# save all runs
DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_3-1.csv")
