import time

import pandas as pd
import ast

# Experiment 2
# Investigate what is the quickest MOO method
from Models.MO_optimization import calculate_weights, set_MOO_method
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from data_loader import data_loader

filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
df = pd.read_csv(filename)
A, C, memory, robustness, sel = data_loader(filename)

# Step 1: load queries
queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

ablation_MAES = {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True}

df_columns = ["Query_num", "model_type", "NF", "preference_type", "num_pred", "MOO_method",
              "opt_acc", "opt_cost", "opt_mem", "comp_time", "output_flag"]

preference_types = [
    ['acc_loss_norm', 'cost_norm', 'memory_norm'],
    [['acc_loss_norm', 'cost_norm', 'memory_norm']],
    ['acc_loss_norm', ['cost_norm', 'memory_norm']]
]

MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
           'weighted_min_max', 'exponential_weighted_criterion',
           'weighted_product', 'goal_method', 'bounded_objective']

p = 3
goals = [0, 0, 0]
lbs = [0, 0, 0]
ubs = [1, 1, 1]

df = []
for query_num, query in enumerate(queries):
    print(query)
    print()
    for method in MOO_methods:
        print(method)
        print()
        for preference_type in preference_types:
            print(preference_type)
            print()

            print("model_opt, CNF")
            item = [query_num, "model_opt", "CNF", str(preference_type), len(query), method]
            model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=0,
                             predicates=query, NF="CNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            print("model_opt, DNF")
            item = [query_num, "model_opt", "DNF", str(preference_type), len(query), method]
            model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=0,
                             predicates=query, NF="DNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            print("order_opt, CNF")
            item = [query_num, "order_opt", "CNF", str(preference_type), len(query), method]
            model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='cost', bound=0,
                             predicates=query, NF="CNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            print("order_opt, DNF")
            item = [query_num, "order_opt", "DNF", str(preference_type), len(query), method]
            model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='cost', bound=0,
                             predicates=query, NF="DNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

# save all runs
DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_2.csv")
