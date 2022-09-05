import time

import pandas as pd
import ast

# Experiment 3
# Investigate how much better my optimizer is at MOO
from Models.MO_optimization import set_MOO_method, utopia_solution, worst_solution
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader

filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
df = pd.read_csv(filename)
A, C, memory, sel = data_loader(filename)

# Step 1: load queries
queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))

# Step 2: choose several queries from list (or all?)

# Step 3: compute solutions
ablation_MAES = {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True}
ablation_ZIYU = {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': False, 'eq45': False}

df_columns = ["Query_num", "query", "model_type", "preference_type", "NF", "num_pred", "MOO_method",
              "ablation", "ub", "opt_acc", "opt_cost", "opt_mem", "comp_time", "output_flag"]

preference_types = [
    ['memory_norm', 'cost_norm', 'acc_loss_norm'],
    [['memory_norm', 'cost_norm', 'acc_loss_norm']],
    ['memory_norm', ['cost_norm', 'acc_loss_norm']],
    [['memory_norm', 'cost_norm'], 'acc_loss_norm']
]

MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
               'weighted_min_max', 'exponential_weighted_criterion',
               'weighted_product', 'goal_method', 'bounded_objective']
p = 3
goals = [0, 0, 0]
lbs = [0, 0, 0]
ubs = [1, 1, 1]

ubs_ZIYU = [i/10 for i in range(4, 11)]

df = []
for query_num, query in enumerate(queries):
    print("{}/{} queries, ".format(query_num+1, len(queries)), query)
    for preference_type in preference_types:
        for method in MOO_methods:
            item = [query_num, str(query), "model_opt", str(preference_type), "CNF", len(flat_list(query)),
                    method, "MAES", 0]
            model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=0,
                             predicates=query, NF="CNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            item = [query_num, str(query), "model_opt", str(preference_type), "DNF", len(flat_list(query)),
                    method, "MAES", 0]
            model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=0,
                             predicates=query, NF="DNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            item = [query_num, str(query), "order_opt", str(preference_type), "CNF", len(flat_list(query)),
                    method, "MAES", 0]
            model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='cost', bound=0,
                             predicates=query, NF="CNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)

            item = [query_num, str(query), "order_opt", str(preference_type), "DNF", len(flat_list(query)),
                    method, "MAES", 0]
            model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='cost', bound=0,
                             predicates=query, NF="DNF", new_equations=ablation_MAES)
            start = time.time()
            set_MOO_method(model, method, preference_type, p=p, goals=goals, lbs=lbs, ubs=ubs)
            end = time.time()
            item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
            item.append(model.output_flag)
            df.append(item)


        item = [query_num, str(query), "model_opt", 'accuracy', "CNF", len(flat_list(query)),
                "", "ZIYU"]
        model = ModelOpt(A=A, C=C, D=memory, goal='accuracy', bound=sum(C),
                         predicates=query, NF="CNF", new_equations=ablation_ZIYU)
        utopia_solution(model)
        worst_solution(model)
        ub = model.utopia_cost + (model.worst_cost - model.utopia_cost)/2
        model.bound = ub
        item.append(ub)
        start = time.time()
        model.optimize()
        end = time.time()
        item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
        item.append(model.output_flag)
        df.append(item)

        item = [query_num, str(query), "model_opt", 'accuracy', "DNF", len(flat_list(query)),
                "", "ZIYU"]
        model = ModelOpt(A=A, C=C, D=memory, goal='accuracy', bound=sum(C),
                         predicates=query, NF="DNF", new_equations=ablation_ZIYU)
        utopia_solution(model)
        worst_solution(model)
        ub = model.utopia_cost + (model.worst_cost - model.utopia_cost)/2
        model.bound = ub
        item.append(ub)
        start = time.time()
        model.optimize()
        end = time.time()
        item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
        item.append(model.output_flag)
        df.append(item)

        item = [query_num, str(query), "order_opt", 'accuracy', "CNF", len(flat_list(query)),
                "", "ZIYU"]
        model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='accuracy', bound=sum(C),
                         predicates=query, NF="CNF", new_equations=ablation_ZIYU)
        utopia_solution(model)
        worst_solution(model)
        ub = model.utopia_cost + (model.worst_cost - model.utopia_cost)/2
        model.bound = ub
        item.append(ub)
        start = time.time()
        model.optimize()
        end = time.time()
        item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
        item.append(model.output_flag)
        df.append(item)

        item = [query_num, str(query), "order_opt", 'accuracy', "DNF", len(flat_list(query)),
                "", "ZIYU"]
        model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal='accuracy', bound=sum(C),
                         predicates=query, NF="DNF", new_equations=ablation_ZIYU)
        utopia_solution(model)
        worst_solution(model)
        ub = model.utopia_cost + (model.worst_cost - model.utopia_cost)/2
        model.bound = ub
        item.append(ub)
        start = time.time()
        model.optimize()
        end = time.time()
        item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory, 1000*(end-start)])
        item.append(model.output_flag)
        df.append(item)
    DF = pd.DataFrame(df, columns=df_columns)
    DF.to_csv("experiment_3.csv")

# save all runs
DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_3.csv")
