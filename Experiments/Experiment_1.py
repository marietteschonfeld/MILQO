from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from data_loader import data_loader
import pandas as pd
import timeit
import ast

# Experiment 1
## Investigate how much faster my optimizer is versus Ziyu's
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

# Step 2: which new equations are pulling most weight?
ablations = [
    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True},
    {'eq14': True, 'accuracy': True, 'eq13': False, 'eq16': True, 'eq45': True},
    {'eq14': True, 'accuracy': False,  'eq13': True, 'eq16': True, 'eq45': True},
    {'eq14': False, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True},
    {'eq14': True, 'accuracy': False,  'eq13': False, 'eq16': True, 'eq45': True},
    {'eq14': False, 'accuracy': True, 'eq13': False, 'eq16': True, 'eq45': True},
    {'eq14': False, 'accuracy': False, 'eq13': True, 'eq16': True, 'eq45': True},
    {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': True, 'eq45': True},
    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': False, 'eq45': True},
    {'eq14': True, 'accuracy': True, 'eq13': False, 'eq16': False, 'eq45': True},
    {'eq14': True, 'accuracy': False,  'eq13': True, 'eq16': False, 'eq45': True},
    {'eq14': False, 'accuracy': True, 'eq13': True, 'eq16': False, 'eq45': True},
    {'eq14': True, 'accuracy': False,  'eq13': False, 'eq16': False, 'eq45': True},
    {'eq14': False, 'accuracy': True, 'eq13': False, 'eq16': False, 'eq45': True},
    {'eq14': False, 'accuracy': False, 'eq13': True, 'eq16': False, 'eq45': True},
    {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': False, 'eq45': True},

    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': False},
    {'eq14': True, 'accuracy': True, 'eq13': False, 'eq16': True, 'eq45': False},
    {'eq14': True, 'accuracy': False,  'eq13': True, 'eq16': True, 'eq45': False},
    {'eq14': False, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': False},
    {'eq14': True, 'accuracy': False,  'eq13': False, 'eq16': True, 'eq45': False},
    {'eq14': False, 'accuracy': True, 'eq13': False, 'eq16': True, 'eq45': False},
    {'eq14': False, 'accuracy': False, 'eq13': True, 'eq16': True, 'eq45': False},
    {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': True, 'eq45': False},
    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': False, 'eq45': False},
    {'eq14': True, 'accuracy': True, 'eq13': False, 'eq16': False, 'eq45': False},
    {'eq14': True, 'accuracy': False,  'eq13': True, 'eq16': False, 'eq45': False},
    {'eq14': False, 'accuracy': True, 'eq13': True, 'eq16': False, 'eq45': False},
    {'eq14': True, 'accuracy': False,  'eq13': False, 'eq16': False, 'eq45': False},
    {'eq14': False, 'accuracy': True, 'eq13': False, 'eq16': False, 'eq45': False},
    {'eq14': False, 'accuracy': False, 'eq13': True, 'eq16': False, 'eq45': False},
    {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': False, 'eq45': False},
    ]

# save all runs in a dataframe, derive figures later
df_columns = ["Query_num", "model_type", "objective", "NF", "num_pred"]
for i in range(len(ablations)):
    df_columns.extend(["ablation_{}".format(i), "output_flag_ablation_{}".format(i)])

# Step 3: calculate bounds for Ziyu's optimizer, 1/3 inbetween worst and best value
acc_bounds = []
cost_bounds = []
for query in queries:
    model_opt = ModelOpt(A, C, memory, 'cost', 0, query, "DNF", ablations[0])
    model_opt.compute_greedy_solution("accuracy", "min")
    min_acc = model_opt.opt_accuracy
    model_opt.compute_greedy_solution("accuracy", "max")
    max_acc = model_opt.opt_accuracy
    acc_bounds.append(min_acc + (max_acc-min_acc)/3)
    model_opt.compute_greedy_solution("cost", "max")
    min_cost = model_opt.opt_cost
    model_opt.compute_greedy_solution("cost", "min")
    max_cost = model_opt.opt_cost
    cost_bounds.append(max_cost + (max_cost-min_cost)/3)

df = []

# step 4: run all queries in every type of model
# treat every query CNF and DNF
for query_num, query in enumerate(queries):
    print(query)
    # for every query, consider both objectives and both model_opt and order_opt
    item = [query_num, "model_opt", "cost", "CNF", len(query)]
    print("cost, modelopt, CNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = ModelOpt(A=A, C=C, D=memory, goal='cost', bound=acc_bounds[query_num],
                         predicates=query, NF="CNF", new_equations=ablation)
        # time in ms
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    item = [query_num, "order_opt", "cost", "CNF", len(query)]

    print("cost, orderopt, CNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = OrderOpt(A, C, memory, sel, 'cost', acc_bounds[query_num], query, "CNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    item = [query_num, "model_opt", "cost", "DNF", len(query)]

    print("cost, modelopt, DNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = ModelOpt(A, C, memory, 'cost', acc_bounds[query_num], query, "DNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    item = [query_num, "order_opt", "cost", "DNF", len(query)]

    print("cost, orderopt, DNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = OrderOpt(A, C, memory, sel, 'cost', acc_bounds[query_num], query, "DNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)

    item = [query_num, "model_opt", "accuracy", "CNF", len(query)]
    print("accuracy, modelopt, CNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = ModelOpt(A, C, memory, 'accuracy', cost_bounds[query_num], query, "CNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    item = [query_num, "order_opt", "accuracy", "CNF", len(query)]
    print("accuracy, orderopt, CNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = OrderOpt(A, C, memory, sel, 'accuracy', cost_bounds[query_num], query, "CNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    print("accuracy, modelopt, DNF")
    item = [query_num, "model_opt", "accuracy", "DNF", len(query)]
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = ModelOpt(A, C, memory, 'accuracy', cost_bounds[query_num], query, "DNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)
    item = [query_num, "order_opt", "accuracy", "DNF", len(query)]
    print("accuracy, orderopt, DNF")
    for ablation_num, ablation in enumerate(ablations):
        print(ablation)
        model = OrderOpt(A, C, memory, sel, 'accuracy', cost_bounds[query_num], query, "DNF", ablation)
        item.append(1000*timeit.timeit('model.optimize()', globals=globals(), number=1))
        item.append(model.output_flag)
    df.append(item)


# save all runs
DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_1.csv")