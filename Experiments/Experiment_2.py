from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader
import pandas as pd
import datetime
import ast

# Experiment 1
## Investigate how much faster my optimizer is versus Ziyu's
filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
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

# Step 2: which new equations are pulling most weight?
ablations = [
    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True},
    {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': False, 'eq45': False},
]
# Step 3: calculate bounds for Ziyu's optimizer, 1/2 inbetween worst and best value
bounds = {}
print("Time starting experiment: ", datetime.datetime.now())
print("calculating bounds...")
objectives = ['accuracy', 'cost']
bound_obj = {'accuracy': 'cost', 'cost': 'accuracy'}
NFs = ["CNF", "DNF"]
for objective in objectives:
    for NF in NFs:
        bounds[(bound_obj[objective], NF, "model_opt")] = []
        bounds[(bound_obj[objective], NF, "order_opt")] = []
        for query_num, query in enumerate(queries):
            model_opt = ModelOpt(A, C, memory, objective, 0, query, NF, ablations[0])
            model_opt.compute_greedy_solution(objective, "min")
            min_obj = model_opt.model.getVarByName("total_{}".format(objective)).x
            model_opt.compute_greedy_solution(objective, "max")
            max_obj = model_opt.model.getVarByName("total_{}".format(objective)).x
            bounds[(bound_obj[objective], NF, "model_opt")].append(min_obj + (max_obj-min_obj)/2)

            order_opt = OrderOpt(A, C, memory, sel, objective, 0, query, NF, ablations[0])
            order_opt.compute_greedy_solution(objective, "min")
            min_obj = order_opt.model.getVarByName("total_{}".format(objective)).x
            order_opt.compute_greedy_solution(objective, "max")
            max_obj = order_opt.model.getVarByName("total_{}".format(objective)).x
            bounds[(bound_obj[objective], NF, "order_opt")].append(min_obj + (max_obj-min_obj)/2)

print(bounds)
df = []
# save all runs in a dataframe, derive figures later
df_columns = ["Query_num", "Query", "model_type", "objective", "NF", "num_pred", "ablation", "trial", "comp_time", "work", "output_flag"]


# step 4: run all queries in every type of model
# treat every query CNF and DNF
trials = 20
for trial in range(trials):
    print('Trial ', trial)
    for query_num, query in enumerate(queries):
        print("{}/{} queries, ".format(query_num+1, len(queries)), query)
        for NF in NFs:
            for objective in objectives:
                for ablation_num, ablation in enumerate(ablations):
                    item = [query_num, str(query), "model_opt", objective, NF, len(flat_list(query)), ablation_num, trial]
                    model = ModelOpt(A=A, C=C, D=memory, goal=objective, bound=bounds[(objective, NF, "model_opt")][query_num],
                                     predicates=query, NF=NF, new_equations=ablation)
                    # time in ms
                    model.optimize(timeout=60*60)
                    item.append(model.model.Runtime*1000)
                    item.append(model.model.Work)
                    item.append(model.output_flag)
                    model.model.dispose()
                    df.append(item)

                    item = [query_num, str(query), "order_opt", objective, NF, len(flat_list(query)), ablation_num, trial]
                    model = OrderOpt(A=A, C=C, D=memory, Sel=sel, goal=objective, bound=bounds[(objective, NF, "order_opt")][query_num],
                                     predicates=query, NF=NF, new_equations=ablation)
                    # time in ms
                    model.optimize(timeout=60*60)
                    item.append(model.model.Runtime*1000)
                    item.append(model.model.Work)
                    item.append(model.output_flag)
                    model.model.dispose()
                    df.append(item)
        DF = pd.DataFrame(df, columns=df_columns)
        DF.to_csv("experiment_2.csv")
    print("Time after trial {}: {}".format(trial, datetime.datetime.now()))

# save all runs
DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_2.csv")