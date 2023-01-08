import time

import pandas as pd
import ast

# Experiment 4
# Investigate how much better my optimizer is at MOO
from Models.MO_optimization import set_MOO_method, utopia_solution, worst_solution
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader
import gurobipy as grb
import multiprocessing as mp

def mp_optimize(input_data):
    A = input_data[0]
    C = input_data[1]
    D = input_data[2]
    Sel = input_data[3]
    query, query_num = input_data[4][0], input_data[4][1]
    NF = input_data[5]
    new_eq, abl = input_data[6][0], input_data[6][1]
    model_type = input_data[7]
    preference_type = input_data[8]
    method = input_data[9]
    goals = input_data[10]
    goal = input_data[11]
    ub = input_data[12]
    with grb.Env() as env:
        if abl == "MAES":
            model = model_type(A=A, C=C, D=D, Sel=Sel, goal='cost', bound=0, predicates=query,
                           NF=NF, new_equations=new_eq, env=env)
            model = set_MOO_method(model, method=method, objectives=preference_type, goals=goals)
        else:
            model = model_type(A=A, C=C, D=D, Sel=Sel, goal=goal, bound=ub, predicates=query,
                               NF=NF, new_equations=new_eq, env=env)
        model.optimize(timeout=30*60)

    DF_item = [query_num, query, model_type.__name__, NF, abl, preference_type, method, goal, ub]
    DF_item.extend([model.opt_accuracy, model.opt_cost, model.opt_memory])
    DF_item.append(model.model.getVarByName("acc_loss_norm").x)
    DF_item.append(model.model.getVarByName("cost_norm").x)
    DF_item.append(model.model.getVarByName("memory_norm").x)

    DF_item.extend([model.model.Runtime*1000, model.model.Work, model.output_flag])
    model.model.dispose()
    del model
    del env
    return DF_item

if __name__ == "__main__":
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    df = pd.read_csv(filename)
    A, C, memory, sel = data_loader(filename)

    # Step 1: load queries
    queries = []
    with open('../queries_8.txt', 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            queries.append(ast.literal_eval(x))

    # Step 2: choose several queries from list (or all?)

    # Step 3: compute solutions
    ablation_MAES = {'eq14': True, 'accuracy': True, 'eq16': True, 'eq45': True, 'memory': True}
    ablation_ZIYU = {'eq14': False, 'accuracy': False, 'eq16': False, 'eq45': False, 'memory': False}

    df_columns = ["Query_num", "query", "model_type", "NF", "ablation", "preference_type", "MOO_method",
                  "goal", "ub", "opt_acc", "opt_cost", "opt_mem",
                  "opt_acc_norm", "opt_cost_norm", "opt_mem_norm",
                  "comp_time", "work", "output_flag"]

    preference_types = [
        ['memory_norm', 'cost_norm', 'acc_loss_norm'],
        ['memory_norm', 'acc_loss_norm', 'cost_norm'],
        ['cost_norm', 'memory_norm', 'acc_loss_norm'],
        ['cost_norm', 'acc_loss_norm', 'memory_norm'],
        ['acc_loss_norm', 'cost_norm', 'memory_norm'],
        ['acc_loss_norm', 'memory_norm', 'cost_norm']
    ]

    MOO_method = 'archimedean_goal_method'
    goals = {"acc_loss_norm": 0,
             "cost_norm": 0,
             "memory_norm": 0}

    ubs_ZIYU = {'accuracy': [i*10 for i in range(1, 10, 2)],
                'cost': [i/10 for i in range(4, 11, 2)]}

    NFs = ['CNF', 'DNF']
    input_list = []

    for query_num, query in enumerate(queries):
        for NF in NFs:
            input_list.append([A, C, memory, sel, (query, query_num), NF, (ablation_MAES, "MAES"),
                               ModelOpt, None, "Greedy_method", goals, 'accuracy', 0])


            for model_type in [ModelOpt, OrderOpt]:
                for preference_type in preference_types:
                    input_list.append([A, C, memory, sel, (query, query_num), NF, (ablation_MAES, "MAES"),
                                       model_type, preference_type, MOO_method, goals, 'accuracy', 0])

                for goal in ['accuracy', 'cost']:
                    for ub in ubs_ZIYU[goal]:
                        input_list.append([A,C,memory,sel,(query, query_num),NF,(ablation_ZIYU, "ZIYU"),
                                           model_type,"", "",{}, goal, ub])

    trials = 1
    df = []
    for trial in range(trials):
        print("trial {} of {}".format(trial+1, trials))
        with mp.Pool() as pool:
            df.extend(pool.map(mp_optimize, input_list))

        # save all runs
        DF = pd.DataFrame(df, columns=df_columns)
        DF.to_csv("experiment_4_1.csv")
