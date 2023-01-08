# Experiment 3
# Investigate what is the quickest MOO method
from Models.MO_optimization import calculate_weights, set_MOO_method
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader
import ast
import pandas as pd
import multiprocessing as mp
import gurobipy as grb
import time

def mp_optimize(input_data):
    A = input_data[0]
    C = input_data[1]
    D = input_data[2]
    Sel = input_data[3]
    preference_type = input_data[4]
    query, query_num = input_data[5][0], input_data[5][1]
    NF = input_data[6]
    new_eq = input_data[7]
    model_type = input_data[8]
    method = input_data[9]
    p = input_data[10]
    goals = input_data[11]
    lbs = input_data[12]
    ubs = input_data[13]
    with grb.Env() as env:
        model = model_type(A=A, C=C, D=D, Sel=Sel, goal="accuracy", bound=0, predicates=query,
                           NF=NF, new_equations=new_eq, env=env)
        start = time.time()
        model = set_MOO_method(model, method=method, objectives=preference_type,
                               p=p, goals=goals, lbs=lbs, ubs=ubs)
        end = time.time()
        if method == "lexicographic":
            runtime = end-start
        else:
            model.optimize(timeout=30*60)
            runtime = model.model.Runtime
    DF_item = [query_num, query, model_type.__name__, NF, preference_type, method]
    DF_item.extend(([model.opt_accuracy, model.opt_cost, model.opt_memory]))
    DF_item.extend([runtime*1000, model.model.Work, model.output_flag])
    model.model.dispose()
    del model
    del env
    return DF_item

if __name__ == '__main__':
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    A, C, memory, sel = data_loader()

    # Step 1: load queries
    queries = []
    with open('../queries_4.txt', 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            queries.append(ast.literal_eval(x))

    ablation_MAES = {'eq14': False, 'accuracy': True, 'eq16': True, 'eq45': True, 'memory': True}

    df_columns = ["Query_num", "query", "model_type", "NF", "preference_type", "MOO_method",
                  "opt_acc", "opt_cost", "opt_mem", "comp_time", "work", "output_flag"]

    preference_types = [
        ['memory_norm', 'cost_norm', 'acc_loss_norm'],
        ['memory_norm', 'acc_loss_norm', 'cost_norm'],
        ['cost_norm', 'memory_norm', 'acc_loss_norm'],
        ['cost_norm', 'acc_loss_norm', 'memory_norm'],
        ['acc_loss_norm', 'cost_norm', 'memory_norm'],
        ['acc_loss_norm', 'memory_norm', 'cost_norm']
    ]

    # MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
    #            'weighted_min_max', 'exponential_weighted_criterion',
    #            'weighted_product', 'goal_method', 'bounded_objective',
    #                'archimedean_goal_method', 'goal_attainment_method']
    MOO_methods = ['lexicographic']

    p = 3
    goals = {"acc_loss_norm": 0.1,
             "cost_norm": 0.1,
             "memory_norm": 0.1}
    lbs = {"acc_loss_norm": 0,
           "cost_norm": 0,
           "memory_norm": 0}
    ubs = {"acc_loss_norm": 0.5,
           "cost_norm": 0.5,
           "memory_norm": 0.5}

    input_list = []
    NFs = ['CNF', 'DNF']
    trials = 10
    for trial in range(trials):
        for query_num, query in enumerate(queries):
            print("{}/{} queries, ".format(query_num+1, len(queries)), query)
            for NF in NFs:
                # input_list.append([A, C, memory, sel, [], (query, query_num),
                #                    NF, ablation_MAES, ModelOpt, "greedy_method", p, goals, lbs, ubs])

                for method in MOO_methods:
                    for preference_type in preference_types:
                        # inefficient...
                        for idx, objective in enumerate(preference_type):
                            goals[objective] = (2-idx)*0.1
                            ubs[objective] = 0.25 + (2-idx)*0.25

                        for model_type in [ModelOpt, OrderOpt]:
                            input_list.append([A, C, memory, sel, preference_type, (query, query_num),
                                               NF, ablation_MAES, model_type, method, p, goals, lbs, ubs])

    df = []
    with mp.Pool() as pool:
        df.extend(pool.map(mp_optimize, input_list))

    # save all runs
    DF = pd.DataFrame(df, columns=df_columns)
    DF.to_csv("experiment_3_1.csv")
