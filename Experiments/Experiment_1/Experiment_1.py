from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader
import pandas as pd
import datetime
import ast
import gurobipy as grb
import multiprocessing as mp

def mp_optimize(input_data):
    A = input_data[0]
    C = input_data[1]
    D = input_data[2]
    Sel = input_data[3]
    objective = input_data[4]
    bound = input_data[5]
    query, query_num = input_data[6][0], input_data[6][1]
    NF = input_data[7]
    new_eq, eq_num = input_data[8][0], input_data[8][1]
    model_type = input_data[9]
    (exp_num, total) = input_data[10]
    with grb.Env() as env:
        model = model_type(A=A, C=C, D=D, Sel=Sel, goal=objective, bound=bound, predicates=query, NF=NF, new_equations=new_eq, env=env)
        model.optimize(timeout=30*60)
    DF_item = [1+query_num, str(query), model_type.__name__, objective, NF, len(flat_list(query)), eq_num+1, 0]
    DF_item.extend([model.model.Runtime*1000, model.model.Work, model.output_flag])
    model.model.dispose()
    del model
    del env
    return DF_item

# Experiment 1
## Investigate how much faster my optimizer is versus Ziyu's
if __name__ == '__main__':
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    # filename = "model_stats_ap.csv"
    # sel_filename = "coco_selectivity.csv"
    A, C, memory, sel = data_loader(filename)#, sel_filename)

    # Step 1: load queries
    queries = []
    with open('../queries_8.txt', 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            queries.append(ast.literal_eval(x))

    # Step 2: which new equations are pulling most weight?
    ablations = {"ModelOpt": [
        {'accuracy': False, 'eq45': False, "memory":False},
        {'accuracy': False, 'eq45': True, "memory":False},
        {'accuracy': True, 'eq45': False, "memory":False},
        {'accuracy': True, 'eq45': True, "memory":False},
    ],
    "OrderOpt":[
        {'eq14': False, 'accuracy': False, 'eq16': False, 'eq45': True, "memory":False},
        {'eq14': False, 'accuracy': False, 'eq16': True, 'eq45': True, "memory":False},
        {'eq14': False, 'accuracy': True, 'eq16': False, 'eq45': True, "memory":False},
        {'eq14': True, 'accuracy': False, 'eq16': False, 'eq45': True, "memory":False},
        {'eq14': True, 'accuracy': True, 'eq16': False, 'eq45': True, "memory":False},
        {'eq14': True, 'accuracy': False, 'eq16': True, 'eq45': True, "memory":False},
        {'eq14': False, 'accuracy': True, 'eq16': True, 'eq45': True, "memory":False},
        {'eq14': True, 'accuracy': True, 'eq16': True, 'eq45': True, "memory":False},
        ]
    }
    # Step 3: calculate bounds for Ziyu's optimizer, 1/2 inbetween worst and best value
    bounds = {}
    print("calculating bounds...")
    objectives = ['accuracy', 'cost']
    bound_obj = {'accuracy': 'cost', 'cost': 'accuracy'}
    NFs = ["CNF", "DNF"]
    for objective in objectives:
        for NF in NFs:
            for model_type in [ModelOpt, OrderOpt]:
                bounds[(bound_obj[objective], NF, model_type.__name__)] = []
                for query_num, query in enumerate(queries):
                    with grb.Env() as env:
                        model = model_type(A=A, C=C, D=memory, Sel=sel, goal=objective, bound=0, predicates=query, NF=NF,
                                           new_equations=ablations[model_type.__name__][-1], env=env)
                        model.compute_greedy_solution(objective, "min")
                        min_obj = model.model.getVarByName("total_{}".format(objective)).X
                        model.compute_greedy_solution(objective, "max")
                        max_obj = model.model.getVarByName("total_{}".format(objective)).X
                        bounds[(bound_obj[objective], NF, model_type.__name__)].append(min_obj + (max_obj-min_obj)/2)
                        model.model.dispose()
                        del model

    with open("bounds.txt", 'w') as fp:
        fp.write(str(bounds))
    print('Done')

    with open("bounds.txt") as fp:
        bounds = fp.read()
        bounds = ast.literal_eval(bounds)

    df = []
    # save all runs in a dataframe, derive figures later
    df_columns = ["Query_num", "Query", "model_type", "objective", "NF", "num_pred", "ablation", "trial", "comp_time", "work", "output_flag"]

    input_list = []
    trials=10
    total = trials*len(objectives)*len(NFs)*2*len(queries)*4+trials*len(objectives)*len(NFs)*2*len(queries)*8
    count = 1
    for trial in range(trials):
        for query_num, query in enumerate(queries):
            for NF in NFs:
                for objective in objectives:
                    for model_type in [ModelOpt, OrderOpt]:
                        for ablation_num, ablation in enumerate(ablations[model_type.__name__]):
                            input_list.append([A, C, memory, sel, objective,
                                               bounds[(objective, NF, model_type.__name__)][query_num],
                                               (query, query_num), NF,
                                               (ablation, ablation_num), model_type,
                                               (count, total)])
                            count+=1

    df = []
    print("start optimization time: ", datetime.datetime.now())
    with mp.Pool() as pool:
        df.extend(pool.map(mp_optimize, input_list))
    print("end optimization time: ", datetime.datetime.now())

    # save all runs
    DF = pd.DataFrame(df, columns=df_columns)
    DF.to_csv("experiment_1_1.csv")
