from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
from Query_tools import flat_list
from data_loader import data_loader
import pandas as pd
import ast
import gurobipy as grb
import multiprocessing as mp

# Experiment 1
## Investigate how much faster my optimizer is versus Ziyu's

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
    with grb.Env() as env:
        model = model_type(A=A, C=C, D=D, Sel=Sel, goal=objective, bound=bound, predicates=query, NF=NF,
                           new_equations=new_eq, env=env)
        model.optimize(timeout=30*60)
    DF_item = [1+query_num, str(query), model_type.__name__, objective, NF, len(flat_list(query)),
               eq_num+1, 0]
    DF_item.extend([model.model.Runtime*1000, model.model.Work, model.output_flag])
    model.model.dispose()
    del model
    del env
    return DF_item

if __name__ == '__main__':
    filename = "C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv"
    A, C, memory, sel = data_loader(filename)

    # Step 1: load queries
    query_file="../queries.txt"
    queries = []
    with open(query_file, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            queries.append(ast.literal_eval(x))

    # Step 2: which new equations are pulling most weight?
    objectives = ['accuracy', 'cost']
    bound_obj = {'accuracy': 'cost', 'cost': 'accuracy'}
    NFs = ["CNF", "DNF"]
    ablations = [
        {'eq14': False, 'accuracy': False, 'eq16': False, 'eq45': False, 'memory': False},
        {'eq14': False, 'accuracy': True, 'eq16': True, 'eq45': True, 'memory': False},
    ]
    df_columns = ["Query_num", "Query", "model_type", "objective", "NF", "num_pred",
                  "ablation", "trial", "comp_time", "work", "output_flag"]
    bounds = {}
    print("calculating bounds...")
    for objective in objectives:
        for NF in NFs:
            for model_type in [ModelOpt, OrderOpt]:
                bounds[(bound_obj[objective], NF, model_type.__name__)] = []
                for query_num, query in enumerate(queries):
                    with grb.Env() as env:
                        model = model_type(A=A, C=C, D=memory, Sel=sel, goal=objective, bound=0,
                                           predicates=query, NF=NF, new_equations=ablations[1], env=env)
                        model.compute_greedy_solution(objective, "min")
                        min_obj = model.model.getVarByName("total_{}".format(objective)).X
                        model.compute_greedy_solution(objective, "max")
                        max_obj = model.model.getVarByName("total_{}".format(objective)).X
                        bounds[(bound_obj[objective], NF, model_type.__name__)].append(
                            min_obj + (max_obj-min_obj)/2)
                        model.model.dispose()
                        del model

    with open("bounds.txt", 'w') as fp:
        fp.write(str(bounds))
    print('Done')
    input_list = []
    print("bounds calculated")
    trials = 1
    for trial in range(trials):
        for query_num, query in enumerate(queries):
            for NF in NFs:
                for objective in objectives:
                    for ablation_num, ablation in enumerate(ablations):
                        for model_type in [ModelOpt, OrderOpt]:
                            input_list.append([A, C, memory, sel, objective,
                                               bounds[(objective, NF, model_type.__name__)][query_num],
                                               (query, query_num), NF,
                                               (ablation, ablation_num), model_type])

    df = []
    for trial in range(trials):
        print("trial {} of {}".format(trial+1, trials))
        with mp.Pool() as pool:
            df.extend(pool.map(mp_optimize, input_list))

        # save all runs
        DF = pd.DataFrame(df, columns=df_columns)
        DF.to_csv("experiment_2_1.csv")