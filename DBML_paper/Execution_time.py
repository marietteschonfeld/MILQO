import pandas as pd
from Query_tools import *
from data_loader import *
import gurobipy as grb
from Models.ModelOpt import *
from Models.MO_optimization import *
import time

A_tweet, C_tweet, D_tweet, _ = data_loader(filename="C:\\Users\\marie\\Documents\\Software\\MILQO\\DBML_paper\\repo_query\\repository\\tweet_eval_model_stats_f1_raw.csv")
A_coco, C_coco, D_coco, _ = data_loader(filename="C:\\Users\\marie\\Documents\\Software\\MILQO\\DBML_paper\\repo_query\\repository\\coco_model_stats_f1_raw_1.csv")

ablation_MAES = {'eq14': False, 'accuracy': True, 'eq16': True, 'eq45': True, 'memory': True}

MOO_methods = ['weighted_sum', 'lexicographic',
               'weighted_min_max', 'greedy_MOO',
               ]

df_columns = ["query", "NF", "preference_type","MOO_method", "zoo",
              "trial", "comp_time", "work", "output_flag"]

preference_profiles = [
    ['memory_norm', 'cost_norm', 'acc_loss_norm'],
    ['memory_norm', 'acc_loss_norm', 'cost_norm'],
    ['cost_norm', 'memory_norm', 'acc_loss_norm'],
    ['cost_norm', 'acc_loss_norm', 'memory_norm'],
    ['acc_loss_norm', 'cost_norm', 'memory_norm'],
    ['acc_loss_norm', 'memory_norm', 'cost_norm']
]

tweet_queries = generate_queries(6, 10, A_tweet)
coco_queries = generate_queries(6, 10, A_coco)

DF = []
trials = 1
for method in MOO_methods:
    for preference_profile in preference_profiles:
        for NF in ["CNF", "DNF"]:
            for trial in range(trials):
                for query in coco_queries:
                    with grb.Env() as env:
                        model = ModelOpt(A=A_coco, C=C_coco, D=D_coco, Sel=[], goal="accuracy", bound=0, predicates=query,
                                         NF=NF, new_equations=ablation_MAES, env=env)
                        model = set_MOO_method(model, method=method, objectives=preference_profile)
                        start = time.time()
                        model.optimize()
                        end = time.time()
                        DF.append([
                            query, NF, preference_profile,
                            method, "COCO", trial,
                            (end-start)*1000, model.model.Work, model.model.Status
                        ])

                for query in tweet_queries:
                    with grb.Env() as env:
                        model = ModelOpt(A=A_tweet, C=C_tweet, D=D_tweet, Sel=[], goal="accuracy", bound=0, predicates=query,
                                         NF=NF, new_equations=ablation_MAES, env=env)
                        model = set_MOO_method(model, method=method, objectives=preference_profile)
                        start = time.time()
                        model.optimize()
                        end = time.time()
                        DF.append([
                            query, NF, preference_profile,
                            method, "tweet", trial,
                            (end-start)*1000, model.model.Work, model.model.Status
                        ])


DF = pd.DataFrame(DF, columns=df_columns)
DF.to_csv("execution_time.csv")