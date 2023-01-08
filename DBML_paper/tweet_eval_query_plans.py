import pandas as pd
from Query_tools import *
from data_loader import *
import gurobipy as grb
from Models.ModelOpt import *
from Models.MO_optimization import *
import time

cnf_queries = pd.read_csv("repo_query/query/tweet_eval_query_cnf_gt.csv")
dnf_queries = pd.read_csv("repo_query/query/tweet_eval_query_dnf_gt.csv")

A, C, D, _ = data_loader(filename="C:\\Users\\marie\\Documents\\Software\\MILQO\\DBML_paper\\repo_query\\repository\\tweet_eval_model_stats_f1_raw.csv")

ablation_MAES = {'eq14': False, 'accuracy': True, 'eq16': True, 'eq45': True, 'memory': True}

MOO_methods = ['weighted_sum', 'lexicographic',
               'weighted_min_max', 'greedy_MOO',
               ]

df_columns = ["query", "NF", "preference_type",
              "MOO_method", "model_selection", "model_ordering",
              "opt_acc", "opt_cost", "opt_mem",
              "comp_time", "work", "output_flag"]

preference_profiles = [
    ['memory_norm', 'cost_norm', 'acc_loss_norm'],
    ['memory_norm', 'acc_loss_norm', 'cost_norm'],
    ['cost_norm', 'memory_norm', 'acc_loss_norm'],
    ['cost_norm', 'acc_loss_norm', 'memory_norm'],
    ['acc_loss_norm', 'cost_norm', 'memory_norm'],
    ['acc_loss_norm', 'memory_norm', 'cost_norm']
]

DF = []
for query in cnf_queries["query"]:
    for method in MOO_methods:
        for preference_profile in preference_profiles:
            with grb.Env() as env:
                model = ModelOpt(A=A, C=C, D=D, Sel=[], goal="accuracy", bound=0, predicates=parse_query(query),
                                 NF="CNF", new_equations=ablation_MAES, env=env)
                model = set_MOO_method(model, method=method, objectives=preference_profile)
                start = time.time()
                model.optimize()
                end = time.time()
                assignment, ordering = model.get_query_plan()
                DF.append([
                    query, "CNF", preference_profile,
                    method, assignment, ordering,
                    model.opt_accuracy, model.opt_cost, model.opt_memory,
                    (end-start)*1000, model.model.Work, model.model.Status
                ])

for query in dnf_queries["query"]:
    for method in MOO_methods:
        for preference_profile in preference_profiles:
            with grb.Env() as env:
                model = ModelOpt(A=A, C=C, D=D, Sel=[], goal="accuracy", bound=0, predicates=parse_query(query),
                                 NF="DNF", new_equations=ablation_MAES, env=env)
                model = set_MOO_method(model, method=method, objectives=preference_profile)
                start = time.time()
                model.optimize()
                end = time.time()
                assignment, ordering = model.get_query_plan()
                DF.append([
                    query, "DNF", preference_profile,
                    method, assignment, ordering,
                    model.opt_accuracy, model.opt_cost, model.opt_memory,
                    (end-start)*1000, model.model.Work, model.model.Status
                ])

DF = pd.DataFrame(DF, columns=df_columns)
DF.to_csv("tweet_eval_query_plans.csv")
