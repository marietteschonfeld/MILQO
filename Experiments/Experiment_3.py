import pandas as pd
import ast

# Experiment 3
# Investigate how much better my optimizer is at MOO
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

# Step 2: choose several queries from list (or all?)

# Step 3: compute solutions
ablation_MAES = {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True, 'eq45': True}
ablation_ZIYU = {'eq14': False, 'accuracy': False, 'eq13': False, 'eq16': False, 'eq45': False}
df_columns = ["Query_num", "model_type", "objective", "NF", "num_pred",
              "opt_acc_MAES", "opt_cost_MAES", "opt_mem_MAES",
              "opt_acc_ZIYU", "opt_cost_ZIYU", "opt_mem_ZIYU"]

objectives = []

