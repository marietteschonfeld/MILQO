import pandas as pd
import random
import numpy as np

def data_loader(filename="C:\\Users\\marie\\Documents\\Software\\MILQO\\model_stats_ap.csv",
                sel_filename="C:\\Users\\marie\\Documents\\Software\\MILQO\\coco_selectivity.csv"):
    df = pd.read_csv(filename)
    df_sel = pd.read_csv(sel_filename)
    df = df.set_index('model_name')
    df.fillna(0, inplace=True)
    costs = df["cost"]
    costs = costs.to_dict()
    df = df.drop(["cost"], axis=1)
    if "model_stats" in filename:
        df = df.drop('hair_drier', axis=1)
        memory = {name:155*(cost-20)**2 + 1000 for (name, cost) in costs.items()}
    # drop large model
    else:
        # df.drop(df.tail(1).index,inplace=True)
        memory = df["memory"]
        memory = memory.to_dict()
        df = df.drop(['memory'], axis=1)
    A = df.to_dict()
    df_sel = df_sel.set_index("class")
    df_sel = df_sel.drop("Unnamed: 0", axis=1)
    sel = df_sel.to_dict()
    sel = sel['selectivity']
    return A, costs, memory, sel