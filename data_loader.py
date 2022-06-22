import pandas as pd

def data_loader(filename):
    df = pd.read_csv(filename)
    df_sel = pd.read_csv("coco_selectivity.csv")
    df['model_name'] = df['model_name'].str[6:]
    df['model_name'] = df['model_name'].astype(int)
    df['model_name'] = df['model_name'] - 26
    #drop columns which only have zero accuracy
    df = df.drop('hair_drier', axis=1)
    df.sort_values(by='model_name', inplace=True)
    df = df.set_index('model_name')
    costs = df["cost"].values
    df = df.drop(["cost"], axis=1)
    df.fillna(0, inplace=True)
    A = df.to_dict()
    for key in A.keys():
        A[key] = list(A[key].values())
    df_sel = df_sel.set_index("class")
    df_sel = df_sel.drop("Unnamed: 0", axis=1)
    sel = df_sel.to_dict()
    sel = sel['selectivity']
    return A, costs, sel