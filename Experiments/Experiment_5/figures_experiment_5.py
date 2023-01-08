import ast
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from Query_tools import flat_list

queries = []
with open('queries.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        queries.append(ast.literal_eval(x))


def filter_data(query, NF, data):
    if NF == "DNF":
        group_dfs = []
        for group in query:
            sub_group_df = (data[group[0]] == 1)
            for pred in group[1:]:
                sub_group_df = sub_group_df & (data[pred] == 1)
            group_dfs.append(sub_group_df)
        res = group_dfs[0]
        for group in group_dfs[1:]:
            res = res | group
    else:
        group_dfs = []
        for group in query:
            sub_group_df = (data[group[0]] == 1)
            for pred in group[1:]:
                sub_group_df = sub_group_df | (data[pred] == 1)
            group_dfs.append(sub_group_df)
        res = group_dfs[0]
        for group in group_dfs[1:]:
            res = res & group
    return res

times = []
with open('times.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        times.append(ast.literal_eval(x))

evals = []
with open('evals.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        evals.append(ast.literal_eval(x))

scores = []
with open('scores.txt', 'r') as fp:
    item = []
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        scores.append(ast.literal_eval(x))


greedy_times = []
with open('greedy_times.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        greedy_times.append(ast.literal_eval(x))

greedy_evals = []
with open('greedy_evals.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        greedy_evals.append(ast.literal_eval(x))

greedy_scores = []
with open('greedy_scores.txt', 'r') as fp:
    item = []
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        greedy_scores.append(ast.literal_eval(x))


true_data = {}
seed=2022
data = pd.read_csv("../../Data/train_extend.csv")
df = data.drop(['Unnamed: 0'], axis=1)
train, test = train_test_split(df, test_size=0.4, random_state=seed)
train = pd.DataFrame(train, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)
for query in queries:
    for NF in ["CNF", "DNF"]:
        true_data[(str(query), NF)] = (filter_data(query, NF, test))

df_columns = ["exp_num", "query", "query_num", "num_pred", "NF", "type", "eval_time", "projected_acc", "actual_acc", "projected_cost", "actual_cost", "memory"]
df = []
count = 1
for nf_ind, NF in enumerate(["CNF", "DNF"]):
    for query_num, query in enumerate(queries):
        for trial in range(len(times[0])):
            acc = f1_score(true_data[(str(query), NF)], evals[count-1])
            item = [trial, str(query),
                    count, len(flat_list(query)), NF,
                    "MO order_opt", times[count-1][trial],
                    scores[(count-1)*4], acc,
                    0,
                    scores[(count-1)*4+2]/len(test.index),
                    scores[(count-1)*4+1]
                    ]
        count +=1
        df.append(item)

count=1
for nf_ind, NF in enumerate(["CNF", "DNF"]):
    for query_num, query in enumerate(queries):
        for trial in range(len(greedy_times[0])):
            acc = f1_score(true_data[(str(query), NF)], greedy_evals[count-1])
            item = [trial, str(query), count, len(flat_list(query)),
                    NF, "Greedy method", greedy_times[count-1][trial],
                    greedy_scores[(count-1)*4], acc,
                    0,
                    greedy_scores[(count-1)*4+2],
                    greedy_scores[(count-1)*4+1]]
            df.append(item)
            count+=1

DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_5.csv")
df = pd.read_csv("experiment_5.csv")

my_pal = {'MO order_opt': "cornflowerblue",
          "Greedy method": "lightcoral"}
err_bar_color = "#F0EF97"
sns.set(font_scale=1.1)

x, y, hue = "num_pred", "proportion", "type"
hue_order = ["Greedy method", "MO order_opt"]

g = sns.barplot(data=df,
                x='query_num',
                y='actual_acc',
                hue='type',
                palette=my_pal,
                errcolor=err_bar_color,
                hue_order=hue_order
                )
g.set_ylabel("F1-score")
g.set_xlabel("Query number")
sns.move_legend(g, "lower left", title="Model type")
for index, label in enumerate(g.get_xticklabels()):
    if index % 2 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.savefig("../figures/experiment5_accuracy.png")
plt.show()

plt.figure(figsize=(8, 6))
g = sns.barplot(data=df,
                x='query_num',
                y='eval_time',
                hue='type',
                palette=my_pal,
                errcolor=err_bar_color,
                hue_order=hue_order
)
g.set_ylabel("Computation time (ms)")
g.set_xlabel("Query number")
sns.move_legend(g, "upper left", title="Model type")
for index, label in enumerate(g.get_xticklabels()):
    if index % 2 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.savefig("../figures/experiment5_comp_time.png")
plt.show()

g = sns.barplot(data=df,
                x='query_num',
                y='memory',
                hue='type',
                palette=my_pal,
                errcolor=err_bar_color,
                hue_order=hue_order
                )
g.set_ylabel("Memory (bytes)")
g.set_xlabel("Query number")
sns.move_legend(g, "upper left", title="Model type")
for index, label in enumerate(g.get_xticklabels()):
    if index % 2 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.savefig("../figures/experiment5_memory.png")
plt.show()

df1 = df[["query_num", "num_pred", "type", "actual_acc"]]
df1_greedy = df1[df1.type == "Greedy method"]["actual_acc"]
df1_MOO= df1[df1.type == "MO order_opt"]["actual_acc"]
df2 = df1[["query_num", "num_pred", "type"]]
df2["proportion"] = 1
proportions = np.ones(df2["proportion"].values.shape)
for i in range(0, 40):
    proportions[i] = df1_MOO.values[i]/df1_greedy.values[i]
df2["proportion"] = proportions.tolist()

g = sns.barplot(data=df,
                x='num_pred',
                y='actual_acc',
                hue='type',
                hue_order=hue_order,
                errcolor=err_bar_color,
                palette=my_pal
                )
g.set_ylabel("F1-score")
g.set_xlabel("Number of predicates")
sns.move_legend(g, "lower left", title="Model type")
# for container in g.containers:
#     g.bar_label(container, fmt="%.2f", padding=5)
plt.savefig("../figures/experiment5_collapsed_acc.png")
plt.show()

df1 = df[["query_num", "num_pred", "type", "eval_time"]]
df1_greedy = df1[df1.type == "Greedy method"]["eval_time"]
df1_MOO= df1[df1.type == "MO order_opt"]["eval_time"]
df2 = df1[["query_num", "num_pred", "type"]]
df2["proportion"] = 1
proportions = np.ones(df2["proportion"].values.shape)
for i in range(0, 40):
    proportions[i] = df1_MOO.values[i]/df1_greedy.values[i]
df2["proportion"] = proportions.tolist()

plt.figure(figsize=(8, 6))
g = sns.barplot(data=df,
                x='num_pred',
                y='eval_time',
                hue='type',
                hue_order=hue_order,
                errcolor=err_bar_color,
                palette=my_pal
                )
g.set_ylabel("Computation time (ms)")
g.set_xlabel("Number of predicates")
sns.move_legend(g, "lower right", title="Model type")
# for container in g.containers:
#     g.bar_label(container, fmt="%.3e", padding=5)
plt.savefig("../figures/experiment5_collapsed_cost.png")
plt.show()

df1 = df[["query_num", "num_pred", "type", "memory"]]
df1_greedy = df1[df1.type == "Greedy method"]["memory"]
df1_MOO= df1[df1.type == "MO order_opt"]["memory"]
df2 = df1[["query_num", "num_pred", "type"]]
df2["proportion"] = 1
proportions = np.ones(df2["proportion"].values.shape)
for i in range(0, 40):
    proportions[i] = df1_MOO.values[i]/df1_greedy.values[i]
df2["proportion"] = proportions.tolist()

g = sns.barplot(data=df,
                x='num_pred',
                y='memory',
                hue='type',
                hue_order=hue_order,
                palette=my_pal,
                errcolor=err_bar_color
                )
g.set_ylabel("Memory (bytes)")
g.set_xlabel("Number of predicates")
# g.set_ylim(0,1.2)
# for container in g.containers:
#     g.bar_label(container, fmt="%.3e", padding=5)
plt.savefig("../figures/experiment5_collapsed_mem.png")
sns.move_legend(g, "lower left", title="Model type")
plt.show()