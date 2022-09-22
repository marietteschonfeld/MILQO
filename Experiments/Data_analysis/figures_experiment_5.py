import ast
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

queries = [
    ([['toxic', 'severe_toxic'], ['negative', 'neutral']], "CNF"),
    ([['insult', 'negative'], ['obscene', 'positive']], "DNF"),
    ([['insult', 'negative'], ['threat', 'neutral'], ['identity_hate', 'negative']], "DNF"),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['negative']], 'CNF'),
    ([['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate'], ['positive', 'neutral']], "CNF")
]


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
        return res
    else:
        group_dfs = []
        for group in query:
            sub_group_df = data[group[0]]==1
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


true_data = []
data = pd.read_csv("../../Data/train_extend.csv")
for query in queries:
    true_data.append(filter_data(query[0], query[1], data, ))

df_columns = ["exp_num", "query", "query_num", "eval_time", "projected_acc", "actual_acc", "projected_cost", "actual_cost"]
df = []

for query_num, query in enumerate(queries):
    for trial in range(len(times)):
        acc = f1_score(true_data[query_num], evals[query_num])
        item = [trial, str(query), query_num, times[query_num][trial], scores[query_num*4], acc, scores[query_num*4+1], times[query_num][trial]/len(data.index)]
        df.append(item)

DF = pd.DataFrame(df, columns=df_columns)
DF.to_csv("experiment_5.csv")

df = pd.read_csv("experiment_5.csv")

g = sns.barplot(data=df,
                x='query_num',
                y='eval_time',
                palette='viridis'
)
plt.savefig("figures/experiment5_comp_time.png")
plt.show()

g = sns.scatterplot(data=df,
                    x='projected_acc',
                    y='actual_acc',
                    )
plt.plot([0, 1], [0, 1])
plt.show()

g = sns.scatterplot(data=df,
                    x='projected_cost',
                    y='actual_cost',
                    )
plt.plot([0, max(df.projected_cost.values)], [0, max(df.projected_cost.values)])
plt.show()