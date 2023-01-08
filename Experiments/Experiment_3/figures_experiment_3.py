import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("experiment_3.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

df['Query_num'] = np.where(df['NF'] == 'DNF',
                           df['Query_num'] + 5,
                           df['Query_num'])

output_flag = {2: "Normal",
               3: "Infeasible",
               9: "Time-out"}
df['output_flag'] = df["output_flag"].map(output_flag)

type_map = {"ModelOpt": "model_opt",
            "OrderOpt": "order_opt"}
df['model_type'] = df["model_type"].map(type_map)

method = {'weighted_global_criterion': "Weighted global criterion",
            'weighted_sum': "Weighted sum",
           'lexicographic': "Lexicographic",
            'weighted_min_max': 'Weighted min-max',
           'exponential_weighted_criterion': 'Exponential weighted criterion',
          'weighted_product': "Weighted product",
           'goal_method': "Goal method",
           'bounded_objective': "Bounded objective",
           'archimedean_goal_method': "Archimedean goal method",
           'goal_attainment_method': "Goal attainment method",
          "greedy_method": "Greedy method"}
df['MOO_method'] = df["MOO_method"].map(method)

model_opt_df = df[df.model_type == "model_opt"]
order_opt_df = df[df.model_type == "order_opt"]

my_pal = {
    "Old version": "cornflowerblue",
    "New version": "lightcoral",
    "Normal": "cornflowerblue",
    "Time-out": "lightcoral",
    "Infeasible": "#F0EF97",
    "Weighted global criterion": "cornflowerblue",
    "Weighted sum": "cornflowerblue",
    "Lexicographic": "cornflowerblue",
    'Weighted min-max': "cornflowerblue",
    'Exponential weighted criterion': "cornflowerblue",
    "Weighted product": "cornflowerblue",
    "Goal method": "cornflowerblue",
    "Bounded objective": "cornflowerblue",
    "Archimedean goal method": "cornflowerblue",
    "Goal attainment method": "cornflowerblue",
    "Greedy method": "cornflowerblue",
    "CNF": "cornflowerblue",
    "DNF": "lightcoral",
    "model_opt": "cornflowerblue",
    "order_opt": "lightcoral"
}
err_bar_color = "#F0EF97"
pad = 5
sns.set(font_scale=1.1)

# How often does bug or timeout occur?
x,y = 'MOO_method', 'output_flag'

df1 = model_opt_df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

plt.figure(figsize=(8,6))
g = sns.barplot(
    data=df1,
    x="MOO_method",
    y='percent',
    hue="output_flag",
    palette=my_pal,
)
plt.xticks(rotation=90)
g.set_ylabel("Percent (%)")
g.set_xlabel("MOO method")
plt.tight_layout()
sns.move_legend(g, "upper right", title="Output status")
plt.savefig("../figures/experiment3_model_opt_bug_report.png")
plt.show()

df1 = order_opt_df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

plt.figure(figsize=(8,6))
g = sns.barplot(
    data=df1,
    x="MOO_method",
    y='percent',
    hue="output_flag",
    palette=my_pal,
)
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
g.set_ylabel("Percent (%)")
plt.tight_layout()
sns.move_legend(g, "upper right", title="Output status")
plt.savefig("../figures/experiment3_order_opt_bug_report.png")
plt.show()

methods = ['Greedy method', 'Weighted sum',
           'Weighted min-max', 'Goal method', 'Bounded objective',
           'Archimedean goal method', 'Goal attainment method']
# Which method is fastest?
plt.figure(figsize=(8,8))
temp = df[(df.output_flag == "Normal")]
g = sns.barplot(data=temp,
                x='MOO_method',
                y="comp_time",
                hue="model_type",
                palette=my_pal,
                errcolor=err_bar_color)
#g.set_xscale('log')
g.set_ylabel("Computation time (ms)")
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
sns.move_legend(g, loc="upper right", title="Model type")
plt.tight_layout()
for container in g.containers:
    g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("../figures/experiment3_comp_time.png")
plt.show()

plt.figure(figsize=(8,8))
temp = df[(df.output_flag == "Normal")]
g = sns.barplot(data=temp,
                x='MOO_method',
                y="work",
                hue='model_type',
                errcolor=err_bar_color,
                palette=my_pal)
g.set_ylabel("Work")
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
sns.move_legend(g, loc="upper right", title="Model type")
plt.tight_layout()
for container in g.containers:
    g.bar_label(container, fmt="%.2f", padding=pad)
plt.savefig("../figures/experiment3_work.png")
plt.show()

# Which method is fastest?
plt.figure(figsize=(8,8))
temp = df[(df.output_flag == "Normal") & (df.MOO_method.isin(methods))]
g = sns.barplot(data=temp,
                x='MOO_method',
                y="comp_time",
                hue='model_type',
                errcolor=err_bar_color,
                palette=my_pal)
g.set_ylabel("Computation time (ms)")
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
sns.move_legend(g, loc="upper right", title="Model type")
for container in g.containers:
    g.bar_label(container, fmt="%d", padding=pad)
plt.tight_layout()
plt.savefig("../figures/experiment3_comp_time_2.png")
plt.show()

plt.figure(figsize=(8,8))
temp = df[(df.output_flag == "Normal") & (df.MOO_method.isin(methods))]
g = sns.barplot(data=temp,
                x='MOO_method',
                y="work",
                hue='model_type',
                errcolor=err_bar_color,
                palette=my_pal)
g.set_ylabel("Work")
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
sns.move_legend(g, loc="upper right", title="Model type")
for container in g.containers:
    g.bar_label(container, fmt="%.2f", padding=pad)
plt.tight_layout()
plt.savefig("../figures/experiment3_work_2.png")
plt.show()
