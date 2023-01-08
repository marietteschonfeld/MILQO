import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("experiment_2_1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

df['Query_num'] = np.where(df['NF'] == 'DNF',
                           df['Query_num'] + 35,
                           df['Query_num'])

output_flag = {2: "Normal",
               3: "Infeasible",
               9: "Time-out"}
df['output_flag'] = df["output_flag"].map(output_flag)

ablations = {1: "Old",
             2: "New"}

df["ablation"] = df["ablation"].map(ablations)


model_opt_df = df[df.model_type == "ModelOpt"]
order_opt_df = df[df.model_type == "OrderOpt"]

my_pal = {
    "Old": "cornflowerblue",
    "New": "lightcoral",
    "Normal": "cornflowerblue",
    "Time-out": "lightcoral"
}
err_bar_color = "#F0EF97"
sns.set(font_scale=1.1)

col_order = ['Old', 'New']

# hue = df[['ablation', 'model_type']].apply(
#     lambda row: f"{row.ablation}, {row.model_type}", axis=1)
# hue.name = 'ablation, model_type'
# g = sns.barplot(data=df[(df.output_flag != 3) & (df.objective=="accuracy")],
#                 x="num_pred",
#                 y="comp_time",
#                 hue=hue,
#                 # col='objective',
#                 # row='NF',
#                 palette='viridis')
# g.set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# plt.savefig("../figures/experiment2_comp_time_accuracy.png")
# plt.show()
#
# g = sns.barplot(data=df[(df.output_flag != 3) & (df.objective=="cost")],
#                 x="num_pred",
#                 y="comp_time",
#                 hue=hue,
#                 # col='objective',
#                 # row='NF',
#                 palette='viridis')
# g.set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# plt.savefig("../figures/experiment2_comp_time_cost.png")
# plt.show()

# x,y = 'ablation', 'output_flag'
#
# df1 = model_opt_df.groupby(x)["num_pred", y].value_counts(normalize=True)
# df1 = df1.mul(100*7)
# df1 = df1.rename('percent').reset_index()

# g = sns.catplot(data=df1,
#                 x="num_pred",
#                 y="percent",
#                 hue="output_flag",
#                 col="ablation",
#                 kind='bar',
#                 col_order=col_order,
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# # g.fig.get_axes()[0].set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# g.set_titles(template='Ablation: {col_name}')
# g._legend.set_title("Output status")
# for ax in g.fig.get_axes():
#     ax.set_ylabel("Percent (%)")
#     ax.set_xlabel("Number of predicates")
# plt.savefig("../figures/experiment2_model_opt_bug_report.png")
# plt.show()
#
# df1 = order_opt_df.groupby(x)["num_pred", y].value_counts(normalize=True)
# df1 = df1.mul(100*7)
# df1 = df1.rename('percent').reset_index()
#
# g = sns.catplot(data=df1,
#                 x="num_pred",
#                 y="percent",
#                 hue="output_flag",
#                 col="ablation",
#                 col_order=col_order,
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# # g.fig.get_axes()[0].set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# g.set_titles(template='Ablation: {col_name}')
# g._legend.set_title("Output status")
# for ax in g.fig.get_axes():
#     ax.set_ylabel("Percent (%)")
#     ax.set_xlabel("Number of predicates")
# plt.savefig("../figures/experiment2_order_opt_bug_report.png")
# plt.show()
#
# temp = model_opt_df[(model_opt_df.output_flag == "Normal") & (model_opt_df.num_pred < 24)]
# g = sns.catplot(data=temp,
#                 x="num_pred",
#                 y="comp_time",
#                 hue="ablation",
#                 col='objective',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# g.set_titles(template='Objective: {col_name}')
# g._legend.set_title("Ablation")
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Number of Predicates")
#     ax.set_ylabel("Computation time (ms)")
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment2_model_opt_comp_time.png")
# plt.show()
#
# temp = model_opt_df[(model_opt_df.output_flag == "Normal") & (model_opt_df.num_pred < 24)]
# g = sns.barplot(data=temp,
#                 x="num_pred",
#                 y="comp_time",
#                 hue="ablation",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp.comp_time))
# # g._legend.set_title("Ablation")
# g.set_xlabel("Number of Predicates")
# g.set_ylabel("Computation time (ms)")
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment2_model_opt_comp_time_defence.png")
# plt.show()
#
# temp = order_opt_df[(order_opt_df.output_flag=="Normal") & (order_opt_df.num_pred < 24)]
# g = sns.catplot(data=temp,
#                 x="num_pred",
#                 y="comp_time",
#                 hue="ablation",
#                 col='objective',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_yscale('log')
# g.set_titles(template='Objective: {col_name}')
# g._legend.set_title("Ablation")
# # g.fig.get_axes()[0].set_ylim(0.00001, 10000+max(temp['comp_time']))
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Number of Predicates")
#     ax.set_ylabel("Computation time (ms)")
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment2_order_opt_comp_time.png")
# plt.show()

temp = order_opt_df[(order_opt_df.output_flag=="Normal") & (order_opt_df.num_pred < 24)]
temp.comp_time = temp.comp_time/1000.
g = sns.catplot(data=temp,
                x="num_pred",
                y="comp_time",
                hue="ablation",
                kind='bar',
                height=4.8,
                aspect=6/4.8,
                errcolor=err_bar_color,
                palette=my_pal)
g.fig.get_axes()[0].set_yscale('log')
g._legend.set_title("Version")
g.fig.get_axes()[0].set_ylim(0.1, 3000)
g.fig.get_axes()[0].set_xlabel("Number of Predicates")
g.fig.get_axes()[0].set_ylabel("Computation time (s)")
for ax in g.fig.get_axes():
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=5)
plt.savefig("../figures/experiment2_order_opt_comp_time_defence.png")
plt.show()

# temp = model_opt_df[(model_opt_df.output_flag=="Normal") & (model_opt_df.num_pred < 24)]
# g = sns.catplot(data=temp,
#                 x="num_pred",
#                 y="work",
#                 hue="ablation",
#                 col='objective',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_yscale('log')
# g.set_titles(template='Objective: {col_name}')
# g._legend.set_title("Ablation")
# # g.fig.get_axes()[0].set_ylim(0.00001, 1+max(temp['work']))
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Number of Predicates")
#     ax.set_ylabel("Work")
#     for container in ax.containers:
#         ax.bar_label(container,fmt="%.2f")
# plt.savefig("../figures/experiment2_model_opt_work.png")
# plt.show()
#
# temp = order_opt_df[(order_opt_df.output_flag=="Normal") & (order_opt_df.num_pred < 24)]
# g = sns.catplot(data=temp,
#                 x="num_pred",
#                 y="work",
#                 hue="ablation",
#                 col='objective',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
#
# g.fig.get_axes()[0].set_yscale('log')
# g.set_titles(template='Objective: {col_name}')
# g._legend.set_title("Ablation")
# # g.fig.get_axes()[0].set_ylim(0.00001, 1+max(temp['work']))
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Number of Predicates")
#     ax.set_ylabel("Work")
#     for container in ax.containers:
#         ax.bar_label(container,fmt="%.2f")
# plt.savefig("../figures/experiment2_order_opt_work.png")
# plt.show()
