import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("experiment_1_1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

df['Query_num'] = np.where(df['NF'] == 'DNF',
                            df['Query_num'] + 5,
                            df['Query_num'])

output_flag = {2: "Normal",
               3: "Infeasible",
               9: "Time-out"}
df['output_flag'] = df["output_flag"].map(output_flag)

model_opt_df = df[df.model_type == "ModelOpt"]
order_opt_df = df[df.model_type == "OrderOpt"]

modelopt_Ablations = {1:"No amendments",
             2: r'$B_m$',
             3: "Accuracy",
             4: "Both"}

model_opt_df["ablation"] = model_opt_df["ablation"].map(modelopt_Ablations)

orderopt_Ablations = {1:"None",
                      2: r'$R_{m,j}$',
                      3: "Accuracy",
                      4: r"$H_{g,j}$",
                      5: r"Accuracy,$H_{g,j}$",
                      6: r"$R_{m,j}, H_{g,j}$",
                      7: r"$R_{m,j}$, Accuracy",
                      8: "All"}

order_opt_df["ablation"] = order_opt_df["ablation"].map(orderopt_Ablations)

my_pal = {"accuracy": "cornflowerblue", "cost": "lightcoral",
          "Normal": "cornflowerblue", "Time-out": "lightcoral",
          "None": "cornflowerblue", r"$R_{m,j}$, Accuracy": "lightcoral"}
err_bar_color = "#F0EF97"
hue_names = ["Accuracy", "Cost"]
sns.set(font_scale=1.1)

# g = sns.catplot(data=model_opt_df,
#                 x="ablation",
#                 y="comp_time",
#                 hue="objective",
#                 col_wrap=3,
#                 col='Query_num',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_yscale('log')
# sns.move_legend(g, "lower center")
# g.fig.get_axes()[0].set_ylim(min(model_opt_df['comp_time']), max(model_opt_df['comp_time'])+1000000)
# g.set_titles(template='Query number: {col_name}')
# plt.tight_layout()
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Amendments")
#     ax.set_ylabel("Computation time (ms)")
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment1_model_opt_ablations_comp_time.png")
# plt.show()
#
# g = sns.barplot(data=model_opt_df,
#                 x="ablation",
#                 y="comp_time",
#                 hue="objective",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# g.set_xlabel("Amendments")
# g.set_ylabel("Computation time (ms)")
# sns.move_legend(g, loc="upper right", title="Objective")
# g.set_ylim(min(model_opt_df['comp_time']), max(model_opt_df['comp_time'])+1000000)
# plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment1_model_opt_ablations_comp_time_collapsed.png")
# plt.show()
#
# filt = (model_opt_df.ablation == "No amendments") | (model_opt_df.ablation == "Both")
# g = sns.barplot(data=model_opt_df[filt],
#                 x="ablation",
#                 y="comp_time",
#                 hue="objective",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# g.set_xlabel("Amendments")
# g.set_ylabel("Computation time (ms)")
# sns.move_legend(g, loc="upper right", title="Objective")
# g.set_ylim(10, max(model_opt_df['comp_time'])+1000000)
# plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment1_model_opt_ablations_comp_time_collapsed_defence.png")
# plt.show()
#
# # g = sns.catplot(data=model_opt_df,
# #                 x="ablation",
# #                 y="comp_time",
# #                 hue="NF",
# #                 row='objective',
# #                 col='Query_num',
# #                 kind='bar',
# #                 palette=my_pal)
# # g.fig.get_axes()[0].set_yscale('log')
# # g.fig.get_axes()[0].set_ylim(min(model_opt_df['comp_time']), max(model_opt_df['comp_time'])+1000000)
# # for ax in g.fig.get_axes():
# #     ax.set_ylabel("Computation time (ms)")
# #     for container in ax.containers:
# #         ax.bar_label(container, fmt="%d", padding=5)
# # plt.savefig("../figures/experiment1_model_opt_ablations_comp_time_horizontal.png")
# # plt.show()
#
#
#
# g = sns.catplot(data=order_opt_df,
#                 x="ablation",
#                 y="comp_time",
#                 hue="objective",
#                 col_wrap=3,
#                 col='Query_num',
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_yscale('log')
# g.fig.get_axes()[0].set_ylim(min(order_opt_df['comp_time']), max(order_opt_df['comp_time'])+1000000)
# g.set_titles(template='Query number: {col_name}')
# g._legend.set_title("Objective")
# for ax in g.fig.get_axes():
#     ax.set_xticks(ax.get_xticklabels(), rotation=45)
#     ax.set_xlabel("Amendments")
#     ax.set_ylabel("Computation time (ms)")
#     # for container in ax.containers:
#     #     ax.bar_label(container, fmt="%0.2e", padding=5)
# plt.tight_layout()
# sns.move_legend(g, "lower center")
# plt.savefig("../figures/experiment1_order_opt_ablations_comp_time.png")
# plt.show()
#
# g = sns.barplot(data=order_opt_df,
#                 x="ablation",
#                 y="comp_time",
#                 hue="objective",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# g.set_ylim(min(order_opt_df['comp_time']), max(order_opt_df['comp_time'])+1000000)
# plt.xticks(rotation=90)
# plt.tight_layout()
# g.set_xlabel("Amendments")
# g.set_ylabel("Computation time (ms)")
# sns.move_legend(g, loc="lower left", title="Objective")
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=5)
# plt.savefig("../figures/experiment1_order_opt_ablations_comp_time_collapsed.png")
# plt.show()

filt = (order_opt_df.ablation == "None") | (order_opt_df.ablation == r"$R_{m,j}$, Accuracy")
g = sns.barplot(data=order_opt_df[filt],
                x="ablation",
                y="comp_time",
                errcolor=err_bar_color,
                palette=my_pal)
g.set_yscale('log')
g.set_xlabel("Amendments")
g.set_ylabel("Computation time (ms)")
# sns.move_legend(g, loc="upper right", title="Objective")
g.set_ylim(10, max(order_opt_df['comp_time'])+1000000)
plt.tight_layout()
for container in g.containers:
    g.bar_label(container, fmt="%d", padding=5)
plt.savefig("../figures/experiment1_order_opt_ablations_comp_time_collapsed_defence.png")
plt.show()

# g = sns.catplot(data=order_opt_df,
#                 x="ablation",
#                 y="comp_time",
#                 hue="NF",
#                 row='objective',
#                 col='Query_num',
#                 kind='bar',
#                 palette='viridis')
# g.fig.get_axes()[0].set_ylim(min(order_opt_df['comp_time']), max(order_opt_df['comp_time'])+1000000)
# g.fig.get_axes()[0].set_yscale('log')
# for ax in g.fig.get_axes():
#     ax.set_ylabel("Computation time (ms)")
#     for container in ax.containers:
#         ax.bar_label(container, fmt="%.1e", padding=5)
# plt.savefig("../figures/experiment1_order_opt_ablations_comp_time_horizontal.png")
# plt.show()

# g = sns.catplot(data=model_opt_df,
#                 x="ablation",
#                 y="work",
#                 hue="objective",
#                 col='Query_num',
#                 col_wrap=3,
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_ylim(min(model_opt_df['work']), max(model_opt_df['work']))
# g.fig.get_axes()[0].set_yscale('log')
# sns.move_legend(g, "lower center")
# g._legend.set_title("Objective")
# g.set_titles(template='Query number: {col_name}')
# plt.tight_layout()
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Amendments")
#     ax.set_ylabel("Work")
#     for container in ax.containers:
#         ax.bar_label(container,fmt="%.2f")
# plt.savefig("../figures/experiment1_model_opt_ablations_work.png")
# plt.show()
#
# g = sns.barplot(data=model_opt_df,
#                 x="ablation",
#                 y="work",
#                 hue="objective",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# g.set_ylim(min(model_opt_df['work']), max(model_opt_df['work']))
# g.set_xlabel("Amendments")
# g.set_ylabel("Work")
# sns.move_legend(g, loc="upper right", title="Objective")
# plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%.2f", padding=5)
# plt.savefig("../figures/experiment1_model_opt_ablations_work_collapsed.png")
# plt.show()
#
# # g = sns.catplot(data=model_opt_df,
# #                 x="ablation",
# #                 y="work",
# #                 hue="NF",
# #                 row='objective',
# #                 col='Query_num',
# #                 kind='bar',
# #                 palette='viridis')
# # g.fig.get_axes()[0].set_ylim(min(model_opt_df['work']), max(model_opt_df['work']))
# # g.fig.get_axes()[0].set_yscale('log')
# # for ax in g.fig.get_axes():
# #     for container in ax.containers:
# #         ax.bar_label(container,fmt="%.2f")
# # plt.savefig("../figures/experiment1_model_opt_ablations_work_horizontal.png")
# # plt.show()
#
# g = sns.catplot(data=order_opt_df,
#                 x="ablation",
#                 y="work",
#                 hue='objective',
#                 col='Query_num',
#                 col_wrap=3,
#                 kind='bar',
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.fig.get_axes()[0].set_ylim(min(order_opt_df['work']), max(order_opt_df['work']))
# g.fig.get_axes()[0].set_yscale('log')
# plt.tight_layout()
# sns.move_legend(g, "lower center")
# g.set_titles(template='Query number: {col_name}')
# plt.xticks(rotation=90)
# for ax in g.fig.get_axes():
#     ax.set_xlabel("Amendments")
#     ax.set_ylabel("Work")
#     for container in ax.containers:
#         ax.bar_label(container,fmt="%.2f")
# plt.savefig("../figures/experiment1_order_opt_ablations_work.png")
# plt.show()
#
# g = sns.barplot(data=order_opt_df,
#                 x="ablation",
#                 y="work",
#                 hue="objective",
#                 errcolor=err_bar_color,
#                 palette=my_pal)
# g.set_yscale('log')
# g.set_ylim(min(order_opt_df['work']), max(order_opt_df['work']))
# plt.xticks(rotation=90)
# plt.tight_layout()
# g.set_ylabel("Work")
# g.set_xlabel("Amendments")
# sns.move_legend(g, loc="upper right", title="Objective")
# for container in g.containers:
#     g.bar_label(container, fmt="%.2f", padding=5)
# plt.savefig("../figures/experiment1_order_opt_ablations_work_collapsed.png")
# plt.show()
#
# # g = sns.catplot(data=order_opt_df,
# #                 x="ablation",
# #                 y="work",
# #                 hue="NF",
# #                 row='objective',
# #                 col='Query_num',
# #                 kind='bar',
# #                 palette='viridis')
# # g.fig.get_axes()[0].set_ylim(min(order_opt_df['work']), max(order_opt_df['work']))
# # g.fig.get_axes()[0].set_yscale('log')
# # for ax in g.fig.get_axes():
# #     for container in ax.containers:
# #         ax.bar_label(container,fmt="%.2f")
# # plt.savefig("../figures/experiment1_order_opt_ablations_work_horizontal.png")
# # plt.show()
#
#
#
# # How often does bug or timeout occur?
# x,y = 'ablation', 'output_flag'
#
# df1 = model_opt_df.groupby(x)[y].value_counts(normalize=True)
# df1 = df1.mul(100)
# df1 = df1.rename('percent').reset_index()
#
# g=sns.barplot(data=df1,
#               x="ablation",
#               y='percent',
#               hue="output_flag",
#               order=modelopt_Ablations.values(),
#               errcolor=err_bar_color,
#               palette=my_pal)
# g.set_xlabel("Amendments")
# g.set_ylabel("Percent (%)")
# sns.move_legend(g, "upper right", title="Output status")
# plt.tight_layout()
# plt.savefig("../figures/experiment1_model_opt_bug_report.png")
# plt.show()
#
# df1 = order_opt_df.groupby(x)[y].value_counts(normalize=True)
# df1 = df1.mul(100)
# df1 = df1.rename('percent').reset_index()
#
# g=sns.barplot(data=df1,
#               x="ablation",
#               y='percent',
#               hue="output_flag",
#               order=orderopt_Ablations.values(),
#               errcolor=err_bar_color,
#               palette=my_pal)
# plt.xticks(rotation=90)
# plt.tight_layout()
# g.set_xlabel("Amendments")
# g.set_ylabel("Percent (%)")
# sns.move_legend(g, "upper right", title="Output status")
# plt.tight_layout()
# plt.savefig("../figures/experiment1_order_opt_bug_report.png")
# plt.show()
#
# percentages_cnf = []
# percentages_dnf = []
# print("Model opt")
# for query_num in range(1, 6):
#     percentage_cnf = sum(model_opt_df[(model_opt_df.Query_num==query_num) &
#                                       (model_opt_df.ablation==1) &
#                                       (model_opt_df.NF=="CNF")]["comp_time"].values)/\
#                  sum(model_opt_df[(model_opt_df.Query_num==query_num) &
#                                   (model_opt_df.ablation==4) &
#                                   (model_opt_df.NF=="CNF")]["comp_time"].values)
#
#     percentage_dnf = sum(model_opt_df[(model_opt_df.Query_num==query_num) &
#                                       (model_opt_df.ablation==1) &
#                                       (model_opt_df.NF=="DNF")]["comp_time"].values)/ \
#                      sum(model_opt_df[(model_opt_df.Query_num==query_num) &
#                                       (model_opt_df.ablation==4) &
#                                       (model_opt_df.NF=="DNF")]["comp_time"].values)
#
#     percentages_cnf.append(percentage_cnf)
#     percentages_dnf.append(percentage_dnf)
#     print("improvement ablation 1 over 4 for query {} CNF: ".format(query_num), percentage_cnf)
#     print("improvement ablation 1 over 4 for query {} DNF: ".format(query_num), percentage_dnf)
#
# print("average improvement cnf: ", sum(percentages_cnf)/6)
# print("average improvement dnf: ", sum(percentages_dnf)/6)
#
# percentages_cnf = []
# percentages_dnf = []
# print("order opt ablation 8")
# for query_num in range(1, 6):
#     percentage_cnf = sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==1) &
#                                       (order_opt_df.NF=="CNF")]["comp_time"].values)/ \
#                      sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==8) &
#                                       (order_opt_df.NF=="CNF")]["comp_time"].values)
#
#     percentage_dnf = sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==1) &
#                                       (order_opt_df.NF=="DNF")]["comp_time"].values)/ \
#                      sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==8) &
#                                       (order_opt_df.NF=="DNF")]["comp_time"].values)
#
#     percentages_cnf.append(percentage_cnf)
#     percentages_dnf.append(percentage_dnf)
#     print("improvement ablation 8 over 1 for query {} CNF: ".format(query_num), percentage_cnf)
#     print("improvement ablation 8 over 1 for query {} DNF: ".format(query_num), percentage_dnf)
#
# print("average improvement cnf: ", sum(percentages_cnf)/6)
# print("average improvement dnf: ", sum(percentages_dnf)/6)
#
# percentages_cnf = []
# percentages_dnf = []
# print("order opt ablation 7")
# for query_num in range(1,6):
#     percentage_cnf = sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==1) &
#                                       (order_opt_df.NF=="CNF")]["comp_time"].values)/ \
#                      sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==7) &
#                                       (order_opt_df.NF=="CNF")]["comp_time"].values)
#
#     percentage_dnf = sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==1) &
#                                       (order_opt_df.NF=="DNF")]["comp_time"].values)/ \
#                      sum(order_opt_df[(order_opt_df.Query_num==query_num) &
#                                       (order_opt_df.ablation==7) &
#                                       (order_opt_df.NF=="DNF")]["comp_time"].values)
#
#     percentages_cnf.append(percentage_cnf)
#     percentages_dnf.append(percentage_dnf)
#     print("improvement ablation 7 over 1 for query {} CNF: ".format(query_num), percentage_cnf)
#     print("improvement ablation 7 over 1 for query {} DNF: ".format(query_num), percentage_dnf)
#
# print("average improvement cnf: ", sum(percentages_cnf)/6)
# print("average improvement dnf: ", sum(percentages_dnf)/6)