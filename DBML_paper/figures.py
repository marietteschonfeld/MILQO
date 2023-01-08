import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("execution_time.csv")

method = {
          'weighted_sum': "Weighted sum",
          'lexicographic': "Lexicographic",
          'weighted_min_max': 'Weighted min-max',
          "greedy_MOO": "Greedy MOO"}
df['MOO_method'] = df["MOO_method"].map(method)

my_pal = {
    "Weighted sum": "#82A3FF",
    "Lexicographic": "#FF9CBD",
    'Weighted min-max': "#FFD469",
    "Greedy MOO": "#75FF82"
}

err_bar_color = "#F0EF97"
pad = 5
sns.set(font_scale=1.2)

plt.figure(figsize=(4,4))
g = sns.barplot(data=df,
                x='MOO_method',
                y="comp_time",
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("Computation time (ms)")
plt.xticks(rotation=90)
g.set_xlabel("MOO method")
# sns.move_legend(g, loc="upper right", title="Model type")
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/comp_time.png")
plt.show()

df = pd.read_csv("tweet_eval_query_results.csv")
df['MOO_method'] = df["MOO_method"].map(method)

pref = "['memory_norm', 'cost_norm', 'acc_loss_norm']"
temp = df[(df.preference_type == pref)]
plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="actual_f1",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("f1-score (%)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/f1-score-1.png")
plt.show()

plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="actual_cost",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("Execution cost (ms)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/cost-1.png")
plt.show()

plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="opt_mem",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("Storage footprint (Bytes)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/storage-1.png")
plt.show()

df = pd.read_csv("coco_query_results.csv")
df['MOO_method'] = df["MOO_method"].map(method)
pref = "['acc_loss_norm', 'memory_norm', 'cost_norm']"
temp = df[(df.preference_type == pref)&(df.num_pred != 3)]
plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="actual_f1",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("f1-score (%)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/f1-score-2.png")
plt.show()

plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="actual_cost",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("Execution cost (ms)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/cost-2.png")
plt.show()

plt.figure(figsize=(4,3))
g = sns.barplot(data=temp,
                x='num_pred',
                y="opt_mem",
                hue='MOO_method',
                palette=my_pal,
                errcolor='black')
# g.set_xscale('log')
g.set_ylabel("Storage footprint (Bytes)")
g.set_xlabel("Number of predicates")
# plt.legend(loc='lower center', ncols=2)
sns.move_legend(g, loc='lower center',
                title="MOO method", ncol=2)
g.get_legend().remove()
plt.tight_layout()
# for container in g.containers:
#     g.bar_label(container, fmt="%d", padding=pad)
plt.savefig("figures/storage-2.png")
plt.show()