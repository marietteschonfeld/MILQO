import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../experiment_1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type == "model_opt"]
order_opt_df = df[df.model_type == "order_opt"]

g = sns.catplot(data=model_opt_df[model_opt_df.output_flag==2],
                x="ablation",
                y="comp_time",
                hue="NF",
                col='objective',
                row='Query_num',
                kind='bar',
                palette='viridis')
# g.fig.get_axes()[0].set_ylim(-0.1, max(model_opt_df['comp_time']))
plt.savefig("figures/experiment1_model_opt_ablations_comp_time.png")
plt.show()

g = sns.catplot(data=order_opt_df[order_opt_df.output_flag==2],
                x="ablation",
                y="comp_time",
                hue="NF",
                col='objective',
                row='Query_num',
                kind='bar',
                palette='viridis')
# g.fig.get_axes()[0].set_ylim(-0, max(order_opt_df['comp_time']))
plt.savefig("figures/experiment1_order_opt_ablations_comp_time.png")
plt.show()

temp = model_opt_df[model_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="ablation",
                y="work",
                hue="NF",
                col='objective',
                row='Query_num',
                kind='bar',
                palette='viridis')
# g.fig.get_axes()[0].set_ylim(-0, 0.1+max(temp['work']))
plt.savefig("figures/experiment1_model_opt_ablations_work.png")
plt.show()

temp = order_opt_df[order_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="ablation",
                y="work",
                hue="NF",
                col='objective',
                row='Query_num',
                kind='bar',
                palette='viridis')

# g.fig.get_axes()[0].set_ylim(-0.1, 0.1+max(temp['work']))
plt.savefig("figures/experiment1_order_opt_ablations_work.png")
plt.show()

# How often does bug or timeout occur?
g = sns.countplot(
    data=model_opt_df,
    y="ablation",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.savefig("figures/experiment1_model_opt_bug_report.png")
plt.show()

g = sns.countplot(
    data=order_opt_df,
    y="ablation",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.savefig("figures/experiment1_order_opt_bug_report.png")
plt.show()