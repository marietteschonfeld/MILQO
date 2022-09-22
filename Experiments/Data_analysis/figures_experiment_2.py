import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../experiment_2.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type == "model_opt"]
order_opt_df = df[df.model_type == "order_opt"]

temp = model_opt_df[model_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="num_pred",
                y="comp_time",
                hue="ablation",
                col='objective',
                row='NF',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 10**6)
plt.savefig("figures/experiment2_model_opt_comp_time.png")
plt.show()

temp = order_opt_df[order_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="num_pred",
                y="comp_time",
                hue="ablation",
                col='objective',
                row='NF',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 1+max(temp['comp_time']))
plt.savefig("figures/experiment2_order_opt_comp_time.png")
plt.show()

temp = model_opt_df[model_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="num_pred",
                y="work",
                hue="ablation",
                col='objective',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(0.00001, max(temp['work']))
plt.savefig("figures/experiment2_model_opt_work.png")
plt.show()

temp = order_opt_df[order_opt_df.output_flag==2]
g = sns.catplot(data=temp,
                x="num_pred",
                y="work",
                hue="ablation",
                col='objective',
                kind='box',
                palette='viridis')

g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(0.00001, 1+max(temp['work']))
plt.savefig("figures/experiment2_order_opt_work.png")
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
plt.savefig("figures/experiment2_model_opt_bug_report.png")
plt.show()

g = sns.countplot(
    data=order_opt_df,
    y="ablation",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.savefig("figures/experiment2_order_opt_bug_report.png")
plt.show()