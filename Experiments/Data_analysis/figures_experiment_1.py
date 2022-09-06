import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../experiment_1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type == "model_opt"]
order_opt_df = df[df.model_type == "order_opt"]

temp = model_opt_df[(model_opt_df.ablation == 0) | (model_opt_df.ablation == 6)]
g = sns.catplot(data=temp[temp.output_flag==2],
                x="num_pred",
                y="comp_time",
                hue="ablation",
                col='objective',
                row='NF',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 10**6)
plt.savefig("figures/experiment1_model_opt_comp_time.png")
plt.show()

temp = order_opt_df[(order_opt_df.ablation == 0) | (order_opt_df.ablation == 6)]
g = sns.catplot(data=temp[temp.output_flag==2],
                x="num_pred",
                y="comp_time",
                hue="ablation",
                col='objective',
                row='NF',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, max(order_opt_df['comp_time']))
plt.savefig("figures/experiment1_order_opt_comp_time.png")
plt.show()

temp = model_opt_df[(model_opt_df.num_pred == 8)]
g = sns.catplot(data=temp[temp.output_flag==2],
                x="ablation",
                y="comp_time",
                hue="NF",
                col='objective',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 10**6)
plt.savefig("figures/experiment1_model_opt_ablations.png")
plt.show()

temp = order_opt_df[(order_opt_df.num_pred == 8)]
g = sns.catplot(data=temp[temp.output_flag==2],
                x="ablation",
                y="comp_time",
                hue="NF",
                col='objective',
                kind='box',
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, max(temp['comp_time']))
plt.savefig("figures/experiment1_order_opt_ablations.png")
plt.show()