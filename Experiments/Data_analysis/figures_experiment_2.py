import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../experiment_2.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type=="model_opt"]
order_opt_df = df[df.model_type=="order_opt"]

MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
 'weighted_min_max', 'exponential_weighted_criterion',
 'weighted_product', 'goal_method', 'bounded_objective']

# How often does bug or timeout occur?
g = sns.countplot(
    data=model_opt_df,
    y="MOO_method",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.subplots_adjust(left=0.36)
plt.title("Output_flag counts per MOO method for model_opt")
plt.savefig("figures/experiment2_model_opt_bug_report.png")
plt.show()

g = sns.countplot(
    data=order_opt_df,
    y="MOO_method",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.subplots_adjust(left=0.36)
plt.title("Output_flag counts per MOO method for order_opt")
plt.savefig("figures/experiment2_order_opt_bug_report.png")
plt.show()

# Which method is fastest?
g = sns.catplot(data=model_opt_df[model_opt_df.output_flag == 2],
                x="num_pred",
                y="comp_time",
                kind='box',
                col="MOO_method",
                col_wrap=3,
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 10**4)
plt.title("Computation time in ms for different MOO methods per number of predicates")
plt.savefig("figures/experiment2_model_opt_comp_time.png")
plt.show()

g = sns.catplot(data=order_opt_df[order_opt_df.output_flag == 2],
                x="num_pred",
                y="comp_time",
                kind='box',
                col="MOO_method",
                col_wrap=3,
                palette='viridis')
g.fig.get_axes()[0].set_yscale('log')
g.fig.get_axes()[0].set_ylim(1, 10**4)
plt.title("Computation time in ms for different MOO methods per number of predicates")
plt.savefig("figures/experiment2_order_opt_comp_time.png")
plt.show()
