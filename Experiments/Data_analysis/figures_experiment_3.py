import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../experiment_3-1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type=="model_opt"]
order_opt_df = df[df.model_type=="order_opt"]

MOO_methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
               'weighted_min_max', 'exponential_weighted_criterion',
               'weighted_product', 'goal_method', 'global_attainment_method',
               'bounded_objective']

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
plt.savefig("figures/experiment3_model_opt_bug_report.png")
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
plt.savefig("figures/experiment3_order_opt_bug_report.png")
plt.show()

# Which method is fastest?
g = sns.boxplot(data=model_opt_df[model_opt_df.output_flag == 2],
                y='MOO_method',
                x="comp_time",
                palette='viridis')
plt.subplots_adjust(left=0.36)
plt.savefig("figures/experiment3_model_opt_comp_time.png")
plt.show()

temp = order_opt_df[order_opt_df.output_flag == 2]
g = sns.boxplot(data=temp,
                y='MOO_method',
                x="comp_time",
                palette='viridis')
plt.subplots_adjust(left=0.36)
plt.savefig("figures/experiment3_order_opt_comp_time.png")
plt.show()
