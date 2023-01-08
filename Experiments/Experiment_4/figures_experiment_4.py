import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from math import pi

df = pd.read_csv("experiment_4_1.csv")

df.rename(columns={"Unnamed: 0": "experiment_num"})

model_opt_df = df[df.model_type == "ModelOpt"]
order_opt_df = df[df.model_type == "OrderOpt"]

g = sns.countplot(
    data=model_opt_df,
    y="ablation",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.savefig("../figures/experiment4_model_opt_bug_report.png")
plt.show()
g = sns.countplot(
    data=order_opt_df,
    y="ablation",
    hue="output_flag",
    orient="v",
    dodge=True,
    palette="viridis"
)
plt.savefig("../figures/experiment4_order_opt_bug_report.png")
plt.show()

# number of variable
categories=["acc_loss_norm", "cost_norm", "memory_norm"]
N = len(categories)

# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25,0.5,0.75], ["0.25","0.5","0.75"], color="grey", size=7)
plt.ylim(0,1)

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()