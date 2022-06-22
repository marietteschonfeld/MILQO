from model_opt import *
from order_opt import *
from data_loader import *
import timeit
import matplotlib.pyplot as plt
import random
import seaborn as sb
from tools import generate_queries

filename = "model_stats_ap.csv"
A, C, sel = data_loader(filename)

def func(model):
    model.reset(1)
    model.optimize()
    return model


ablations = [
    # {'eq14': True, 'accuracy': True, 'eq13': False},
    # {'eq14': False, 'accuracy': True, 'eq13': False},
    # {'eq14': True, 'accuracy': False,  'eq13': False},
    # {'eq14': False, 'accuracy': False, 'eq13': False},
    {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': True},
    # {'eq14': True, 'accuracy': True, 'eq13': True, 'eq16': False},
    # {'eq14': False, 'accuracy': True, 'eq13': True},
    # {'eq14': True, 'accuracy': False,  'eq13': True},
    # {'eq14': False, 'accuracy': False, 'eq13': True}
]

num_predicates = 20
num_queries = 1
queries = []
for num_predicates in range(1, num_predicates+1):
    queries.append(generate_queries(num_predicates, num_queries, A))

times = []
query_bounds = {}
for ablation in ablations:
    times.append([])
    for num_pred in range(0, num_predicates):
        time_res = 0
        query_bounds[num_pred] = [0] * num_queries
        for query in range(0, num_queries):
            print(queries[num_pred][query])
            model = order_opt(A, C, sel, 'cost', query_bounds[num_pred][query], queries[num_pred][query], ablation)
            time_res += timeit.timeit('model.optimize()', globals=globals(), number=1)
            if ablation == ablations[0]:
                query_bounds[num_pred][query] = model.getVarByName('total_accuracy').x - 0.1
        times[-1].append(time_res*1000/num_queries)
    ax = sb.scatterplot(range(1, num_predicates+1), times[-1], label="Ablation {}".format(ablations.index(ablation)))
plt.legend(ncol=4, bbox_to_anchor=(0.5, -0.1), loc='upper center')
plt.title("Computation times for several queries with order_opt")
plt.xlabel("Amount of predicates")
plt.ylabel("Computation time in milliseconds")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.95])
#plt.yscale('log')
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(nrows=2, ncols=3)
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.grid(False)
# count = 0
# for row in ax:
#     for col in row:
#         query = query_list[count]
#         times_new = []
#         times_old = []
#         for ub in range(0, 100, 5):
#             model = order_opt(A, C, sel, 'cost', query)
#             model_old = order_opt_old(A, C, sel, 'cost', ub/100, query)
#             model.update()
#             model_old.update()
#             n=1
#             time_res_old = timeit.timeit('model_old.optimize()', globals=globals(), number=n)
#             time_res = timeit.timeit('model.optimize()', globals=globals(), number=n)
#             times_new.append(time_res*1000/n)
#             times_old.append(time_res_old*1000/n)
#         col.plot(range(0, 100, 5), times_new, '*')
#         col.plot(range(0, 100, 5), times_old, '+')
#         count += 1
# fig.suptitle("Computation times for several queries")
# plt.xlabel("Upper bound cost")
# plt.ylabel("Computation time in milliseconds")
# plt.show()

