from Models.model_opt import *
from Models.order_opt import *
from data_loader import *
from Query_tools import generate_queries
import timeit

filename = "model_stats_ap.csv"
A, C, sel = data_loader(filename)

# # Dummy modelDB
# A = {'road': [0.9, 0.95, 0, 0, 0, 0],
#      'person': [0, 0., 0, 0, 0.94, 0.96],
#      'light': [0, 0., 0, 0, 0.91, 0.95],
#      'car': [0, 0., 0.91, 0.93, 0, 0]}
# C = [25, 35, 20, 40, 5, 10]
# sel = {'road': 0.1,
#        'person': 0.2,
#        'light': 0.3,
#        'car': 0.4}

# query needs to be in DNF: [[.. & .. &..] | [.. & ..] | [..]]
# predicates = [['banana', 'cat'], ['mouse', 'carrot']]
# predicates = [['chair'], ['laptop']]
query = [['umbrella'], ['teddy_bear'], ['person'], ['suitcase', 'baseball_bat', 'dog'], ['laptop', 'traffic_light'], ['wine_glass', 'train'], ['hot_dog'], ['potted_plant']]
flat_predicates = [item for sublist in query for item in sublist]

new_eq = {'eq45': True,
          'accuracy': True,
          'eq13': True,
          'eq14': True,
          'eq16': False}
old_eq = {'eq45': False,
          'accuracy': False,
          'eq13': False,
          'eq14': False,
          'eq16': False}

query = generate_queries(4, 1, A)[0]
print(query)

flat_predicates = [item for sublist in query for item in sublist]

model = order_opt(A=A, C=C, Sel=sel, goal='accuracy', bound=0.99, predicates=query, NF="DNF", new_equations=new_eq)
new_time = timeit.timeit('model.optimize()', globals=globals(), number=1)
bound = model.model.getVarByName('total_accuracy').x
print('Total accuracy: ', model.model.getVarByName('total_accuracy').x)
print('Total cost: ', model.model.getVarByName('total_cost').x)
print('Time taken: ', new_time)

# accuracies = []
# costs = []
#
# eps = 0.001
# model.setParam(grb.GRB.Param.OutputFlag, 0)
# while model.solCount > 0:
#     for solution in range(model.solCount):
#         model.setParam(grb.GRB.Param.SolutionNumber, solution)
#         accuracies.append(model.getVarByName('total_accuracy').Xn)
#         costs.append(model.getVarByName('total_cost').Xn)
#     model.addConstr(model.getVarByName('total_accuracy') <= model.objVal - eps)
#     model.optimize()
#
# model = model_opt(A, C, 'cost', predicates)
# model.update()
# model.optimize()
# model.setParam(grb.GRB.Param.OutputFlag, 0)
# min_cost = model.objVal
# eps = 0.5
# while model.solCount > 0:
#     for solution in range(model.solCount):
#         model.setParam(grb.GRB.Param.SolutionNumber, solution)
#         accuracies.append(model.getVarByName('total_accuracy').Xn)
#         costs.append(model.getVarByName('total_cost').Xn)
#     model.addConstr(model.getVarByName('total_cost') >= min_cost + eps)
#     min_cost += eps
#     model.optimize()
#
# fig = plt.figure()
# ax = plt.subplot()
#
# ax.plot(accuracies, costs,  '.', color='grey', alpha=0.5, label='all solutions')
#
# objectives = ['acc_loss_norm', 'cost_norm']
# weights = [0.25, 0.75]
# p = 3
# goals = [0.02, 0.8]
# lbs, ubs = [0.02], [0.6]
#
# methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
#            'weighted_min_max', 'exponential_weighted_criterion',
#            'weighted_product', 'goal_method', 'bounded_objective']
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan']
# arrows = ['<', '^', '>', 'V']
#
# solutions = {}
# axes = plt.gca()
# y_min, y_max = min(costs), max(costs)
# x_min, x_max = min(accuracies), max(accuracies)
# deviation_x, deviation_y = (x_max-x_min)/15, (y_max-y_min)/8
# for method in methods:
#     model = model_opt(A, C, 'accuracy', predicates)
#     model.update()
#     model = normalize_objectives(model)
#     accuracy, cost = MOO_method(model, method, objectives, weights, p, goals, lbs, ubs)
#     point = (round(accuracy, 5), round(cost, 0))
#     new_point = [round(accuracy, 5), round(cost, 0)]
#     print(method, accuracy, cost)
#     if point in solutions.keys():
#         box = ax.get_position()
#         if solutions[point] == 1:
#             x, y = deviation_x, 0
#         if solutions[point] == 2:
#             x, y = 0, -deviation_y
#         if solutions[point] == 3:
#             x, y = -deviation_x, 0
#         if solutions[point] == 4:
#             x, y = 0, deviation_y
#         ax.plot(accuracy + x/2, cost + y/2,
#                 marker=arrows[solutions[point]-1],
#                 markersize=4, color='black')
#         ax.plot(accuracy+x, cost+y, '*', color=colors.pop(0), label=method)
#         solutions[point] += 1
#     else:
#         solutions[point] = 1
#         ax.plot(accuracy, cost, '*', color=colors.pop(0), label=method)
#
# plt.title('Various MO optimization methods with prior preferences on a simple query')
# plt.xlabel('Accuracy')
# plt.ylabel('Execution cost')
#
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
# plt.legend(loc='lower center', bbox_to_anchor=(0.475, -0.64), ncol=2)
#
# plt.show()
#
