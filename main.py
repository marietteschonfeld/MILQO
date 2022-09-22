from Models.ModelOpt import *
from Models.OrderOpt import *
from Models.MO_optimization import *
from data_loader import *
import matplotlib.pyplot as plt
from Query_tools import generate_queries
import timeit

filename = "model_stats_ap.csv"
A, C, D, sel = data_loader(filename)
print(D)

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

new_eq = {'eq45': True,
          'accuracy': True,
          'eq13': True,
          'eq14': True,
          'eq16': True}

temp_eq = {'eq45': False,
           'accuracy': True,
           'eq13': True,
           'eq14': True,
           'eq16': True}

old_eq = {'eq45': False,
          'accuracy': False,
          'eq13': False,
          'eq14': False,
          'eq16': False}

# query needs to be in DNF: [[.. & .. &..] | [.. & ..] | [..]]
# query = [['banana', 'cat'], ['mouse', 'carrot']]
# predicates = [['chair'], ['laptop']]
# query = [['umbrella'], ['teddy_bear'], ['person'], ['suitcase', 'baseball_bat', 'dog'], ['laptop', 'traffic_light'], ['wine_glass', 'train'], ['hot_dog'], ['potted_plant']]

query = generate_queries(8, 1, A)[0]
print(query)

flat_predicates = [item for sublist in query for item in sublist]

model = OrderOpt(A=A, C=C, D=D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq)
new_time = timeit.timeit('model.optimize()', globals=globals(), number=1)
plan, order = model.get_query_plan()
print(plan)
print(order)
# bound = model.model.getVarByName('total_accuracy').x
# print('Total accuracy: ', model.opt_accuracy)
# print('Total cost: ', model.opt_cost)
# print('Total memory: ', model.opt_memory)
#
# print('Time taken: ', new_time)
#
# accuracies, costs, memories = [], [], []
#
# eps = 0.01
# model.model.setParam(grb.GRB.Param.OutputFlag, 0)
# print("Generating accuracy points...")
# while model.model.solCount > 0:
#     for solution in range(model.model.solCount):
#         model.model.setParam(grb.GRB.Param.SolutionNumber, solution)
#         accuracies.append(model.model.getVarByName('total_accuracy').Xn)
#         costs.append(model.model.getVarByName('total_cost').Xn)
#         memories.append(model.model.getVarByName('total_memory').Xn)
#     print("current bound: ", model.model.objVal)
#     print("new bound: ", model.model.objVal - eps)
#     model.model.addConstr(model.model.getVarByName('total_accuracy') <= model.model.objVal - eps)
#     model.optimize()
# print("Finished general solutions")

# model = OrderOpt(A, C, sel, goal='cost', bound=0.95, predicates=query, NF="DNF", new_equations=temp_eq)
# model.model.update()
# model.optimize()
# model.model.setParam(grb.GRB.Param.OutputFlag, 0)
# min_cost = model.model.objVal
# eps = 1
# print("Generating cost points...")
# while model.model.solCount > 0:
#     for solution in range(model.model.solCount):
#         model.model.setParam(grb.GRB.Param.SolutionNumber, solution)
#         accuracies.append(model.model.getVarByName('total_accuracy').Xn)
#         costs.append(model.model.getVarByName('total_cost').Xn)
#     print("current bound: ", model.model.objVal)
#     model.model.addConstr(model.model.getVarByName('total_cost') >= model.model.objVal + eps)
#     min_cost += eps
#     model.optimize()

# fig = plt.figure()
# ax = plt.subplot(projection='3d')
#
# ax.plot(accuracies, costs, memories, '.', color='grey', alpha=0.5, label='all solutions')
#
# objectives = ['acc_loss_norm', 'cost_norm', 'fairness_norm', 'memory_norm']
# weights = [1/10, 2/10, 3/10, 4/10]
# p = 3
# goals = [0, 0, 0, 0]
# lbs, ubs = [0.0, 0.0, 0.0], [0.4, 0.5, 0.3]
#
# methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
#            'weighted_min_max', 'exponential_weighted_criterion',
#            'weighted_product', 'goal_method', 'bounded_objective']
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan']
# arrows = ['<', '^', '>', 'v']
#
# solutions = {}
# axes = plt.gca()
# z_min, z_max = min(memories), max(memories)
# y_min, y_max = min(costs), max(costs)
# x_min, x_max = min(accuracies), max(accuracies)
# deviation_x, deviation_y = (x_max-x_min)/15, (y_max-y_min)/8
# for method in methods:
#     model = ModelOpt(A, C, D, 'accuracy', 0.95, query, "DNF", new_eq)
#     set_MOO_method(model, method, objectives, weights, p, goals, lbs, ubs)
#     accuracy, cost, fairness, memory = model.opt_accuracy, model.opt_cost, model.opt_fairness, model.opt_memory
#     point = (round(accuracy, 5), round(cost, 0), round(fairness, 0), round(memory, 0))
#     print(method, accuracy, cost, fairness, memory)
#     if fairness == 0:
#         for m in range(len(C)):
#             for p in range(len(A.keys())):
#                 temp_var = model.model.getVarByName("X[{},{}]".format(m, p))
#                 print(temp_var)
#                 if temp_var.x > 0:
#                     print(temp_var)
#                     print(A[flat_predicates[p]][m], F[m, p])
#     ax.plot(accuracy, cost, memory, '*', color=colors.pop(0), label=method)
    # if point in solutions.keys():
    #     box = ax.get_position()
    #     if solutions[point] == 1:
    #         x, y = deviation_x, 0
    #     if solutions[point] == 2:
    #         x, y = 0, -deviation_y
    #     if solutions[point] == 3:
    #         x, y = -deviation_x, 0
    #     if solutions[point] == 4:
    #         x, y = 0, deviation_y
    #     ax.plot(accuracy + x/2, cost + y/2,
    #             marker=arrows[(solutions[point]-1)%4],
    #             markersize=4, color='black')
    #     ax.plot(accuracy+x, cost+y, memory, '*', color=colors.pop(0), label=method)
    #     solutions[point] += 1
    # else:
    #     solutions[point] = 1
    #     ax.plot(accuracy, cost, memory, '*', color=colors.pop(0), label=method)

# plt.title('Various MO optimization methods with prior preferences on a simple query')
# plt.xlabel('Accuracy')
# plt.ylabel('Execution cost')
# ax.set_zlabel('Memory')
#
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
# plt.legend(loc='lower center', bbox_to_anchor=(0.475, -0.64), ncol=2)
#
# plt.show()
