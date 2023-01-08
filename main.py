from Models.ModelOpt import *
from Models.OrderOpt import *
from Models.MO_optimization import *
from data_loader import *
import matplotlib.pyplot as plt
from Query_tools import generate_queries
import timeit

filename = "model_stats_ap.csv"
A, C, D, sel = data_loader(filename)

# Dummy modelDB
# A = {'road': {'m1':0.9,'m2': 0.95, 'm3': 0, 'm4':0, 'm5':0, 'm6':0},
#     'person': {'m1':0,'m2': 0, 'm3': 0, 'm4':0, 'm5':0.94, 'm6':0.96},
#      'light': {'m1':0,'m2': 0, 'm3': 0, 'm4':0, 'm5':0.91, 'm6':0.95},
#      'car': {'m1':0,'m2': 0, 'm3': 0.91, 'm4':0.93, 'm5':0.0, 'm6':0.0}}
# C = {'m1':25, 'm2': 35, 'm3':20, 'm4':40, 'm5':5, 'm6':10}
# D = {'m1':10, 'm2': 20, 'm3':30, 'm4':40, 'm5':50, 'm6':0}
# sel = {'road': 0.1,
#        'person': 0.2,
#        'light': 0.3,
#        'car': 0.4}

new_eq = {'eq45': True,
          'accuracy': True,
          'eq14': False,
          'eq16': True,
          'memory': False}


# query needs to be in DNF: [[.. & .. &..] | [.. & ..] | [..]]
# query = [['banana', 'cat'], ['mouse', 'carrot']]
# predicates = [['chair'], ['laptop']]
# query = [['umbrella'], ['teddy_bear'], ['person'], ['suitcase', 'baseball_bat', 'dog'], ['laptop', 'traffic_light'], ['wine_glass', 'train'], ['hot_dog'], ['potted_plant']]

# query = [['car', 'person'], ['light', 'road']]
query = [['zebra', 'handbag'], ['parking_meter'], ['skis']]

# weights = {'acc_loss_norm': 0.5, 'cost_norm': 0.25, 'memory_norm': .25}
# objectives = ['memory_norm', 'cost_norm', 'acc_loss_norm']
# model = set_MOO_method(model, method='weighted_sum', objectives=objectives, weights=weights)
# model.optimize()
# print(model.opt_accuracy, model.opt_cost, model.opt_memory)
# print(model.model.Runtime, "seconds")

# print(model.utopia_acc)
# print(model.worst_acc)
# print(model.opt_accuracy)
# print(model.opt_cost)
# assignment, ordering = model.get_query_plan()
# print(assignment)
# bound = model.model.getVarByName('total_accuracy').x
# print('Total accuracy: ', model.opt_accuracy)
# print('Total cost: ', model.opt_cost)
# print('Total memory: ', model.opt_memory)

# print('Time taken: ', new_time)
#
accuracies, costs = [], []

eps = 0.0001
with grb.Env() as env:
    model = ModelOpt(A=A, C=C, D=D, Sel=sel, goal='cost', bound=0, predicates=query, NF="DNF", new_equations=new_eq, env=env)
    model.model.setParam(grb.GRB.Param.OutputFlag, 0)
    print("Generating accuracy points...")
    while model.model.solCount > 0:
        for solution in range(model.model.solCount):
            model.model.setParam(grb.GRB.Param.SolutionNumber, solution)
            accuracies.append(model.model.getVarByName('total_accuracy').Xn)
            costs.append(model.model.getVarByName('total_cost').Xn)
        print("current bound: ", model.model.objVal)
        print("new bound: ", model.model.objVal - eps)
        model.model.addConstr(model.model.getVarByName('total_accuracy') <= model.model.objVal - eps)
        model.optimize()
    print("Finished general solutions")

with grb.Env() as env:
    model = ModelOpt(A, C, D, sel, goal='cost', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
    model.model.update()
    model.optimize()
    model.model.setParam(grb.GRB.Param.OutputFlag, 0)
    min_cost = model.model.objVal
    eps = 1
    print("Generating cost points...")
    while model.model.solCount > 0:
        for solution in range(model.model.solCount):
            model.model.setParam(grb.GRB.Param.SolutionNumber, solution)
            accuracies.append(model.model.getVarByName('total_accuracy').Xn)
            costs.append(model.model.getVarByName('total_cost').Xn)
        model.model.addConstr(model.model.getVarByName('total_cost') >= model.model.objVal + eps)
        min_cost += eps
        model.optimize()

plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

objectives = ['acc_loss_norm','cost_norm']
weights = {'acc_loss_norm': 5/6,
           'cost_norm': 1/6}
p = 3
goals = {'acc_loss_norm':0.4,
         'cost_norm':0.3}
lbs, ubs = {'acc_loss_norm':0,'cost_norm':0}, {'acc_loss_norm':0.4,'cost_norm':0.5}

methods = ['weighted_global_criterion', 'weighted_sum', 'lexicographic',
           'weighted_min_max',
           #'exponential_weighted_criterion',
           #'weighted_product',
           'goal_method', 'bounded_objective',
           'greedy_MOO',
           # 'archimedean_goal_method', 'goal_attainment_method']
           ]

method_label = {'weighted_global_criterion':'Weighted global criterion',
                'weighted_sum': 'Weighted sum',
                'lexicographic': 'Lexicographic method',
                'weighted_min_max': 'Weighted min-max',
                'exponential_weighted_criterion': 'Exponential weighted criterion',
                'weighted_product': 'Weighted product',
                'goal_method': 'Goal method',
                'bounded_objective': 'Bounded objective',
                'greedy_MOO': 'Greedy MOO',
                'archimedean_goal_method': 'Archimedean goal method',
                'goal_attainment_method': 'Goal attainment method'
                }
colors = ['cornflowerblue', 'mediumaquamarine', 'lightcoral', 'plum', 'tan', 'lightseagreen', 'lightpink', 'blue']
arrows = ['<', '^', '>', 'v']

solutions = {}
y_min, y_max = min(costs), max(costs)
x_min, x_max = min(accuracies), max(accuracies)
deviation_x, deviation_y = (x_max-x_min)/15, (y_max-y_min)/8
for idx, method in enumerate(methods):
    with grb.Env() as env:
        model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
        set_MOO_method(model, method=method, objectives=objectives, weights=weights, p=p, goals=goals, lbs=lbs, ubs=ubs)
        model.optimize()
        accuracy, cost = model.opt_accuracy, model.opt_cost
        point = (round(accuracy, 5), round(cost, 0))
        print(method, accuracy, cost)
    if point in solutions.keys():
        # box = ax.get_position()
        if solutions[point] == 1:
            x, y = deviation_x, 0
        if solutions[point] == 2:
            x, y = 0, -deviation_y
        if solutions[point] == 3:
            x, y = -deviation_x, 0
        if solutions[point] == 4:
            x, y = 0, deviation_y
        plt.plot(accuracy + x/2, cost + y/2,
                marker=arrows[(solutions[point]-1)%4],
                markersize=4, color='black')
        plt.plot(accuracy+x, cost+y, '*', color=colors[idx], markersize=10, label=method_label[method])
        solutions[point] += 1
    else:
        solutions[point] = 1
        plt.plot(accuracy, cost, '*', color=colors[idx], markersize=10, label=method_label[method])

plt.xlabel(r'Accuracy ($f_{acc}$)')
plt.ylabel(r'Execution cost ($f_{cost}$)')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Experiments/figures/MOO_methods")
plt.show()

weights = [{'acc_loss_norm': 1/10, 'cost_norm': 9/10},
           {'acc_loss_norm': 5/10, 'cost_norm': 5/10},
           {'acc_loss_norm': 9/10, 'cost_norm': 1/10},
]
p = [2,4]
goals = [{'acc_loss_norm':0.0, 'cost_norm':0.0},
         {'acc_loss_norm':0.1, 'cost_norm':0.3},
         {'acc_loss_norm':0.3, 'cost_norm':0.1}
         ]
lbs = [{'acc_loss_norm':0.0,'cost_norm':0.0},
       {'acc_loss_norm':0.1,'cost_norm':0.3},
       {'acc_loss_norm':0.3,'cost_norm':0.1},
       {'acc_loss_norm':0.5,'cost_norm':0.5}
]
ubs = [{'acc_loss_norm':0.4,'cost_norm':0.4},
        {'acc_loss_norm':0.3,'cost_norm':0.5},
       {'acc_loss_norm':0.5,'cost_norm':0.3},
       {'acc_loss_norm':0.7,'cost_norm':0.7}
       ]

objectives = [['acc_loss_norm', 'cost_norm'], ['cost_norm', 'acc_loss_norm']]
y_min, y_max = min(costs), max(costs)
x_min, x_max = min(accuracies), max(accuracies)
deviation_x, deviation_y = (x_max-x_min)/15, (y_max-y_min)/8
for method in ['weighted_sum', 'weighted_min_max', 'weighted_product']:
    solutions = {}
    for i in range(0, 3):
        with grb.Env() as env:
            model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
            set_MOO_method(model, method=method, objectives=objectives[i%2], weights=weights[i], p=p[i%2], goals=goals[i], lbs=lbs[i], ubs=ubs[i])
            model.optimize()
            accuracy, cost = model.opt_accuracy, model.opt_cost
            point = (round(accuracy, 5), round(cost, 0))
        if point in solutions.keys():
            # box = ax.get_position()
            if solutions[point] == 1:
                x, y = deviation_x, 0
            if solutions[point] == 2:
                x, y = 0, -deviation_y
            if solutions[point] == 3:
                x, y = -deviation_x, 0
            if solutions[point] == 4:
                x, y = 0, deviation_y
            plt.plot(accuracy + x/2, cost + y/2,
                     marker=arrows[(solutions[point]-1)%4],
                     markersize=4, color='black')
            plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="Weights are {} for accuracy and cost".format(list(weights[i].values())))
            solutions[point] += 1
        else:
            solutions[point] = 1
            plt.plot(accuracy, cost, '*', color=colors[i], markersize=10,label="Weights are {} for accuracy and cost".format(list(weights[i].values())))

    plt.xlabel('Accuracy')
    plt.ylabel('Execution cost')
    plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Experiments/figures/MOO_methods_{}".format(method))
    plt.show()



for method in ['weighted_global_criterion', 'exponential_weighted_criterion']:
    solutions = {}
    for i in range(0, 3):
        with grb.Env() as env:
            model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
            set_MOO_method(model, method=method, objectives=objectives[i%2], weights=weights[i], p=p[i%2], goals=goals[i], lbs=lbs[i], ubs=ubs[i])
            model.optimize()
            accuracy, cost = model.opt_accuracy, model.opt_cost
            point = (round(accuracy, 5), round(cost, 0))
        if point in solutions.keys():
            # box = ax.get_position()
            if solutions[point] == 1:
                x, y = deviation_x, 0
            if solutions[point] == 2:
                x, y = 0, -deviation_y
            if solutions[point] == 3:
                x, y = -deviation_x, 0
            if solutions[point] == 4:
                x, y = 0, deviation_y
            plt.plot(accuracy + x/2, cost + y/2,
                     marker=arrows[(solutions[point]-1)%4],
                     markersize=4, color='black')
            plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="p={}, weights are {} for accuracy and cost".format(p[i%2], list(weights[i].values())))
            solutions[point] += 1
        else:
            solutions[point] = 1
            plt.plot(accuracy, cost, '*', color=colors[i], markersize=10,label="p={}, weights are {} for accuracy and cost".format(p[i%2], list(weights[i].values())))

    plt.xlabel('Accuracy')
    plt.ylabel('Execution cost')
    plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Experiments/figures/MOO_methods_{}".format(method))
    plt.show()


solutions = {}
objective_labels = {"acc_loss_norm": "accuracy", "cost_norm": "cost"}
for i in range(0, 2):
    with grb.Env() as env:
        model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
        set_MOO_method(model, method='lexicographic', objectives=objectives[i%2], weights=weights[i%2], p=p[i%2], goals=goals[i], lbs=lbs[i], ubs=ubs[i])
        model.optimize()
        accuracy, cost = model.opt_accuracy, model.opt_cost
        point = (round(accuracy, 5), round(cost, 0))
    if point in solutions.keys():
        # box = ax.get_position()
        if solutions[point] == 1:
            x, y = deviation_x, 0
        if solutions[point] == 2:
            x, y = 0, -deviation_y
        if solutions[point] == 3:
            x, y = -deviation_x, 0
        if solutions[point] == 4:
            x, y = 0, deviation_y
        plt.plot(accuracy + x/2, cost + y/2,
                 marker=arrows[(solutions[point]-1)%4],
                 markersize=4, color='black')
        plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="Hierarchy is {}, then {}".format(objective_labels[objectives[i][-1]], objective_labels[objectives[i][-2]]))
        solutions[point] += 1
    else:
        solutions[point] = 1
        plt.plot(accuracy, cost, '*', color=colors[i], markersize=10,label="Hierarchy is {}, then {}".format(objective_labels[objectives[i][-1]], objective_labels[objectives[i][-2]]))

plt.xlabel('Accuracy')
plt.ylabel('Execution cost')
plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Experiments/figures/MOO_methods_{}".format('lexicographic'))
plt.show()


solutions = {}
for i in range(0, 3):
    with grb.Env() as env:
        model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
        set_MOO_method(model, method='goal_method', objectives=objectives[i%2], weights=weights[i%2], p=p[i%2], goals=goals[i], lbs=lbs[i], ubs=ubs[i])
        model.optimize()
        accuracy, cost = model.opt_accuracy, model.opt_cost
        point = (round(accuracy, 5), round(cost, 0))
    if point in solutions.keys():
        # box = ax.get_position()
        if solutions[point] == 1:
            x, y = deviation_x, 0
        if solutions[point] == 2:
            x, y = 0, -deviation_y
        if solutions[point] == 3:
            x, y = -deviation_x, 0
        if solutions[point] == 4:
            x, y = 0, deviation_y
        plt.plot(accuracy + x/2, cost + y/2,
                 marker=arrows[(solutions[point]-1)%4],
                 markersize=4, color='black')
        plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="Goals are {}".format(list(goals[i].values())))
        solutions[point] += 1
    else:
        solutions[point] = 1
        plt.plot(accuracy, cost, '*', color=colors[i], markersize=10, label="Goals are {}".format(list(goals[i].values())))

plt.xlabel('Accuracy')
plt.ylabel('Execution cost')
plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Experiments/figures/MOO_methods_{}".format('goal_method'))
plt.show()


for method in ['goal_attainment_method', 'archimedean_goal_method']:
    solutions = {}
    for i in range(0, 3):
        with grb.Env() as env:
            model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
            set_MOO_method(model, method=method, objectives=objectives[i%2], weights=weights[i%2], p=p[i%2], goals=goals[i], lbs=lbs[i], ubs=ubs[i])
            model.optimize()
            accuracy, cost = model.opt_accuracy, model.opt_cost
            point = (round(accuracy, 5), round(cost, 0))
        if point in solutions.keys():
            # box = ax.get_position()
            if solutions[point] == 1:
                x, y = deviation_x, 0
            if solutions[point] == 2:
                x, y = 0, -deviation_y
            if solutions[point] == 3:
                x, y = -deviation_x, 0
            if solutions[point] == 4:
                x, y = 0, deviation_y
            plt.plot(accuracy + x/2, cost + y/2,
                     marker=arrows[(solutions[point]-1)%4],
                     markersize=4, color='black')
            plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="Goals are {}".format(list(goals[i].values()))+
                                                                                    ", weights are {} for accuracy and cost".format(list(weights[i].values())))
            solutions[point] += 1
        else:
            solutions[point] = 1
            plt.plot(accuracy, cost, '*', color=colors[i], markersize=10, label="Goals are {}".format(list(goals[i].values()))+
                                                                                ", weights are {} for accuracy and cost".format(list(weights[i].values())))

    plt.xlabel('Accuracy')
    plt.ylabel('Execution cost')
    plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Experiments/figures/MOO_methods_{}".format(method))
    plt.show()



solutions = {}
for i in range(0, 4):
    with grb.Env() as env:
        model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
        set_MOO_method(model, method='bounded_objective', objectives=objectives[i%2], lbs=lbs[i], ubs=ubs[i])
        model.optimize()
        accuracy, cost = model.opt_accuracy, model.opt_cost
        point = (round(accuracy, 5), round(cost, 0))
    if point in solutions.keys():
        # box = ax.get_position()
        if solutions[point] == 1:
            x, y = deviation_x, 0
        if solutions[point] == 2:
            x, y = 0, -deviation_y
        if solutions[point] == 3:
            x, y = -deviation_x, 0
        if solutions[point] == 4:
            x, y = 0, deviation_y
        plt.plot(accuracy + x/2, cost + y/2,
                 marker=arrows[(solutions[point]-1)%4],
                 markersize=4, color='black')
        plt.plot(accuracy+x, cost+y, '*', color=colors[i], markersize=10, label="Objective is {}, {} bounds are {} - {}".format(objective_labels[objectives[i%2][-1]],objective_labels[objectives[i%2][-2]],
                                                                                                                          lbs[i][objectives[i%2][0]], ubs[i][objectives[i%2][0]]))
        solutions[point] += 1
    else:
        solutions[point] = 1
        plt.plot(accuracy, cost, '*', color=colors[i], markersize=10, label="Objective is {}, {} bounds are {} - {}".format(objective_labels[objectives[i%2][-1]], objective_labels[objectives[i%2][-2]],
                                                                                                                      lbs[i][objectives[i%2][0]], ubs[i][objectives[i%2][0]]))

plt.xlabel('Accuracy')
plt.ylabel('Execution cost')
plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Experiments/figures/MOO_methods_{}".format("bounded_objective"))
plt.show()


solutions = {}
with grb.Env() as env:
    model = ModelOpt(A, C, D, Sel=sel, goal='accuracy', bound=0.95, predicates=query, NF="DNF", new_equations=new_eq, env=env)
    set_MOO_method(model, method='greedy_MOO', objectives=objectives[0])
    model.optimize()
    accuracy, cost = model.opt_accuracy, model.opt_cost
    point = (round(accuracy, 5), round(cost, 0))
if point in solutions.keys():
    # box = ax.get_position()
    if solutions[point] == 1:
        x, y = deviation_x, 0
    if solutions[point] == 2:
        x, y = 0, -deviation_y
    if solutions[point] == 3:
        x, y = -deviation_x, 0
    if solutions[point] == 4:
        x, y = 0, deviation_y
    plt.plot(accuracy + x/2, cost + y/2,
             marker=arrows[(solutions[point]-1)%4],
             markersize=4, color='black')
    plt.plot(accuracy+x, cost+y, '*', color=colors[0], markersize=10, label="greedy MOO")
    solutions[point] += 1
else:
    solutions[point] = 1
    plt.plot(accuracy, cost, '*', color=colors[0], markersize=10, label="greedy MOO")

plt.xlabel('Accuracy')
plt.ylabel('Execution cost')
plt.plot(accuracies, costs, '.', color='grey', alpha=0.5, label='All solutions')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Experiments/figures/MOO_methods_{}".format("greedy_MOO"))
plt.show()
