from Generate_model import *
from MO_optimization import *
import matplotlib.pyplot as plt

A = {
    'road': [0.9, 0.95, 0, 0, 0, 0],
    'person': [0, 0, 0, 0, 0.94, 0.96],
    'light': [0, 0, 0, 0, 0.91, 0.95],
    'car': [0, 0, 0.91, 0.93, 0, 0],
}

inf = float("inf")
Costs = {
    'road': [25, 35, inf, inf, inf, inf],
    'person': [inf, inf, inf, inf, 5, 10],
    'light': [inf, inf, inf, inf, 5, 10],
    'car': [inf, inf, 20, 40, inf, inf],
}

C = [25, 35, 20, 40, 5, 10]

# query needs to be in DNF: [[.. & .. &] | [.. & ..] | [..]]
predicates = [['road', 'person'], ['car', 'light']]

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model.optimize()
# TODO this normalization is not ideal
cost_worst = 35 + 10 + 40
accuracies = []
costs = []

eps = 0.00001
model.setParam(grb.GRB.Param.OutputFlag, 0)
while model.solCount > 0:
    for solution in range(model.solCount):
        model.setParam(grb.GRB.Param.SolutionNumber, solution)
        accuracies.append(model.getVarByName('total_accuracy').Xn)
        costs.append(model.getVarByName('total_cost').Xn)
    model.addConstr(model.getVarByName('total_accuracy') <= model.objVal - eps)
    model.optimize()

model = generate_model(A, C, 'cost', predicates)
model.update()
model.optimize()
acc_worst = model.getVarByName('accuracy_loss').x
model.setParam(grb.GRB.Param.OutputFlag, 0)
eps=1
while model.solCount > 0:
    for solution in range(model.solCount):
        model.setParam(grb.GRB.Param.SolutionNumber, solution)
        accuracies.append(model.getVarByName('total_accuracy').Xn)
        costs.append(model.getVarByName('total_cost').Xn)
    model.addConstr(model.getVarByName('total_cost') >= model.objVal + eps)
    model.optimize()

fig = plt.figure()
ax = plt.subplot()

ax.plot(accuracies, costs,  '.', color='grey', label='all solutions')

weights = [0.15, 0.85]

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = weighted_global_criterion(model, weights, p=5)
ax.plot(cost, accuracy, '*', label='weighted criterion')

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = weighted_sum(model, weights)
ax.plot(cost, accuracy, '*', label='weighted sum')

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = lexicographic_method(model, ['accuracy_loss', 'total_cost'])
ax.plot(cost, accuracy, '*', label='lexicographic method')

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = weighted_min_max(model, weights)
ax.plot(cost, accuracy, '*', label='weighted min-max')

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = exponential_weighted_criterion(model, weights, p=2)
ax.plot(cost, accuracy, '*', label='exponential weighted criterion')

model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = weighted_product(model, weights)
ax.plot(cost, accuracy, '*', label='weighted product')

utopia = utopia_solution(model)
model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = goal_method(model, [0.98, 60])
ax.plot(cost, accuracy, '*', label='goal method')
#
model = generate_model(A, C, 'accuracy', predicates)
model.update()
model = normalize_objectives(model, acc_worst, cost_worst)
cost, accuracy = bounded_objective(model, 50, 60, 'accuracy_loss', 'total_cost')
ax.plot(cost, accuracy, '*', label='bounded objective method')

plt.title('Various MO optimization methods with prior preferences on a simple query')
plt.xlabel('Accuracy')
plt.ylabel('Execution cost')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height*0.8])
plt.legend(loc='lower center', bbox_to_anchor=(0.475, -0.425), ncol=2)

plt.show()
#
