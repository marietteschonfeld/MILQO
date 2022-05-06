from Generate_model import *
from MO_optimization import *

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

predicates = ['road', 'person', 'light', 'car']

model = generate_model(A, C, 'accuracy', predicates)
model.update()
acc_worst = 1
cost_worst = 35 + 10 + 40

# model = normalize_objectives(model, acc_worst, cost_worst)
print(bounded_objective(model, 0.93, 1, 'total_cost', 'total_accuracy'))

#
# vars = model.getVars()
# for m in range(len(C)):
#     print('B', m+1, model.getVarByName('B[{}]'.format(m)).x)
#     for p in range(len(predicates)):
#         print('X', m+1, model.getVarByName('X[{},{}]'.format(m, p)).x)
#     print()
