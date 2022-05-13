import gurobipy as grb
from math import exp


def utopia_solution(model):
    model.setObjective(model.getVarByName('total_accuracy'), grb.GRB.MAXIMIZE)
    model.optimize()
    accuracy = model.getObjective().getValue()
    model.setObjective(model.getVarByName('total_cost'), grb.GRB.MINIMIZE)
    model.optimize()
    cost = model.getObjective().getValue()
    print('utopia: ', accuracy, cost)
    return accuracy, cost


# TODO this normalization needs to be thought about carefully
def normalize_objectives(model, acc_worst, cost_worst):
    acc_utopia, cost_utopia = utopia_solution(model)
    acc_loss_norm = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='acc_loss_norm')
    model.addConstr(acc_loss_norm == (model.getVarByName('accuracy_loss')-1+acc_utopia)/(acc_worst-1+acc_utopia))
    cost_norm = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='cost_norm')
    model.addConstr(cost_norm == (model.getVarByName('total_cost')-cost_utopia)/(cost_worst-cost_utopia))
    model.update()
    return model


# TODO fix the error to do with p
def weighted_global_criterion(model, weights, p):
    temp_products = [[1, 1]]
    objectives = ['acc_loss_norm', 'cost_norm']
    for i in range(p):
        temp_temp_product = model.addVars(2, vtype=grb.GRB.CONTINUOUS, name='temp_temp_product')
        model.addConstrs(temp_temp_product[j] == temp_products[-1][j] *
                         model.getVarByName(objectives[j]) for j in range(2))
        temp_products.append([temp_temp_product[0], temp_temp_product[1]])
    goal = weights[0]*temp_products[-1][0] + weights[1]*temp_products[-1][1]
    model.setObjective(goal, grb.GRB.MINIMIZE)
    model.update()
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_sum(model, weights):
    return weighted_global_criterion(model, weights, p=1)


def lexicographic_method(model, ordering):
    for objective in ordering:
        model.setObjective(model.getVarByName(objective), grb.GRB.MINIMIZE)
        model.optimize()
        bound = model.getVarByName(objective).x
        model.addConstr(model.getVarByName(objective) <= bound)
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_min_max(model, weights):
    # lambda is a protected name...
    l = model.addVar(vtype=grb.GRB.CONTINUOUS, name='l')
    # due to true normalization the utopia points are just 1
    model.addConstr(weights[0]*(model.getVarByName('acc_loss_norm')) <= l)
    model.addConstr(weights[1]*(model.getVarByName('cost_norm')) <= l)
    model.setObjective(l, grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def exponential_weighted_criterion(model, weights, p):
    objectives = ['acc_loss_norm', 'cost_norm']
    exp_var = model.addVars(len(objectives), vtype=grb.GRB.CONTINUOUS, name='exp_var')
    for obj in range(len(objectives)):
        model.addGenConstrExpA(model.getVarByName(objectives[obj]), exp_var[obj], exp(p))
    model.setObjective(grb.quicksum((exp(p*weights[i])-1)*exp_var[i] for i in range(len(objectives))), grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_product(model, weights):
    inv_norm_weights = [x*(1/min(weights)) for x in weights]
    temp_product = [1]
    objectives = ['acc_loss_norm', 'cost_norm']
    for obj in range(len(objectives)):
        for w in range(int(inv_norm_weights[obj])):
            temp_temp_product = model.addVar(vtype=grb.GRB.CONTINUOUS)
            model.addConstr(temp_temp_product == temp_product[-1] * model.getVarByName(objectives[obj]))
            temp_product.append(temp_temp_product)
    model.setObjective(temp_product[-1], grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def goal_method(model, goals):
    d = model.addVars(len(goals), 2, vtype=grb.GRB.CONTINUOUS, name='d')
    model.addConstr(model.getVarByName('total_accuracy') + d[0, 0] - d[0, 1] == goals[0])
    model.addConstr(model.getVarByName('total_cost') + d[1, 0] - d[1, 1] == goals[1])
    model.addConstr(d[0, 0] * d[0, 1] == 0)
    model.addConstr(d[1, 0] * d[1, 1] == 0)
    model.setObjective(d.sum('*', '*'), grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def bounded_objective(model, lb, ub, objective1, objective2):
    model.setObjective(model.getVarByName(objective1), grb.GRB.MINIMIZE)
    model.addConstr(model.getVarByName(objective2) <= ub)
    model.addConstr(model.getVarByName(objective2) >= lb)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x
