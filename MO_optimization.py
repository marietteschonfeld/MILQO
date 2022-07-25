import gurobipy as grb
from math import exp


def utopia_solution(model):
    model.setObjective(model.getVarByName('total_accuracy'), grb.GRB.MAXIMIZE)
    model.optimize()
    accuracy = model.getObjective().getValue()
    model.setObjective(model.getVarByName('total_cost'), grb.GRB.MINIMIZE)
    model.optimize()
    cost = model.getObjective().getValue()
    return accuracy, cost


def worst_solution(model):
    model.setObjective(model.getVarByName('total_accuracy'), grb.GRB.MINIMIZE)
    model.update()
    model.optimize()
    acc_worst = model.getVarByName('accuracy_loss').x
    model.setObjective(model.getVarByName('total_cost'), grb.GRB.MAXIMIZE)
    model.update()
    model.optimize()
    cost_worst = model.getVarByName('total_cost').x
    return acc_worst, cost_worst


# TODO this normalization needs to be thought about carefully
def normalize_objectives(model):
    acc_utopia, cost_utopia = utopia_solution(model)
    print("utopia: ", acc_utopia, cost_utopia)
    acc_worst, cost_worst = worst_solution(model)
    print("worst: ", acc_worst, cost_worst)
    acc_loss_norm = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='acc_loss_norm')
    model.addConstr(acc_loss_norm == (model.getVarByName('accuracy_loss')-1+acc_utopia)/(acc_worst-1+acc_utopia))
    cost_norm = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='cost_norm')
    model.addConstr(cost_norm == (model.getVarByName('total_cost')-cost_utopia)/(cost_worst-cost_utopia))
    model.update()
    return model


def MOO_method(model, method, objectives, weights, p, goals, lbs, ubs):
    if method == 'weighted_global_criterion':
        return weighted_global_criterion(model, objectives, weights, p)
    elif method == 'weighted_sum':
        return weighted_sum(model, objectives, weights)
    elif method == 'lexicographic':
        return lexicographic_method(model, objectives)
    elif method == 'weighted_min_max':
        return weighted_min_max(model, objectives, weights)
    elif method == 'exponential_weighted_criterion':
        return exponential_weighted_criterion(model, objectives, weights, p)
    elif method == 'weighted_product':
        return weighted_product(model, objectives, weights)
    elif method == 'goal_method':
        return goal_method(model, objectives, goals)
    elif method == 'bounded_objective':
        return bounded_objective(model, objectives, lbs, ubs)


def weighted_global_criterion(model, objectives, weights, p):
    temp_products = [[1]*len(objectives)]
    for i in range(p):
        temp_temp_product = model.addVars(len(objectives), vtype=grb.GRB.CONTINUOUS, name='temp_temp_product')
        model.addConstrs(temp_temp_product[j] == temp_products[-1][j] *
                         model.getVarByName(objectives[j]) for j in range(len(objectives)))
        temp_products.append([temp_temp_product[0], temp_temp_product[1]])
    goal = grb.quicksum(weights[i] * temp_products[-1][i] for i in range(len(objectives)))
    model.setObjective(goal, grb.GRB.MINIMIZE)
    model.update()
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_sum(model, objectives, weights):
    return weighted_global_criterion(model, objectives, weights, p=1)


def lexicographic_method(model, ordering):
    for objective in ordering:
        model.setObjective(model.getVarByName(objective), grb.GRB.MINIMIZE)
        model.optimize()
        bound = model.getVarByName(objective).x
        model.addConstr(model.getVarByName(objective) <= bound)
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_min_max(model, objectives, weights):
    # lambda is a protected name...
    labda = model.addVar(vtype=grb.GRB.CONTINUOUS, name='labda')
    # due to true normalization the utopia points are just 1
    model.addConstrs(weights[obj] * model.getVarByName(objectives[obj]) <= labda for obj in range(len(objectives)))
    model.setObjective(labda, grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def exponential_weighted_criterion(model, objectives, weights, p):
    exp_var = model.addVars(len(objectives), vtype=grb.GRB.CONTINUOUS, name='exp_var')
    for obj in range(len(objectives)):
        model.addGenConstrExpA(model.getVarByName(objectives[obj]), exp_var[obj], exp(p))
    model.setObjective(grb.quicksum((exp(p*weights[i])-1)*exp_var[i] for i in range(len(objectives))), grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def weighted_product(model, objectives, weights):
    inv_norm_weights = [x*(1/min(weights)) for x in weights]
    temp_product = [1]
    for obj in range(len(objectives)):
        for w in range(int(inv_norm_weights[obj])):
            temp_temp_product = model.addVar(vtype=grb.GRB.CONTINUOUS)
            model.addConstr(temp_temp_product == temp_product[-1] * model.getVarByName(objectives[obj]))
            temp_product.append(temp_temp_product)
    model.setObjective(temp_product[-1], grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def goal_method(model, objectives, goals):
    d = model.addVars(len(objectives), 2, vtype=grb.GRB.CONTINUOUS, name='d')
    model.addConstrs(model.getVarByName(objectives[obj]) + d[obj, 0] - d[obj, 1] == goals[obj]
                     for obj in range(len(objectives)))
    model.addConstrs(d[obj, 0] * d[obj, 1] == 0 for obj in range(len(objectives)))
    model.setObjective(d.sum('*', '*'), grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x


def bounded_objective(model, objectives, lbs, ubs):
    model.setObjective(model.getVarByName(objectives[0]), grb.GRB.MINIMIZE)
    model.addConstrs(model.getVarByName(objectives[obj+1]) <= ubs[obj] for obj in range(len(objectives)-1))
    model.addConstrs(model.getVarByName(objectives[obj+1]) >= lbs[obj] for obj in range(len(objectives)-1))
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x
