import gurobipy as grb


def utopia_solution(model):
    model.setObjective(model.getVarByName('total_accuracy'), grb.GRB.MAXIMIZE)
    model.optimize()
    accuracy = model.getObjective().getValue()
    model.setObjective(model.getVarByName('total_cost'), grb.GRB.MINIMIZE)
    model.optimize()
    cost = model.getObjective().getValue()
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
    goal = weights[0]*model.getVarByName('acc_loss_norm') + weights[1]*model.getVarByName('cost_norm')
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
    acc_utopia, cost_utopia = utopia_solution(model)
    # lambda is a protected name...
    l = model.addVar(vtype=grb.GRB.CONTINUOUS, name='l')
    model.addConstr(weights[0]*(model.getVarByName('accuracy_loss') - 1 + acc_utopia) <= l)
    model.addConstr(weights[1]*(model.getVarByName('total_cost') - cost_utopia) <= l)
    model.setObjective(l, grb.GRB.MINIMIZE)
    model.optimize()
    return model.getVarByName('total_accuracy').x, model.getVarByName('total_cost').x