import gurobipy as grb
from numpy import exp


def utopia_solution(model):
    model.compute_greedy_solution('accuracy', 'max')
    model.utopia_acc = model.opt_accuracy
    model.model.setObjective(model.model.getVarByName('total_cost'), grb.GRB.MINIMIZE)
    model.optimize()
    model.utopia_cost = model.opt_cost
    model.model.setObjective(model.model.getVarByName('total_memory'), grb.GRB.MINIMIZE)
    model.optimize()
    model.utopia_memory = model.opt_memory


def worst_solution(model):
    model.compute_greedy_solution('accuracy', 'min')
    model.worst_acc = model.opt_accuracy
    model.compute_greedy_solution('cost', 'max')
    model.worst_cost = model.opt_cost
    model.compute_greedy_solution('memory', 'max')
    model.worst_memory = model.opt_memory


def normalize_objectives(model):
    utopia_solution(model)
    worst_solution(model)
    acc_loss_norm = model.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='acc_loss_norm')
    model.model.addConstr(acc_loss_norm == (model.utopia_acc - model.model.getVarByName('total_accuracy') )/(model.utopia_acc-model.worst_acc))
    cost_norm = model.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='cost_norm')
    model.model.addConstr(cost_norm == (model.model.getVarByName('total_cost')-model.utopia_cost)/(model.worst_cost-model.utopia_cost))
    memory_norm = model.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='memory_norm')
    model.model.addConstr(memory_norm == (model.model.getVarByName('total_memory')-model.utopia_memory)/(model.worst_memory-model.utopia_memory))
    model.model.update()
    return model.model


def calculate_weights(objectives):
    weights = {}
    count = 1
    total = 0
    for objective_list in objectives:
        if type(objective_list) == list:
            for objective in objective_list:
                weights[objective] = count
                total += count
        else:
            weights[objective_list] = count
            total += count
        count += 1
    for key, value in weights.items():
        weights[key] = value/total
    return weights


def flatten_objectives(objectives):
    flat_objectives = []
    for item in objectives:
        if type(item) == list:
            for objective in item:
                flat_objectives.append(objective)
        else:
            flat_objectives.append(item)
    return flat_objectives


def set_MOO_method(model, method, objectives, p=0, weights=None, goals=None, lbs=None, ubs=None):
    model.model = normalize_objectives(model)
    if weights == None:
        weights = calculate_weights(objectives)
    objectives = flatten_objectives(objectives)
    if method == 'weighted_global_criterion':
        weighted_global_criterion(model, objectives, weights, p)
    elif method == 'weighted_sum':
        weighted_sum(model, objectives, weights)
    elif method == 'lexicographic':
        lexicographic_method(model, objectives)
    elif method == 'weighted_min_max':
        weighted_min_max(model, objectives, weights)
    elif method == 'exponential_weighted_criterion':
        exponential_weighted_criterion(model, objectives, weights, p)
    elif method == 'weighted_product':
        weighted_product(model, objectives, weights)
    elif method == 'goal_method':
        goal_method(model, objectives, goals)
    elif method == 'archimedean_goal_method':
        archimedean_goal_method(model, objectives, weights, goals)
    elif method == 'goal_attainment_method':
        goal_attainment_method(model, objectives, weights, goals)
    elif method == 'bounded_objective':
        bounded_objective(model, objectives, lbs, ubs)
    elif method == "greedy_method":
        greedy_MOO(model, weights)
    model.model.update()
    return model

def weighted_global_criterion(model, objectives, weights, p):
    temp_products = [[1]*len(objectives)]
    for i in range(p):
        temp_temp_product = model.model.addVars(len(objectives), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_temp_product')
        model.model.addConstrs(temp_temp_product[j] ==
                               temp_products[-1][j] *
                         model.model.getVarByName(objectives[j]) for j in range(len(objectives)))
        temp_products.append(temp_temp_product)
    goal = grb.quicksum(weights[objectives[i]] * temp_products[-1][i] for i in range(len(objectives)))
    model.model.setObjective(goal, grb.GRB.MINIMIZE)


def weighted_sum(model, objectives, weights):
    weighted_global_criterion(model, objectives, weights, p=1)


def lexicographic_method(model, ordering):
    ordering.reverse()
    for objective in ordering:
        model.model.setObjective(model.model.getVarByName(objective), grb.GRB.MINIMIZE)
        model.optimize(timeout=30*60)
        if model.model.Status == 2:
            bound = model.model.getVarByName(objective).X
            model.model.addConstr(model.model.getVarByName(objective) <= bound)
        else:
            break


def weighted_min_max(model, objectives, weights):
    # lambda is a protected name...
    labda = model.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='labda')
    # due to true normalization the utopia points are just 0
    model.model.addConstrs(weights[objectives[obj]] * model.model.getVarByName(objectives[obj]) <= labda for obj in range(len(objectives)))
    model.model.setObjective(labda, grb.GRB.MINIMIZE)


def exponential_weighted_criterion(model, objectives, weights, p):
    exp_var = model.model.addVars(len(objectives), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='exp_var')
    for obj in range(len(objectives)):
        model.model.addGenConstrExpA(model.model.getVarByName(objectives[obj]), exp_var[obj], exp(p))
    model.model.setObjective(grb.quicksum((exp(p*weights[objectives[i]])-1)*exp_var[i] for i in range(len(objectives))), grb.GRB.MINIMIZE)
    model.model.optimize()
    if model.model.Status == 3:
        model.model.params.FeasibilityTol = 1e-2
        model.model.params.IntFeasTol = 1e-1
        model.model.params.OptimalityTol = 1e-2
        model.model.feasRelaxS(1, False, False, True)
        model.model.optimize()


def weighted_product(model, objectives, weights):
    inv_norm_weights = [x*(1/min(weights.values())) for x in weights.values()]
    temp_product = [1]
    for obj in range(len(objectives)):
        for w in range(int(inv_norm_weights[obj])):
            temp_temp_product = model.model.addVar(lb=1, ub=1, vtype=grb.GRB.CONTINUOUS)
            model.model.addConstr(temp_temp_product == temp_product[-1] * model.model.getVarByName(objectives[obj]))
            temp_product.append(temp_temp_product)
    model.model.setObjective(temp_product[-1], grb.GRB.MINIMIZE)
    model.model.optimize()
    if model.model.Status == 3:
        model.model.params.FeasibilityTol = 1e-2
        model.model.params.IntFeasTol = 1e-1
        model.model.params.OptimalityTol = 1e-2
        model.model.feasRelaxS(1, False, False, True)
        model.model.optimize()


def goal_method(model, objectives, goals):
    d = model.model.addVars(len(objectives), 2, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='d')
    model.model.addConstrs(model.model.getVarByName(objectives[idx]) + d[idx, 0] - d[idx, 1] == goals[obj]
                     for idx,obj in enumerate(objectives))
    model.model.addConstrs(d[obj, 0] * d[obj, 1] == 0 for obj in range(len(objectives)))
    model.model.setObjective(d.sum('*', '*'), grb.GRB.MINIMIZE)

def archimedean_goal_method(model, objectives, weights, goals):
    d = model.model.addVars(len(objectives), 2, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='d')
    model.model.addConstrs(model.model.getVarByName(objectives[idx]) + d[idx, 0] - d[idx, 1] == goals[obj]
                           for idx,obj in enumerate(objectives))
    model.model.addConstrs(d[obj, 0] * d[obj, 1] == 0 for obj in range(len(objectives)))
    model.model.setObjective(grb.quicksum(weights[objectives[obj]]*(d[obj, 0] + d[obj, 1]) for obj in range(len(objectives))))

def goal_attainment_method(model, objectives, weights, goals):
    labda = model.model.addVar(vtype=grb.GRB.CONTINUOUS, name='labda')
    model.model.addConstrs(model.model.getVarByName(obj)
                            <= goals[obj] + weights[obj]*labda for obj in objectives)
    model.model.setObjective(labda, grb.GRB.MINIMIZE)


def bounded_objective(model, objectives, lbs, ubs):
    model.model.setObjective(model.model.getVarByName(objectives[-1]), grb.GRB.MINIMIZE)
    model.model.addConstrs(model.model.getVarByName(obj) <= ubs[obj] for obj in objectives[0:-1])
    model.model.addConstrs(model.model.getVarByName(obj) >= lbs[obj] for obj in objectives[0:-1])

def greedy_MOO(model, weights):
    flat_predicates = [item for sublist in model.predicates for item in sublist]
    max_C = max(model.C.values())
    max_D = max(model.D.values())
    models = list(model.C.keys())
    for p in range(len(flat_predicates)):
        min_val = 100000
        min_loc = 0
        min_A = min(model.A[flat_predicates[p]].values())
        for m in range(len(models)):
            val = weights['memory_norm'] * model.D[models[m]]/max_D +\
                  weights['acc_loss_norm']*(1-model.A[flat_predicates[p]][models[m]])/(1-min_A) + \
                  weights['cost_norm']*model.C[models[m]]/max_C
            if val < min_val and model.A[flat_predicates[p]][models[m]] > 0:
                min_val = val
                min_loc = m
        eps = 0.1
        model.model.getVarByName("B[{}]".format(min_loc)).lb = 1-eps
        for m in range(len(models)):
            if m == min_loc:
                model.X[m,p].lb=1-eps
                model.X[m,p].Start = 1
            else:
                #model.X[m, p].ub = 0+eps
                model.X[m,p].Start = 0
        model.model.optimize()
        if model.model.Status == 3:
            model.model.params.FeasibilityTol = 1e-2
            model.model.params.IntFeasTol = 1e-1
            model.model.params.OptimalityTol = 1e-2
            model.model.feasRelaxS(1, False, False, True)
            model.model.optimize()