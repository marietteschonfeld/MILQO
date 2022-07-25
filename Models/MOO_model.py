from Models.Model import Model
from Models.ModelOpt import ModelOpt
from Models.OrderOpt import OrderOpt
import gurobipy as grb
from numpy import exp


class MOO_model(Model):
    def __init__(self, A, C, sel, model_type, goal, bound, predicates, NF, new_equations):
        super().__init__(A, C, goal, bound, predicates, NF, new_equations)
        if model_type == 'ModelOpt':
            print("init?")
            temp = ModelOpt(A, C, goal, bound, predicates, NF, new_equations)
            temp.extend_model()
            self.model = temp.model.copy()
            self.X = temp.X.copy()
        elif model_type == 'OrderOpt':
            print("init?")
            temp = OrderOpt(A, C, sel, goal, bound, predicates, NF, new_equations)
            temp.extend_model()
            self.model = temp.model.copy()
            self.X = temp.X.copy()
        self.model.update()

    def utopia_solution(self):
        self.compute_greedy_solution('accuracy', 'max')
        self.utopia_acc = self.opt_accuracy
        self.compute_greedy_solution('cost', 'min')
        self.utopia_cost = self.opt_cost

    def worst_solution(self):
        self.compute_greedy_solution('accuracy', 'min')
        self.worst_acc = self.opt_accuracy
        self.compute_greedy_solution('cost', 'max')
        self.worst_cost = self.opt_cost

    # TODO this normalization needs to be thought about carefully
    def normalize_objectives(self):
        self.utopia_solution()
        self.worst_solution()
        acc_loss_norm = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='acc_loss_norm')
        self.model.addConstr(acc_loss_norm == (self.model.getVarByName('accuracy_loss')-1+self.utopia_acc)/(self.worst_acc-1+self.utopia_acc))
        cost_norm = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='cost_norm')
        self.model.addConstr(cost_norm == (self.model.getVarByName('total_cost')-self.utopia_cost)/(self.worst_cost-self.utopia_cost))
        self.model.update()

    def set_MOO_method(self, method, objectives, weights, p, goals, lbs, ubs):
        if method == 'weighted_global_criterion':
            self.weighted_global_criterion(objectives, weights, p)
        elif method == 'weighted_sum':
            self.weighted_sum(objectives, weights)
        elif method == 'lexicographic':
            self.lexicographic_method(objectives)
        elif method == 'weighted_min_max':
            self.weighted_min_max(objectives, weights)
        elif method == 'exponential_weighted_criterion':
            self.exponential_weighted_criterion(objectives, weights, p)
        elif method == 'weighted_product':
            self.weighted_product(objectives, weights)
        elif method == 'goal_method':
            self.goal_method(objectives, goals)
        elif method == 'bounded_objective':
            self.bounded_objective(objectives, lbs, ubs)
        self.model.update()
        self.optimize()

    def weighted_global_criterion(self, objectives, weights, p):
        temp_products = [[1]*len(objectives)]
        for i in range(p):
            temp_temp_product = self.model.addVars(len(objectives), vtype=grb.GRB.CONTINUOUS, name='temp_temp_product')
            self.model.addConstrs(temp_temp_product[j] == temp_products[-1][j] *
                             self.model.getVarByName(objectives[j]) for j in range(len(objectives)))
            temp_products.append([temp_temp_product[0], temp_temp_product[1]])
        goal = grb.quicksum(weights[i] * temp_products[-1][i] for i in range(len(objectives)))
        self.model.setObjective(goal, grb.GRB.MINIMIZE)

    def weighted_sum(self, objectives, weights):
        self.weighted_global_criterion(objectives, weights, p=1)

    def lexicographic_method(self, ordering):
        for objective in ordering:
            self.model.setObjective(self.model.getVarByName(objective), grb.GRB.MINIMIZE)
            self.model.optimize()
            bound = self.model.getVarByName(objective).x
            self.model.addConstr(self.model.getVarByName(objective) <= bound)

    def weighted_min_max(self, objectives, weights):
        # lambda is a protected name...
        labda = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='labda')
        # due to true normalization the utopia points are just 1
        self.model.addConstrs(weights[obj] * self.model.getVarByName(objectives[obj]) <= labda for obj in range(len(objectives)))
        self.model.setObjective(labda, grb.GRB.MINIMIZE)

    def exponential_weighted_criterion(self, objectives, weights, p):
        exp_var = self.model.addVars(len(objectives), vtype=grb.GRB.CONTINUOUS, name='exp_var')
        for obj in range(len(objectives)):
            self.model.addGenConstrExpA(self.model.getVarByName(objectives[obj]), exp_var[obj], exp(p))
        self.model.setObjective(grb.quicksum((exp(p*weights[i])-1)*exp_var[i] for i in range(len(objectives))), grb.GRB.MINIMIZE)

    def weighted_product(self, objectives, weights):
        inv_norm_weights = [x*(1/min(weights)) for x in weights]
        temp_product = [1]
        for obj in range(len(objectives)):
            for w in range(int(inv_norm_weights[obj])):
                temp_temp_product = self.model.addVar(vtype=grb.GRB.CONTINUOUS)
                self.model.addConstr(temp_temp_product == temp_product[-1] * self.model.getVarByName(objectives[obj]))
                temp_product.append(temp_temp_product)
        self.model.setObjective(temp_product[-1], grb.GRB.MINIMIZE)

    def goal_method(self, objectives, goals):
        d = self.model.addVars(len(objectives), 2, vtype=grb.GRB.CONTINUOUS, name='d')
        self.model.addConstrs(self.model.getVarByName(objectives[obj]) + d[obj, 0] - d[obj, 1] == goals[obj]
                         for obj in range(len(objectives)))
        self.model.addConstrs(d[obj, 0] * d[obj, 1] == 0 for obj in range(len(objectives)))
        self.model.setObjective(d.sum('*', '*'), grb.GRB.MINIMIZE)

    def bounded_objective(self, objectives, lbs, ubs):
        self.model.setObjective(self.model.getVarByName(objectives[0]), grb.GRB.MINIMIZE)
        self.model.addConstrs(self.model.getVarByName(objectives[obj+1]) <= ubs[obj] for obj in range(len(objectives)-1))
        self.model.addConstrs(self.model.getVarByName(objectives[obj+1]) >= lbs[obj] for obj in range(len(objectives)-1))
