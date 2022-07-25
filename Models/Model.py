import gurobipy as grb
from math import ceil
from Query_tools import terror_list, powerset


# Parent class for the general model
class Model:
    def __init__(self, A, C, goal, bound, predicates, NF, new_equations):
        self.A = A
        self.C = C
        self.goal = goal
        self.bound = bound
        self.predicates = predicates
        self.NF = NF
        self.new_equations = new_equations
        self.model = grb.Model(name="MILQO")
        self.opt_cost = 0
        self.opt_accuracy = 0
        self.generate_model()

    def generate_model(self):
        self.model.setParam(grb.GRB.Param.OutputFlag, 0)
        self.model.params.NonConvex = 2

        # self.model predicate as (x & y) | (w & z)
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        self.P = len(flat_predicates)
        self.M = len(self.C)

        self.X = self.model.addVars(self.M, self.P, vtype=grb.GRB.BINARY, name='X')

        self.model.addConstrs(self.X.sum('*', p) == 1 for p in range(self.P))

        if self.new_equations['accuracy']:
            self.model.addConstrs(self.X[m, p] <= ceil(self.A[flat_predicates[p]][m]) for m in range(self.M) for p in range(self.P))

        Accs = self.model.addVars(self.P, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Accs')
        self.model.addConstrs(Accs[p] == grb.quicksum(self.A[flat_predicates[p]][m]*self.X[m, p] for m in range(self.M)) for p in range(self.P))

        total_accuracy = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name='total_accuracy')
        sub_predicate_acc = self.model.addVars(len(self.predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

        terrorlist = terror_list(self.predicates)
        if self.NF == 'DNF':
            for index, sub_predicate in enumerate(self.predicates):
                temp_accs = [1]
                for index2, sub_sub_predicate in enumerate(sub_predicate):
                    temp_acc = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                    self.model.addConstr(temp_acc == temp_accs[-1] * Accs[terrorlist[index][index2]])
                    temp_accs.append(temp_acc)
                self.model.addConstr(sub_predicate_acc[index] == temp_accs[-1])

            predicate_powerset = powerset(range(len(self.predicates)))[1:]
            conj_acc = self.model.addVars(len(predicate_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS)

            for index, predicate_comb in enumerate(predicate_powerset):
                p = (-1) ** (len(predicate_comb) - 1)
                temp_vars = [1]
                for ind_predicate in list(predicate_comb):
                    temp_var = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                    self.model.addConstr(temp_var == temp_vars[-1] * sub_predicate_acc[ind_predicate])
                    temp_vars.append(temp_var)
                self.model.addConstr(conj_acc[index] == p * temp_vars[-1])
            self.model.addConstr(total_accuracy == conj_acc.sum('*'))

        elif self.NF == 'CNF':
            for index, sub_predicate in enumerate(self.predicates):
                sub_pred_powerset = powerset(range(len(sub_predicate)))[1:]
                sub_pred_vars = self.model.addVars(len(sub_pred_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS)
                for index2, predicate_comb in enumerate(sub_pred_powerset):
                    p = (-1) ** (len(predicate_comb) - 1)
                    temp_vars = [1]
                    for ind_predicate in list(predicate_comb):
                        temp_var = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                        self.model.addConstr(temp_var == temp_vars[-1] * Accs[terrorlist[index][ind_predicate]])
                        temp_vars.append(temp_var)
                    self.model.addConstr(sub_pred_vars[index2] == p * temp_vars[-1])
                self.model.addConstr(sub_predicate_acc[index] == sub_pred_vars.sum('*'))

            temp_products = [1]
            for sub_predicate in range(len(self.predicates)):
                temp_product = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                self.model.addConstr(temp_product == temp_products[-1] * sub_predicate_acc[sub_predicate])
                temp_products.append(temp_product)
            self.model.addConstr(total_accuracy == temp_products[-1])

        else:
            print("Query not in proper normal form, please retry")

        accuracy_loss = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='accuracy_loss')
        self.model.addConstr(accuracy_loss == 1 - total_accuracy)

        total_cost = self.model.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='total_cost')

        if self.goal == 'cost':
            self.minmax = 'min'
            self.model.setObjective(total_cost, grb.GRB.MINIMIZE)
            if not self.new_equations['accuracy']:
                self.model.addConstr(total_accuracy >= self.bound)
        elif self.goal == 'accuracy':
            self.minmax = 'max'
            self.model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)
            if not self.new_equations['accuracy']:
                self.model.addConstr(total_cost <= self.bound)
        self.model.update()

    def addConstr(self, constraint_name, type, value):
        if type == 'leq':
            self.model.addConstr(self.model.getVarByName(constraint_name) <= value)
        elif type == 'geq':
            self.model.addConstr(self.model.getVarByName(constraint_name) >= value)
        elif type == 'eq':
            self.model.addConstr(self.model.getVarByName(constraint_name) == value)

    def optimize(self):
        self.model.update()
        self.model.optimize()
        if self.model.Status == 3:
            print("Solution not found, starting from greedy solution")
            self.compute_start_solution()
            self.model.optimize()
        elif self.model.Status == 2:
            self.opt_accuracy = self.model.getVarByName('total_accuracy').x
            self.opt_cost = self.model.getVarByName('total_cost').x

    # compute greedy solution for max accuracy for difficult optimization
    def compute_start_solution(self):
        self.model.reset()
        self.model.params.FeasibilityTol = 1e-8
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            max_loc = self.A[flat_predicates[p]].index(max(self.A[flat_predicates[p]]))
            self.X[max_loc, p].Start = 1
            for m in range(len(self.C)):
                if m != max_loc:
                    self.X[m, p].Start = 0

    def compute_greedy_solution(self, objective, minmax):
        self.model.reset()
        self.model.params.FeasibilityTol = 1e-8
        if objective == 'accuracy':
            if minmax == 'max':
                self.compute_max_accuracy()
            elif minmax == 'min':
                self.compute_min_accuracy()
            else:
                "No valid minmax, try again"
                return
        elif objective == 'cost':
            if minmax == 'max':
                self.compute_max_cost()
            elif minmax == 'min':
                self.compute_min_cost()
            else:
                "No valid minmax, try again"
        else:
            print("no valid objective, try again")
        self.optimize()
        self.reset_bounds()

    def reset_bounds(self):
        for m in range(self.M):
            for p in range(self.P):
                self.X[m, p].lb = 0
                self.X[m, p].ub = 1

    def compute_min_accuracy(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            min_loc = self.A[flat_predicates[p]].index(max(self.A[flat_predicates[p]]))
            min_acc = 1
            for m in range(self.M):
                temp_acc = self.A[flat_predicates[p]][m]
                if min_acc > temp_acc > 0:
                    min_loc = m
                    min_acc = self.A[flat_predicates[p]][m]
            self.X[min_loc, p].lb = 1
            for m in range(self.M):
                if m != min_loc:
                    self.X[m, p].ub = 0

    def compute_max_accuracy(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            max_loc = self.A[flat_predicates[p]].index(max(self.A[flat_predicates[p]]))
            print(self.X[max_loc, p])
            self.X[max_loc, p].lb = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].ub = 0

    def compute_min_cost(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            min_cost = max(self.C)
            min_loc = self.A[flat_predicates[p]].index(max(self.A[flat_predicates[p]]))
            for m in range(self.M):
                if self.A[flat_predicates[p]][m] > 0 and self.C[m] < min_cost:
                    min_cost = self.C[m]
                    min_loc = m
            self.X[min_loc, p].lb = 1
            for m in range(self.M):
                if m != min_loc:
                    self.X[m, p].ub = 0

    def compute_max_cost(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            max_cost = min(self.C)
            max_loc = self.A[flat_predicates[p]].index(min(self.A[flat_predicates[p]]))
            for m in range(self.M):
                if self.A[flat_predicates[p]][m] > 0 and self.C[m] > max_cost:
                    max_cost = self.C[m]
                    max_loc = m
            self.X[max_loc, p].lb = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].ub = 0

