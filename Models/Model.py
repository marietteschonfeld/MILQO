import gurobipy as grb
from math import ceil
from Query_tools import terror_list, powerset


# Parent class for the general model
class Model:
    def __init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations, env):
        self.A = A
        self.C = C
        self.D = D
        self.Sel = Sel
        self.goal = goal
        self.bound = bound
        self.predicates = predicates
        self.NF = NF
        self.new_equations = new_equations
        self.model = grb.Model(name="MILQO", env=env)
        self.opt_accuracy = 0
        self.opt_cost = 0
        self.opt_memory = 0
        self.output_flag = 0
        self.generate_model()

    def generate_model(self):
        self.model.setParam(grb.GRB.Param.OutputFlag, 0)
        self.model.params.NonConvex = 2

        # self.model predicate as (x & y) | (w & z)
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        self.P = len(flat_predicates)
        models = list(self.C.keys())
        self.M = len(models)

        self.X = self.model.addVars(self.M, self.P, vtype=grb.GRB.BINARY, name='X')

        self.model.addConstrs(self.X.sum('*', p) == 1 for p in range(self.P))

        if self.new_equations['accuracy']:
            self.model.addConstrs(self.X[m, p] <= ceil(self.A[flat_predicates[p]][models[m]]) for m in range(self.M) for p in range(self.P))
            for m in range(self.M):
                for p in range(self.P):
                    self.X[m, p].ub = ceil(self.A[flat_predicates[p]][models[m]])

        Accs = self.model.addVars(self.P, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Accs')
        self.model.addConstrs(Accs[p] == grb.quicksum(self.A[flat_predicates[p]][models[m]]*self.X[m, p] for m in range(self.M)) for p in range(self.P))

        total_accuracy = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name='total_accuracy')
        group_acc = self.model.addVars(len(self.predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

        terrorlist = terror_list(self.predicates)
        if self.NF == 'DNF':
            for index, group in enumerate(self.predicates):
                temp_accs = [1]
                for index2, predicate in enumerate(group):
                    temp_acc = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                    self.model.addConstr(temp_acc == temp_accs[-1] * Accs[terrorlist[index][index2]])
                    temp_accs.append(temp_acc)
                self.model.addConstr(group_acc[index] == temp_accs[-1])

            predicate_powerset = powerset(range(len(self.predicates)))[1:]
            conj_acc = self.model.addVars(len(predicate_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS)

            for index, group_comb in enumerate(predicate_powerset):
                p = (-1) ** (len(group_comb) - 1)
                temp_accs = [1]
                for ind_predicate in list(group_comb):
                    temp_acc = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                    self.model.addConstr(temp_acc == temp_accs[-1] * group_acc[ind_predicate])
                    temp_accs.append(temp_acc)
                self.model.addConstr(conj_acc[index] == p * temp_accs[-1])
            self.model.addConstr(total_accuracy == conj_acc.sum('*'))

        elif self.NF == 'CNF':
            for group_num, group in enumerate(self.predicates):
                group_powerset = powerset(range(len(group)))[1:]
                group_accs = self.model.addVars(len(group_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS)
                for predicate_num, predicate_comb in enumerate(group_powerset):
                    p = (-1) ** (len(predicate_comb) - 1)
                    temp_accs = [1]
                    for ind_predicate in list(predicate_comb):
                        temp_acc = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                        self.model.addConstr(temp_acc == temp_accs[-1] * Accs[(terrorlist[group_num][ind_predicate])])
                        temp_accs.append(temp_acc)
                    self.model.addConstr(group_accs[predicate_num] == p * temp_accs[-1])
                self.model.addConstr(group_acc[group_num] == group_accs.sum('*'))

            temp_accs = [1]
            for group_num in range(len(self.predicates)):
                temp_acc = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                self.model.addConstr(temp_acc == temp_accs[-1] * group_acc[group_num])
                temp_accs.append(temp_acc)
            self.model.addConstr(total_accuracy == temp_accs[-1])

        else:
            print("Query not in proper normal form, please retry")

        accuracy_loss = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='accuracy_loss')
        self.model.addConstr(accuracy_loss == 1 - total_accuracy)

        total_cost = self.model.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='total_cost')
        total_memory = self.model.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='total_memory')


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

    def optimize(self, timeout=60*60):
        self.model.update()
        self.model.params.TimeLimit = timeout
        self.model.optimize()
        self.output_flag = self.model.Status
        if self.model.Status == 3:
            self.compute_start_solution()
            # self.model.feasRelaxS(1, False, False, True)
            self.model.optimize()
            if self.model.Status == 2:
                self.output_flag = self.model.Status
                self.opt_accuracy = self.model.getVarByName('total_accuracy').x
                self.opt_cost = self.model.getVarByName('total_cost').x
                self.opt_memory = self.model.getVarByName('total_memory').x
            else:
                # TODO: calculate greedy values
                self.output_flag = self.model.Status
                self.opt_accuracy = 0
                self.opt_cost = sum(self.C.values())
                self.opt_memory = sum(self.D.values())
        if self.model.Status == 2 or (self.model.Status == 9 and self.model.solCount>0):
            self.output_flag = self.model.Status
            self.opt_accuracy = self.model.getVarByName('total_accuracy').x
            self.opt_cost = self.model.getVarByName('total_cost').x
            self.opt_memory = self.model.getVarByName('total_memory').x
        else:
            self.output_flag = self.model.Status
            self.opt_accuracy = 0
            self.opt_cost = sum(self.C.values())
            self.opt_memory = sum(self.D.values())

    # compute greedy solution for max accuracy for difficult optimization
    def compute_start_solution(self):
        self.model.reset()
        self.model.params.FeasibilityTol = 1e-2
        self.model.params.IntFeasTol = 1e-2
        self.model.params.OptimalityTol = 1e-2
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            max_loc = list(self.A[flat_predicates[p]].values()).index(max(self.A[flat_predicates[p]].values()))
            self.X[max_loc, p].Start = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].Start = 0
        self.model.update()

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
        elif objective == 'memory':
            if minmax == 'max':
                self.compute_max_memory()
            elif minmax == 'min':
                self.compute_min_memory()
            else:
                "No valid minmax, try again"
        else:
            print("no valid objective, try again")
        self.optimize(timeout=60*5)
        self.reset_bounds()

    def reset_bounds(self):
        for m in range(self.M):
            for p in range(self.P):
                self.X[m, p].lb = 0
                self.X[m, p].ub = 1

    def compute_min_accuracy(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            min_loc = list(self.A[flat_predicates[p]].values()).index(max(self.A[flat_predicates[p]].values()))
            min_acc = 1
            for m in range(self.M):
                temp_acc = self.A[flat_predicates[p]][models[m]]
                if min_acc > temp_acc > 0:
                    min_loc = m
                    min_acc = self.A[flat_predicates[p]][models[m]]
            self.X[min_loc, p].lb = 1
            for m in range(self.M):
                if m != min_loc:
                    self.X[m, p].ub = 0

    def compute_max_accuracy(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        for p in range(self.P):
            max_loc = list(self.A[flat_predicates[p]].values()).index(max(self.A[flat_predicates[p]].values()))
            self.X[max_loc, p].lb = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].ub = 0

    def compute_min_cost(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            min_cost = max(self.C.values())
            min_loc = list(self.A[flat_predicates[p]].values()).index(max(self.A[flat_predicates[p]].values()))
            for m in range(self.M):
                if self.A[flat_predicates[p]][models[m]] > 0 and self.C[models[m]] < min_cost:
                    min_cost = self.C[models[m]]
                    min_loc = m
            self.X[min_loc, p].lb = 1
            for m in range(self.M):
                if m != min_loc:
                    self.X[m, p].ub = 0

    def compute_max_cost(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            max_cost = min(self.C.values())
            max_loc = list(self.A[flat_predicates[p]].values()).index(min(self.A[flat_predicates[p]].values()))
            for m in range(self.M):
                if self.A[flat_predicates[p]][models[m]] > 0 and self.C[models[m]] > max_cost:
                    max_cost = self.C[models[m]]
                    max_loc = m
            self.X[max_loc, p].lb = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].ub = 0


    def compute_min_memory(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            min_memory = max(self.D.values())
            min_loc = list(self.A[flat_predicates[p]].values()).index(max(self.A[flat_predicates[p]].values()))
            for m in range(self.M):
                if self.A[flat_predicates[p]][models[m]] > 0 and self.D[models[m]] < min_memory:
                    min_memory = self.D[models[m]]
                    min_loc = m
            self.X[min_loc, p].lb = 1
            for m in range(self.M):
                if m != min_loc:
                    self.X[m, p].ub = 0

    def compute_max_memory(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            max_memory = min(self.D.values())
            max_loc = list(self.A[flat_predicates[p]].values()).index(min(self.A[flat_predicates[p]].values()))
            for m in range(self.M):
                if self.A[flat_predicates[p]][models[m]] > 0 and self.D[models[m]] > max_memory:
                    max_memory = self.D[models[m]]
                    max_loc = m
            self.X[max_loc, p].lb = 1
            for m in range(self.M):
                if m != max_loc:
                    self.X[m, p].ub = 0

