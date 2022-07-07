import gurobipy as grb
from math import ceil
from Query_tools import powerset, terror_list


class model_opt(object):
    def __init__(self, A, C, goal, bound, predicates, NF, new_equations):
        self.A = A
        self.C = C
        self.goal = goal
        self.bound = bound
        self.predicates = predicates
        self.NF = NF
        self.new_equations = new_equations
        self.model = self.generate_model()

    def generate_model(self):
        self.model = grb.Model(name="MILQO")
        self.model.setParam(grb.GRB.Param.OutputFlag, 0)
        self.model.setParam(grb.GRB.Param.IntFeasTol, 10**-9)
        self.model.params.NonConvex = 2

        # self.model predicate as (x & y) | (w & z)
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        P = len(flat_predicates)
        M = len(self.C)

        X = self.model.addVars(M, P, vtype=grb.GRB.BINARY, name='X')
        B = self.model.addVars(M, vtype=grb.GRB.BINARY, name='B')

        self.model.addConstrs(X.sum('*', p) == 1 for p in range(P))
        if self.new_equations['eq45']:
            self.model.addConstrs(X[m, p] <= B[m] for m in range(M) for p in range(P))
            self.model.addConstrs(X.sum(m, '*') >= B[m] for m in range(M))
        else:
            eps, l, u = 0.01, 0, P
            self.model.addConstrs(X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(M))
            self.model.addConstrs(X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(M))

        if self.new_equations['accuracy']:
            self.model.addConstrs(X[m, p] <= ceil(self.A[flat_predicates[p]][m]) for m in range(M) for p in range(P))

        Accs = self.model.addVars(P, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Accs')
        self.model.addConstrs(Accs[p] == grb.quicksum(self.A[flat_predicates[p]][m]*X[m, p] for m in range(M)) for p in range(P))

        total_accuracy = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name='total_accuracy')
        sub_predicate_acc = self.model.addVars(len(self.predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

        terrorlist = terror_list(self.predicates)
        if self.NF == 'DNF':
            for index, sub_predicate in enumerate(self.predicates):
                temp_accs = [1]
                for index2, sub_sub_predicate in range(len(self.predicates[sub_predicate])):
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
                print(sub_predicate)
                sub_pred_powerset = powerset(range(len(sub_predicate)))[1:]
                sub_pred_vars = self.model.addVars(len(sub_pred_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS)
                for index2, predicate_comb in enumerate(sub_pred_powerset):
                    print(predicate_comb)
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
        self.model.addConstr(total_cost == grb.quicksum(self.C[m] * B[m] for m in range(M)))

        if self.goal == 'cost':
            self.model.setObjective(total_cost, grb.GRB.MINIMIZE)
        elif self.goal == 'accuracy':
            self.model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)

        if self.goal == 'cost' and (not self.new_equations['accuracy']):
            self.model.addConstr(total_accuracy >= self.bound)
        elif self.goal == 'accuracy' and (not self.new_equations['accuracy']):
            self.model.addConstr(total_cost <= self.bound)
        return self.model

    def optimize(self):
        self.model.update()
        self.model.optimize()
