import gurobipy as grb
from Models.Model import Model


class ModelOpt(Model):
    def __init__(self, A, C, goal, bound, predicates, NF, new_equations):
        super().__init__(A, C, goal, bound, predicates, NF, new_equations)
        self.extend_model()

    def extend_model(self):
        B = self.model.addVars(self.M, vtype=grb.GRB.BINARY, name='B')
        if self.new_equations['eq45']:
            self.model.addConstrs(self.X[m, p] <= B[m] for m in range(self.M) for p in range(self.P))
            self.model.addConstrs(self.X.sum(m, '*') >= B[m] for m in range(self.M))
        else:
            eps, l, u = 0.01, 0, self.P
            self.model.addConstrs(self.X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(self.M))
            self.model.addConstrs(self.X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(self.M))

        total_cost = self.model.getVarByName('total_cost')
        self.model.addConstr(total_cost == grb.quicksum(self.C[m] * B[m] for m in range(self.M)))
