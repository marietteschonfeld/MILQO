import gurobipy as grb
from Models.Model import Model


class ModelOpt(Model):
    def __init__(self, A, C, D, goal, bound, predicates, NF, new_equations):
        Model.__init__(self, A, C, D, goal, bound, predicates, NF, new_equations)
        self.extend_model()

    def extend_model(self):
        total_cost = self.model.getVarByName('total_cost')
        B = [self.model.getVarByName('B[{}]'.format(m)) for m in range(self.M)]
        self.model.addConstr(total_cost == grb.quicksum(self.C[m] * B[m] for m in range(self.M)))
