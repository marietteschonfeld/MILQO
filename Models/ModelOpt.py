import gurobipy as grb
from Models.Model import Model


class ModelOpt(Model):
    def __init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations, env):
        Model.__init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations, env)
        self.extend_model()

    def extend_model(self):
        models = list(self.C.keys())
        total_cost = self.model.getVarByName('total_cost')
        total_memory = self.model.getVarByName('total_memory')
        B = self.model.addVars(self.M, vtype=grb.GRB.BINARY, name='B')
        if self.new_equations['eq45']:
            self.model.addConstrs(self.X[m, p] <= B[m] for m in range(self.M) for p in range(self.P))
            self.model.addConstrs(self.X.sum(m, '*') >= B[m] for m in range(self.M))
        else:
            eps, l, u = 0.01, 0, self.P
            self.model.addConstrs(self.X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(self.M))
            self.model.addConstrs(self.X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(self.M))

        self.model.addConstr(total_cost == grb.quicksum(self.C[models[m]] * B[m] for m in range(self.M)))
        if self.new_equations['memory']:
            self.model.addConstr(total_memory == grb.quicksum(self.D[models[m]] * B[m] for m in range(self.M)))

    def get_query_plan(self):
        if self.output_flag == 2:
            assignment = {}
            flat_predicates = [item for sublist in self.predicates for item in sublist]
            models = list(self.C.keys())
            for p in range(self.P):
                for m in range(self.M):
                    if round(self.X[m, p].x) == 1:
                        assignment[flat_predicates[p]] = models[m]
            ordering = list(assignment.keys())
            return assignment, ordering
        else:
            return {}, []

