import gurobipy as grb
from Models.Model import Model


class ModelOpt(Model):
    def __init__(self, A, C, D, goal, bound, predicates, NF, new_equations):
        Model.__init__(self, A, C, D, goal, bound, predicates, NF, new_equations)
        self.extend_model()

    def extend_model(self):
        models = list(self.C.keys())
        total_cost = self.model.getVarByName('total_cost')
        B = [self.model.getVarByName('B[{}]'.format(m)) for m in range(self.M)]
        self.model.addConstr(total_cost == grb.quicksum(self.C[models[m]] * B[m] for m in range(self.M)))

    def get_query_plan(self):
        assignment = {}
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        models = list(self.C.keys())
        for p in range(self.P):
            for m in range(self.M):
                if round(self.X[m, p].x) == 1:
                    assignment[flat_predicates[p]] = models[m]
        ordering = list(assignment.keys())
        return assignment, ordering

