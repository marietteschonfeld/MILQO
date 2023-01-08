from math import ceil

import gurobipy as grb
from Query_tools import terror_list
from functools import reduce
from Models.Model import Model


class OrderOpt(Model):
    def __init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations, env):
        Model.__init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations, env)
        OrderOpt.extend_model(self)

    def extend_model(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        J = self.P
        Pg = len(self.predicates)
        models = list(self.C.keys())
        new_C = self.extend_C()

        self.O = self.model.addVars(self.P, J, vtype=grb.GRB.BINARY, name='O')
        G = self.model.addVars(Pg, J, vtype=grb.GRB.BINARY, name='G')

        H = self.model.addVars(Pg, J, lb=0, ub=1,  vtype=grb.GRB.CONTINUOUS, name='H')
        W = self.model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='W')
        Sj = self.model.addVars(J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Sj')
        Sg = []
        # Selectives
        if self.NF == 'DNF':
            for g in range(Pg):
                Sg.append(reduce((lambda x, y: x * y), [self.Sel[x] for x in self.predicates[g]]))
        elif self.NF == 'CNF':
            for g in range(Pg):
                Sg.append(reduce((lambda x, y: x * y), [1 - self.Sel[x] for x in self.predicates[g]]))
        else:
            print("Not a valid NF")

        # Eq 7, 8
        self.model.addConstrs(self.O.sum('*', j) == 1 for j in range(J))
        self.model.addConstrs(self.O.sum(p, '*') == 1 for p in range(self.P))

        # Eq 11
        terrorlist = terror_list(self.predicates)
        self.model.addConstrs(G[g, j] >= 1 - len(self.predicates[g]) +
                         grb.quicksum(self.O[p, i]
                                      for p in terrorlist[g]
                                      for i in range(0, j))
                         for g in range(Pg) for j in range(J))

        # Eq 12
        self.model.addConstrs(G[g, j] <= grb.quicksum(self.O[p, i] for i in range(0, j))
                         for g in range(Pg) for j in range(J) for p in terrorlist[g])

        # Eq 13
        self.model.addConstrs(W[g, j] == 1 - G[g, j] * Sg[g] for g in range(Pg) for j in range(J))

        # Eq 14
        self.model.addConstrs(H[g, 0] == 1 for g in range(Pg))
        if self.new_equations['eq14']:
            M2 = 1
            Q = self.model.addVars(Pg, self.P, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Q')
            self.model.addConstrs(Q[g, p, j] <= M2 * self.O[p, j-1]
                                  for g in range(Pg) for p in range(self.P) for j in range(1, J))
            self.model.addConstrs(Q[g, p, j] >= H[g, j-1] - M2 * (1 - self.O[p, j-1])
                                  for g in range(Pg) for p in range(self.P) for j in range(1, J))
            self.model.addConstrs(Q[g, p, j] <= H[g, j-1]
                                  for g in range(Pg) for p in range(self.P) for j in range(1, J))
            self.model.addConstrs(H[g, j] == H[g, j-1] - grb.quicksum(Q[g, p, j] *
                                                                      (1 - self.Sel[flat_predicates[p]])
                                                                      for p in terrorlist[g])
                                  for g in range(Pg) for j in range(1, J))
        else:
            for g in range(Pg):
                for j in range(1, J):
                    self.model.addConstr(H[g, j] == H[g, j-1] *
                                         (1-grb.quicksum(self.O[p, j-1]*(1-self.Sel[flat_predicates[p]])
                                                         for p in terrorlist[g])))

        # H_prime = self.model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name="H_prime")
        # temp_sum = self.model.addVars(Pg, J, vtype=grb.GRB.CONTINUOUS)
        # self.model.addConstrs(temp_sum[g, j] == 1-grb.quicksum(self.O[p, j] for p in terrorlist[g])
        #                       for g in range(Pg) for j in range(J))
        # self.model.addConstrs(H_prime[g, j] == grb.max_([H[g, j],
        #                                                 temp_sum[g, j],
        #                                                 G[g, j]])
        #                       for g in range(Pg) for j in range(J))

        # Eq 15
        for j in range(J):
            temp_selectives = [1]
            for g in range(Pg):
                temp_selective = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sel_{}'.format(j))
                temp_temp_selective = self.model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sel_temp_{}'.format(j))
                self.model.addConstr(temp_temp_selective == W[g, j]*H[g, j])
                self.model.addConstr(temp_selective == temp_selectives[-1] * temp_temp_selective)
                temp_selectives.append(temp_selective)
            self.model.addConstr(Sj[j] == temp_selectives[-1])

        # cost calculation
        R = self.model.addVars(self.M, J, lb=0, vtype=grb.GRB.CONTINUOUS, name='R')
        T = self.model.addVars(self.M, J, lb=0, vtype=grb.GRB.CONTINUOUS, name='T')

        if self.new_equations['eq16']:
            M3 = 1
            Y = self.model.addVars(self.M, self.P, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Y')
            self.model.addConstrs(Y[m, p, j] <= self.X[m, p] * M3 for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] <= self.O[p, j] * M3 for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] >= Sj[j] - M3*(2 - self.X[m, p] - self.O[p, j]) for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] <= Sj[j] for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(R[m, j] == grb.quicksum(Y[m, p, j]*self.C[models[m]] for p in range(self.P)) for m in range(self.M) for j in range(J))

        else:
            temp_cost = self.model.addVars(self.M, J, vtype=grb.GRB.CONTINUOUS, name='temp_cost')
            self.model.addConstrs(temp_cost[m, j] == grb.quicksum(self.X[m, p] * self.O[p, j] * new_C[flat_predicates[p]][models[m]] for p in range(self.P))
                                  for m in range(self.M)
                                  for j in range(J))
            self.model.addConstrs(R[m, j] == Sj[j] * temp_cost[m, j] for m in range(self.M) for j in range(J))

        if self.new_equations['memory']:
            total_memory = self.model.getVarByName('total_memory')
            B = self.model.addVars(self.M, vtype=grb.GRB.BINARY, name='B')
            if self.new_equations['eq45']:
                self.model.addConstrs(self.X[m, p] <= B[m] for m in range(self.M) for p in range(self.P))
                self.model.addConstrs(self.X.sum(m, '*') >= B[m] for m in range(self.M))
            else:
                eps, l, u = 0.01, 0, self.P
                self.model.addConstrs(self.X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(self.M))
                self.model.addConstrs(self.X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(self.M))
            self.model.addConstr(total_memory == grb.quicksum(self.D[models[m]] * B[m] for m in range(self.M)))

        else:
            # B = self.model.addVars(self.M, vtype=grb.GRB.BINARY, name='B')
            # eps, l, u = 0.01, 0, self.P
            # self.model.addConstrs(self.X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(self.M))
            # self.model.addConstrs(self.X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(self.M))
            total_memory = self.model.getVarByName('total_memory')
            # self.model.addConstr(total_memory == grb.quicksum(self.D[models[m]] * B[m] for m in range(self.M)))

        max_R = self.model.addVars(self.M, lb=0, ub=100000, vtype=grb.GRB.CONTINUOUS, name='max_R')
        self.model.addConstrs(max_R[m] == grb.max_(R[m, j] for j in range(J)) for m in range(self.M))

        total_cost = self.model.getVarByName('total_cost')
        self.model.addConstr(total_cost == max_R.sum('*'))


    def get_query_plan(self):
        if self.output_flag == 2:
            assignment = {}
            flat_predicates = [item for sublist in self.predicates for item in sublist]
            models = list(self.C.keys())
            for p in range(self.P):
                for m in range(self.M):
                    if round(self.X[m, p].x) == 1:
                        assignment[flat_predicates[p]] = models[m]
            ordering = []
            for j in range(self.P):
                for p in range(self.P):
                    if round(self.O[p, j].x) == 1:
                        ordering.append(flat_predicates[p])
            return assignment, ordering
        else:
            return {}, []

    def extend_C(self):
        inf=100000
        new_C = {}
        for pred in self.A.keys():
            new_C[pred] = {}
            for mod in self.C.keys():
                new_C[pred][mod] = ceil(self.A[pred][mod])*self.C[mod] + (1-ceil(self.A[pred][mod]))*inf
        return new_C
