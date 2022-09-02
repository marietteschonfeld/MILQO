import gurobipy as grb
from Query_tools import terror_list
from functools import reduce
from Models.Model import Model


class OrderOpt(Model):
    def __init__(self, A, C, D, Sel, goal, bound, predicates, NF, new_equations):
        Model.__init__(self, A, C, D, goal, bound, predicates, NF, new_equations)
        self.Sel = Sel
        OrderOpt.extend_model(self)

    def extend_model(self):
        flat_predicates = [item for sublist in self.predicates for item in sublist]
        J = self.P
        Pg = len(self.predicates)

        O = self.model.addVars(self.P, J, vtype=grb.GRB.BINARY, name='O')
        G = self.model.addVars(Pg, J, vtype=grb.GRB.BINARY, name='G')

        H = self.model.addVars(Pg, J, lb=0, ub=1,  vtype=grb.GRB.CONTINUOUS, name='H')
        W = self.model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='W')
        Sj = self.model.addVars(J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Sj')
        Sg = self.model.addVars(Pg, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Sg')

        # Eq 7, 8
        self.model.addConstrs(O.sum('*', j) == 1 for j in range(J))
        self.model.addConstrs(O.sum(p, '*') == 1 for p in range(self.P))

        # Eq 11
        terrorlist = terror_list(self.predicates)
        self.model.addConstrs(G[g, j] >= 1 - len(self.predicates[g]) +
                         grb.quicksum(O[p, i]
                                      for p in terrorlist[g]
                                      for i in range(0, j))
                         for g in range(Pg) for j in range(J))

        # Eq 12
        self.model.addConstrs(G[g, j] <= grb.quicksum(O[p, i] for i in range(0, j))
                         for g in range(Pg) for j in range(J) for p in terrorlist[g])

        # Eq 13
        if self.new_equations['eq13']:
            M1 = 1
            Z = self.model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Z')
            self.model.addConstrs(Z[g, j] <= G[g, j]*M1 for g in range(Pg) for j in range(J))
            self.model.addConstrs(Z[g, j] >= Sg[g] - M1 * (1-G[g, j]) for g in range(Pg) for j in range(J))
            self.model.addConstrs(Z[g, j] <= Sg[g] for g in range(Pg) for j in range(J))
            self.model.addConstrs(W[g, j] == 1 - Z[g, j] for g in range(Pg) for j in range(J))
        else:
            self.model.addConstrs(W[g, j] == 1 - G[g, j] * Sg[g] for g in range(Pg) for j in range(J))

        # Eq 14
        self.model.addConstrs(H[g, 0] == 1 for g in range(Pg))
        if self.new_equations['eq14']:
            M2 = 1
            Q = self.model.addVars(Pg, self.P, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Q')
            self.model.addConstrs(Q[g, p, j] <= M2 * O[p, j-1]
                                  for g in range(Pg) for p in range(self.P) for j in range(1, J))
            self.model.addConstrs(Q[g, p, j] >= H[g, j-1] - M2 * (1 - O[p, j-1])
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
                    temp_H = [1]
                    for i in range(0, j):
                        new_Var = self.model.addVar(vtype=grb.GRB.CONTINUOUS)
                        self.model.addConstr(new_Var == temp_H[-1]*(1 - grb.quicksum(O[p, i] - self.Sel[flat_predicates[p]]*O[p,i] for p in terrorlist[g])))
                        temp_H.append(new_Var)
                    self.model.addConstr(H[g, j] == temp_H[-1])

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

        # Selectives
        if self.NF == 'DNF':
            self.model.addConstrs(Sg[g] == reduce((lambda x, y: x * y), [self.Sel[x] for x in self.predicates[g]])
                                  for g in range(Pg))
        elif self.NF == 'CNF':
            self.model.addConstrs(Sg[g] == reduce((lambda x, y: x * y), [1 - self.Sel[x] for x in self.predicates[g]])
                                  for g in range(Pg))
        else:
            print("Not a valid NF")

        # cost calculation
        R = self.model.addVars(self.M, J, lb=0, vtype=grb.GRB.CONTINUOUS, name='R')

        if self.new_equations['eq16']:
            M3 = 1
            Y = self.model.addVars(self.M, self.P, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Y')
            self.model.addConstrs(Y[m, p, j] <= self.X[m, p] * M3 for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] <= O[p, j] * M3 for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] >= Sj[j] - M3*(2 - self.X[m, p] - O[p, j]) for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(Y[m, p, j] <= Sj[j] for m in range(self.M) for p in range(self.P) for j in range(J))
            self.model.addConstrs(R[m, j] == grb.quicksum(Y[m, p, j]*self.C[m] for p in range(self.P)) for m in range(self.M) for j in range(J))
        else:
            temp_cost = self.model.addVars(self.M, J, vtype=grb.GRB.CONTINUOUS, name='temp_cost')
            self.model.addConstrs(temp_cost[m, j] == grb.quicksum(self.X[m, p] * O[p, j] * self.C[m] for p in range(self.P))
                                  for m in range(self.M)
                                  for j in range(J))
            self.model.addConstrs(R[m, j] == Sj[j] * temp_cost[m, j] for m in range(self.M) for j in range(J))

        max_R = self.model.addVars(self.M, lb=0, ub=sum(self.C), vtype=grb.GRB.CONTINUOUS, name='max_R')
        self.model.addConstrs(max_R[m] == grb.max_(R[m, j] for j in range(J)) for m in range(self.M))

        total_cost = self.model.getVarByName('total_cost')
        self.model.addConstr(total_cost == max_R.sum('*'))
