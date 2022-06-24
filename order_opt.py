import gurobipy as grb
from math import ceil
from itertools import chain, combinations
from functools import reduce


def power_set(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def order_opt(A, C, Sel, goal, bound, predicates, new_equations):
    model = grb.Model(name="MILQO")
    grb.resetParams()
    # model.setParam(grb.GRB.Param.OutputFlag, 0)
    # model.setParam(grb.GRB.Param.IntFeasTol, 10**-2)
    model.setParam(grb.GRB.Param.FeasibilityTol, 10**-3)
    # model.setParam(grb.GRB.Param.IntegralityFocus, 1)
    # model.setParam(grb.GRB.Param.FeasRelaxBigM, 1000)
    model.params.NonConvex = 2

    # model predicate as (x & y) | (w & z)
    flat_predicates = [item for sublist in predicates for item in sublist]
    P = len(flat_predicates)
    M = len(C)
    J = P
    Pg = len(predicates)

    X = model.addVars(M, P, vtype=grb.GRB.BINARY, name='X')
    O = model.addVars(P, J, vtype=grb.GRB.BINARY, name='O')
    G = model.addVars(Pg, J, vtype=grb.GRB.BINARY, name='G')

    H = model.addVars(Pg, J, lb=0, ub=1,  vtype=grb.GRB.CONTINUOUS, name='H')
    W = model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='W')
    Sj = model.addVars(J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Sj')
    Sg = model.addVars(Pg, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Sg')

    # Eq 1
    model.addConstrs(X.sum('*', p) == 1 for p in range(P))

    if new_equations['accuracy']:
        for m in range(M):
            for p in range(P):
                X[m, p].ub = ceil(A[flat_predicates[p]][m])

    # Eq 7, 8
    model.addConstrs(O.sum('*', j) == 1 for j in range(J))
    model.addConstrs(O.sum(p, '*') == 1 for p in range(P))

    # Eq 11
    # don't ask questions, it works
    terror_list = reduce(lambda a, b: a + [list(range(a[-1][-1]+1, 1+len(b)+a[-1][-1]))],
                         predicates[1:], [list(range(len(predicates[0])))])
    model.addConstrs(G[g, j] >= 1 - len(predicates[g]) +
                     grb.quicksum(O[p, i]
                                  for p in terror_list[g]
                                  for i in range(0, j))
                     for g in range(Pg) for j in range(J))

    # Eq 12
    model.addConstrs(G[g, j] <= grb.quicksum(O[p, i] for i in range(0, j))
                     for g in range(Pg) for j in range(J) for p in terror_list[g])

    # Eq 13
    if new_equations['eq13']:
        M1 = 1
        Z = model.addVars(Pg, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Z')
        model.addConstrs(Z[g, j] <= G[g, j]*M1 for g in range(Pg) for j in range(J))
        model.addConstrs(Z[g, j] >= Sg[g] - M1 * (1-G[g, j]) for g in range(Pg) for j in range(J))
        model.addConstrs(Z[g, j] <= Sg[g] for g in range(Pg) for j in range(J))
        model.addConstrs(W[g, j] == 1 - Z[g, j] for g in range(Pg) for j in range(J))
    else:
        model.addConstrs(W[g, j] == 1 - G[g, j] * Sg[g] for g in range(Pg) for j in range(J))

    # Eq 14
    model.addConstrs(H[g, 0] == 1 for g in range(Pg))
    if new_equations['eq14']:
        M2 = 1
        Q = model.addVars(Pg, P, J-1, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Q')
        model.addVars(Q[g, p, j] <= M2 * O[p, j] for g in range(Pg) for p in range(P) for j in range(J-1))
        model.addVars(Q[g, p, j] >= H[g, j] - M2 * (1 - O[p, j]) for g in range(Pg) for p in range(P) for j in range(J-1))
        model.addVars(Q[g, p, j] <= H[g, j] for g in range(Pg) for p in range(P) for j in range(J-1))
        model.addConstrs(H[g, j] == H[g, j-1] - grb.quicksum(Q[g, p, j-1] * (1 - Sel[flat_predicates[p]]) for p in terror_list[g])
                         for g in range(Pg) for j in range(1, J))
    else:
        for g in range(Pg):
            for j in range(1, J):
                temp_H = [1]
                for i in range(0, j):
                    new_Var = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS)
                    model.addConstr(new_Var == temp_H[-1]*(1 - grb.quicksum(O[p, i] - Sel[flat_predicates[p]]*O[p,i] for p in terror_list[g])))
                    temp_H.append(new_Var)
                model.addConstr(H[g, j] == temp_H[-1])

    # Eq 15
    for j in range(J):
        temp_selectives = [1]
        for g in range(Pg):
            temp_selective = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sel_{}'.format(j))
            temp_temp_selective = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sel_temp_{}'.format(j))
            model.addConstr(temp_temp_selective == W[g, j]*H[g, j])
            model.addConstr(temp_selective == temp_selectives[-1] * temp_temp_selective)
            temp_selectives.append(temp_selective)
        model.addConstr(Sj[j] == temp_selectives[-1])

    # Selectives
    model.addConstrs(Sg[g] == reduce((lambda x, y: x * y), [Sel[x] for x in predicates[g]])
                     for g in range(Pg))

    # Accuracy calculation
    Accs = model.addVars(P, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Accs')
    model.addConstrs(Accs[p] - grb.quicksum(A[flat_predicates[p]][m]*X[m, p] for m in range(M)) == 0 for p in range(P))

    sub_predicate_acc = model.addVars(len(predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

    index = 0
    for sub_predicate in range(len(predicates)):
        temp_temp_accs = [1]
        for sub_sub_predicate in range(len(predicates[sub_predicate])):
            temp_temp_acc = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_temp_acc'.format(index))
            model.addConstr(temp_temp_acc == temp_temp_accs[-1] * Accs[index])
            temp_temp_accs.append(temp_temp_acc)
            index += 1
        model.addConstr(sub_predicate_acc[sub_predicate] == temp_temp_accs[-1])

    total_accuracy = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name='total_accuracy')
    predicate_powerset = power_set(range(len(predicates)))[1:]
    conj_acc = model.addVars(len(predicate_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS, name='conj_acc')

    for predicate_comb in range(len(predicate_powerset)):
        p = (-1) ** (len(predicate_powerset[predicate_comb]) - 1)
        temp_vars = [1]
        for ind_predicate in list(predicate_powerset[predicate_comb]):
            temp_var = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_var'.format(predicate_comb))
            model.addConstr(temp_var == temp_vars[-1] * sub_predicate_acc[ind_predicate])
            temp_vars.append(temp_var)
        model.addConstr(conj_acc[predicate_comb] == p * temp_vars[-1])

    model.addConstr(total_accuracy == conj_acc.sum('*'))
    accuracy_loss = model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='accuracy_loss')
    model.addConstr(accuracy_loss == 1 - total_accuracy)

    # cost calculation
    R = model.addVars(M, J, lb=0, vtype=grb.GRB.CONTINUOUS, name='R')

    if new_equations['eq16']:
        M3 = 1
        Y = model.addVars(M, P, J, vtype=grb.GRB.BINARY, name='Y')
        D = model.addVars(M, P, J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='D')
        model.addConstrs(Y[m, p, j] <= X[m, p] for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(Y[m, p, j] <= O[p, j] for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(Y[m, p, j] >= X[m, p] + O[p, j] - 1 for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(D[m, p, j] <= M3*Y[m, p, j] for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(D[m, p, j] >= Sj[j] - M3*(1-Y[m, p, j]) for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(D[m, p, j] <= Sj[j] for m in range(M) for p in range(P) for j in range(J))
        model.addConstrs(R[m, j] == grb.quicksum(D[m, p, j]*C[m] for p in range(P)) for m in range(M) for j in range(J))
    else:
        temp_cost = model.addVars(M, J, ub=0, lb=sum(C), vtype=grb.GRB.CONTINUOUS, name='temp_cost')
        model.addConstrs(temp_cost[m, j] == grb.quicksum(X[m, p] * O[p, j] * C[m] for p in range(P))
                         for m in range(M)
                         for j in range(J))
        model.addConstrs(R[m, j] == Sj[j] * temp_cost[m, j] for m in range(M) for j in range(J))

    max_R = model.addVars(M, lb=0, ub=sum(C), vtype=grb.GRB.CONTINUOUS, name='max_R')
    model.addConstrs(max_R[m] >= R[m, j] for j in range(J) for m in range(M))

    total_cost = model.addVar(lb=0, ub=sum(C), vtype=grb.GRB.CONTINUOUS, name='total_cost')
    model.addConstr(total_cost == max_R.sum('*'))

    if goal == 'cost':
        model.setObjective(total_cost, grb.GRB.MINIMIZE)
    elif goal == 'accuracy':
        model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)

    # only set bound when not using ceiling on X with accuracy
    if goal == 'cost' and (not new_equations['accuracy']):
        model.addConstr(total_accuracy >= bound)
    elif goal == 'accuracy' and (not new_equations['accuracy']):
        model.addConstr(total_cost <= bound)

    return model
