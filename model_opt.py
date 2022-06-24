import gurobipy as grb
import numpy as np
from math import ceil
from itertools import chain, combinations
from functools import reduce


def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def model_opt(A, C, goal, bound, predicates, new_equations):
    model = grb.Model(name="MILQO")
    grb.resetParams()
    # model.setParam(grb.GRB.Param.OutputFlag, 0)
    # model.setParam(grb.GRB.Param.IntFeasTol, 10**-9)
    # model.setParam(grb.GRB.Param.FeasibilityTol, 10**-9)
    # model.setParam(grb.GRB.Param.IntegralityFocus, 1)
    # model.setParam(grb.GRB.Param.FeasRelaxBigM, 1000
    model.params.NonConvex = 2

    # model predicate as (x & y) | (w & z)
    flat_predicates = [item for sublist in predicates for item in sublist]
    P = len(flat_predicates)
    M = len(C)

    X = model.addVars(M, P, vtype=grb.GRB.BINARY, name='X')
    B = model.addVars(M, vtype=grb.GRB.BINARY, name='B')

    model.addConstrs(X.sum('*', p) == 1 for p in range(P))
    if new_equations['eq45']:
        model.addConstrs(X[m, p] <= B[m] for m in range(M) for p in range(P))
        model.addConstrs(X.sum(m, '*') >= B[m] for m in range(M))
    else:
        eps, l, u = 0.01, 0, P
        model.addConstrs(X.sum(m, '*') <= 1 - eps + (u-1+eps)*B[m] for m in range(M))
        model.addConstrs(X.sum(m, '*') >= B[m] + l*(1-B[m]) for m in range(M))

    if new_equations['accuracy']:
        for m in range(M):
            for p in range(P):
                X[m, p].ub = ceil(A[flat_predicates[p]][m])

    Accs = model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='Accs')
    model.addConstrs(Accs[p] - grb.quicksum(A[flat_predicates[p]][m]*X[m, p] for m in range(M)) == 0 for p in range(P))

    sub_predicate_acc = model.addVars(len(predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

    terror_list = reduce(lambda a, b: a + [list(range(a[-1][-1]+1, 1+len(b)+a[-1][-1]))],
                         predicates[1:], [list(range(len(predicates[0])))])
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
    predicate_powerset = powerset(range(len(predicates)))[1:]
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

    total_cost = model.addVar(lb=0, vtype=grb.GRB.CONTINUOUS, name='total_cost')
    model.addConstr(total_cost == grb.quicksum(C[m] * B[m] for m in range(M)))

    if goal == 'cost':
        model.setObjective(total_cost, grb.GRB.MINIMIZE)
    elif goal == 'accuracy':
        model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)

    if goal == 'cost' and (not new_equations['accuracy']):
        model.addConstr(total_accuracy >= bound)
    elif goal == 'accuracy' and (not new_equations['accuracy']):
        model.addConstr(total_cost <= bound)
    return model
