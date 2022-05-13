import gurobipy as grb
import numpy as np
from itertools import chain, combinations


def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def generate_model(A, C, goal, predicates):
    opt_model = grb.Model(name="MILQO")
    opt_model.params.NonConvex = 2
    opt_model.setParam(grb.GRB.Param.OutputFlag, 1)

    # model predicate as (x & y) | (w & z)
    flat_predicates = [item for sublist in predicates for item in sublist]
    P = len(flat_predicates)
    M = len(C)

    k = 100
    X = opt_model.addVars(M, P, vtype=grb.GRB.BINARY, name='X')
    B = opt_model.addVars(M, vtype=grb.GRB.BINARY, name='B')

    opt_model.addConstrs(X.sum('*', p) == 1 for p in range(P))
    opt_model.addConstrs(X[m, p] <= B[m] for m in range(M) for p in range(P))
    opt_model.addConstrs(X.sum(m, '*') >= B[m] for m in range(M))
    opt_model.addConstrs(X[m, p] <= k*A[flat_predicates[p]][m] for m in range(M) for p in range(P))

    Accs = opt_model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='Accs')
    opt_model.addConstrs(Accs[p] - grb.quicksum(A[flat_predicates[p]][m]*X[m, p] for m in range(M)) == 0 for p in range(P))

    sub_predicate_acc = opt_model.addVars(len(predicates), lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='sub_predicate_acc')

    index = 0
    for sub_predicate in range(len(predicates)):
        temp_temp_accs = [1]
        for sub_sub_predicate in range(len(predicates[sub_predicate])):
            temp_temp_acc = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_temp_acc'.format(index))
            opt_model.addConstr(temp_temp_acc == temp_temp_accs[-1] * Accs[index])
            temp_temp_accs.append(temp_temp_acc)
            index += 1
        opt_model.addConstr(sub_predicate_acc[sub_predicate] == temp_temp_accs[-1])

    total_accuracy = opt_model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name='total_accuracy')
    predicate_powerset = powerset(range(len(predicates)))[1:]
    conj_acc = opt_model.addVars(len(predicate_powerset), lb=-1, ub=1, vtype=grb.GRB.CONTINUOUS, name='conj_acc')

    print(predicate_powerset)
    for predicate_comb in range(len(predicate_powerset)):
        print(predicate_powerset[predicate_comb])
        p = (-1) ** (len(predicate_powerset[predicate_comb]) - 1)
        temp_vars = [1]
        for ind_predicate in list(predicate_powerset[predicate_comb]):
            temp_var = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_var'.format(predicate_comb))
            opt_model.addConstr(temp_var == temp_vars[-1] * sub_predicate_acc[ind_predicate])
            temp_vars.append(temp_var)
        opt_model.addConstr(conj_acc[predicate_comb] == p * temp_vars[-1])

    opt_model.addConstr(total_accuracy == conj_acc.sum('*'))
    accuracy_loss = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='accuracy_loss')
    opt_model.addConstr(accuracy_loss == 1 - total_accuracy)

    total_cost = opt_model.addVar(lb=0, ub=sum(C), vtype=grb.GRB.CONTINUOUS, name='total_cost')
    opt_model.addConstr(total_cost == grb.quicksum(C[m] * B[m] for m in range(M)))

    if goal == 'cost':
        opt_model.setObjective(total_cost, grb.GRB.MINIMIZE)
    elif goal == 'accuracy':
        opt_model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)

    return opt_model
