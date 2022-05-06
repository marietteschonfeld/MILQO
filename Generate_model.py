import gurobipy as grb
import numpy as np

def generate_model(A, C, goal, predicates):
    opt_model = grb.Model(name="MILQO")
    opt_model.params.NonConvex = 2
    opt_model.setParam(grb.GRB.Param.OutputFlag, 1)

    # model predicate as (x & y) | (w & z)
    P = len(predicates)
    M = len(C)

    X = opt_model.addVars(M, P, vtype=grb.GRB.BINARY, name='X')
    B = opt_model.addVars(M, vtype=grb.GRB.BINARY, name='B')

    opt_model.addConstrs(X.sum('*', p) == 1 for p in range(P))
    opt_model.addConstrs(X[m, p] - B[m] <= 0 for m in range(M) for p in range(P))

    Accs = opt_model.addVars(P, vtype=grb.GRB.CONTINUOUS, name='Accs')
    opt_model.addConstrs(Accs[p] - grb.quicksum(A[predicates[p]][m]*X[m, p] for m in range(M)) == 0 for p in range(P))

    temp_acc_1 = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_acc_1')
    opt_model.addConstr(temp_acc_1 == Accs[0] * Accs[1])
    temp_acc_2 = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_acc_2')
    opt_model.addConstr(temp_acc_2 == Accs[2] * Accs[3])
    temp_acc_3 = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='temp_acc_3')
    opt_model.addConstr(temp_acc_3 == temp_acc_1 * temp_acc_2)
    total_accuracy = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='total_accuracy')
    opt_model.addConstr(total_accuracy == temp_acc_1 + temp_acc_2 - temp_acc_3)
    accuracy_loss = opt_model.addVar(lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='accuracy_loss')
    opt_model.addConstr(accuracy_loss == 1 - total_accuracy)

    total_cost = opt_model.addVar(lb=0, ub=sum(C), vtype=grb.GRB.CONTINUOUS, name='total_cost')
    opt_model.addConstr(total_cost == grb.quicksum(C[m] * B[m] for m in range(M)))

    if goal == 'cost':
        opt_model.setObjective(total_cost, grb.GRB.MINIMIZE)
    elif goal == 'accuracy':
        opt_model.setObjective(total_accuracy, grb.GRB.MAXIMIZE)

    return opt_model
