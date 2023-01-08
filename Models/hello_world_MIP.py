import gurobipy as grb

model = grb.Model()
a = model.addVar(vtype=grb.GRB.BINARY, name='a')
b = model.addVar(vtype=grb.GRB.BINARY, name='b')

model.addConstr(a+b <= 2)
obj = a+b
model.setObjective(obj, grb.GRB.MAXIMIZE)
model.optimize()

print(a)
print(b)
print(model.objVal)