import cvxpy as cp
import numpy as np

e = 100
g = 500
b = np.array([30, 80, 40, 70, 20, 50, 20, 40])
s = np.array([50, 175, 60, 110, 35, 80, 30, 30])

selection = cp.Variable(len(b), integer=True)

budget_constraint = b @ selection <= g
energy_constraint = np.ones(len(b)) @ selection.T <= e
pos_constraint = selection >= 0
revenue = s @ selection.T

problem = cp.Problem(cp.Maximize(revenue), [budget_constraint, energy_constraint, pos_constraint])

problem.solve(solver=cp.GLPK_MI)

print(selection.value)
print()