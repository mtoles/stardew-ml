import cvxpy as cp
import numpy as np
# trees on average produce 58g worth of materials

e = 270 # starting energy
g = 500 # starting gold
b = np.array([30, 80, 40, 70, 20, 50, 20, 40, 0])
s = np.array([50, 175, 60, 110, 35, 80, 30, 30, 58])
f = np.array([4, 4, 4, 4, 4, 4, 4, 4, 15])
foraging_arbitrage = np.zeros(len(b))
foraging_arbitrage[-1] = 1

selection = cp.Variable(len(b), integer=True)

budget_constraint = b @ selection <= g + foraging_arbitrage @ selection * s[-1]
energy_constraint = f @ selection <= e
pos_constraint = selection >= 0
revenue = s @ selection

problem = cp.Problem(cp.Maximize(revenue), [budget_constraint, energy_constraint, pos_constraint])

problem.solve(solver=cp.GLPK_MI)

print(selection.value)
print("expenses: ", b @ selection.value)
print("revenue: ",  s @ selection.value)
print("profit: ", (s - b) @ selection.value)
print()