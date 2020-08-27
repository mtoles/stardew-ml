import cvxpy as cp
import numpy as np
# trees on average produce 58g worth of materials

e = 270 # starting energy
g = 500 # starting gold

b = np.array([30, 80])
s = np.array([50, 175])
f = np.array([4, 4])

m = 4 # days in a season
n = len(b) # number of different crops
#foraging_arbitrage = np.zeros(len(b))
#foraging_arbitrage[-1] = 1

selection = cp.Variable((m,n), integer=True)

energy_constraints = [] #f @ selection <= e
pos_constraint = selection >= 0
budget_constraints = []

for i in range(m):

    total_profits = cp.sum(selection[:i] @ (s - b))
    budget_constraints.append(
        selection[i] @ b <= total_profits + g
    )

    energy_constraints.append(
        cp.sum(selection[i]) * f <= e 
    )

constraints = budget_constraints + energy_constraints + [pos_constraint]
profit = sum(selection @ (s-b))

problem = cp.Problem(cp.Maximize(profit), constraints)

problem.solve(solver=cp.GLPK_MI)

print(selection.value)
#print("expenses: ", b @ selection.value)
#print("revenue: ",  s @ selection.value)
#print("profit: ", (s - b) @ selection.value)
print()