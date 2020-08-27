import cvxpy as cp
import numpy as np
# trees on average produce 58g worth of materials

e = 270 # starting energy
g = 500 # starting gold

b = np.array([30, 80]) # seed/crop buy price
s = np.array([50, 175]) # seed/crop sell price
f = np.array([2, 2]) # planting energy
w = np.array([2, 2]) # watering energy
t = np.array([7, 12]) # growing time

m = 28 # days in a season
n = len(b) # number of different crops
#foraging_arbitrage = np.zeros(len(b))
#foraging_arbitrage[-1] = 1

selection = cp.Variable((m,n), integer=True)

energy_constraints = [] #f @ selection <= e
pos_constraint = selection >= 0
budget_constraints = []

# create budget constraints for each of m days
for i in range(m): # i = today

    total_profits = cp.sum(selection[:i] @ (s - b))
    budget_constraints.append(
        selection[i] @ b <= total_profits + g
    )

    watering_budget = 0
    watering_relevancy = np.zeros((m, n))
    for k in range(m): # each day
        for l in range(n): # each crop
            if k <= i and k+t[l] >= i: # the day is before today but within the growing period
                watering_relevancy[k,l] = 1
    print(watering_relevancy)



    # account energy for each crop
    #for j in range(w):
        #selection[:i][]
        #watering_budget += selection[
        #energy_constraints.append(
            
            #cp.sum(selection[i]) * f <= e 
        #)

constraints = budget_constraints + energy_constraints + [pos_constraint]
profit = sum(selection @ (s-b))

problem = cp.Problem(cp.Maximize(profit), constraints)

problem.solve(solver=cp.GLPK_MI)

print(selection.value)
#print("expenses: ", b @ selection.value)
#print("revenue: ",  s @ selection.value)
#print("profit: ", (s - b) @ selection.value)
print()