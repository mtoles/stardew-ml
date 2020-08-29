import cvxpy as cp
import numpy as np
import pandas as pd
# trees on average produce 58g worth of materials
import cvxopt.solvers as cvxopt
cvxopt.options["maxiters"] = 1000

e = 270 # starting energy
g = 500 # starting gold

#crop_names = np.array(["jazz", "cauliflower", "garlic", "green bean", "kale", "parsnip", "potato", "rhubarb", "strawberry", "tulip", "rice", "foraging"])
#b = np.array([30, 80, 40, 60, 70, 20, 50, 100, 100, 20, 40, 20, ]) # seed/crop buy price
crop_names = np.array(["jazz", "cauliflower", "foraging"])
b = np.array([30, 80, 0]) # seed/crop buy price
s = np.array([80, 175, 50]) # seed/crop sell price
f = np.array([2, 2, 30]) # planting energy cost
w = np.array([2, 2, 0]) # watering energy cost
t = np.array([7, 12, 0]) # growing time

m = 28 # days in a season
n = len(b) # number of different crops
#foraging_arbitrage = np.zeros(len(b))
#foraging_arbitrage[-1] = 1

selection = cp.Variable((m,n), integer=True)

energy_demands = [] #f @ selection <= e
pos_constraint = selection >= 0
budget_constraints = []

# create constraints for each of m days
for i in range(m): # i = today
    # budget constraints
    '''
    total_profits = cp.sum(selection[:i] @ (s - b))
    budget_constraints.append(
        selection[i] @ b <= total_profits + g
    )
    '''
    expense_relevancy = np.zeros((m, n))
    revenue_relevancy = np.zeros((m, n))
    for k in range(m): # each day
        for l in range(n): # each crop
            if k <= i: # the buy date is on or before today
                expense_relevancy[k,l] = 1 
            if k+t[l] <= i: # the sell date is on or before today
                revenue_relevancy[k,l] = 1
    expenses = np.multiply(expense_relevancy, b) # TODO: replace 1 with b in line 3 lines before and delete this line.
    revenue = np.multiply(revenue_relevancy, s)


    budget_constraint = cp.sum(sum(cp.multiply(selection, (revenue - expenses)))) + g >= 0 #+ cp.sum(sum(cp.multiply(selection, (expenses))))
    budget_constraints.append(budget_constraint)

    # energy constraints
    watering_relevancy = np.zeros((m, n))
    planting_relevancy = np.zeros((m, n))
    for k in range(m): # each day
        for l in range(n): # each crop
            if k <= i and k+t[l] > i: # the day is before today but within the growing period
                watering_relevancy[k,l] = 1
            if k == i:
                planting_relevancy[k,l] = 1
    watering_costs = np.multiply(watering_relevancy, w)
    planting_costs = np.multiply(planting_relevancy, f)
    
    energy_demand = cp.sum(sum(cp.multiply(selection, (watering_costs + planting_costs)))) <= e
    energy_demands.append(energy_demand)
    #energy_constraints = cp.hstack(energy_demands) <= e

constraints = budget_constraints + energy_demands + [pos_constraint]
#profit = sum(selection @ (s-b))
profit = cp.sum(sum(cp.multiply(selection, (revenue - expenses))))

problem = cp.Problem(cp.Maximize(profit), constraints)

problem.solve(solver=cp.CBC, verbose=True, allowablePercentageGap=2)

########################################################################################################################################

print(selection.value)
#print("expenses: ", b @ selection.value)
#print("revenue: ",  s @ selection.value)
#print("profit: ", (s - b) @ selection.value)


# Results
df = pd.DataFrame(selection.value)
df.columns = crop_names
planted_names = ["planted " + x for x in crop_names] 
df = df.reindex(df.columns.tolist() + planted_names + ["energy expended", "daily expense", "daily revenue", "cash on hand"], axis=1)
df = df.fillna(0)

# Energy Results
for i, results_col in enumerate(planted_names):
    for j, row in enumerate(range(len(df))):
        df[results_col].iloc[row] += df[df.columns[i]][max(0, j-t[i]+1):j+1].sum()
for i, row in enumerate(df["energy expended"]):
    energy_expended = 0
    for j, crop in enumerate(crop_names):
        energy_expended += f[j] * df[crop].iloc[i]
    for j, planted_crop in enumerate(planted_names):
        energy_expended += w[j] * df[planted_crop].iloc[i]
    df["energy expended"].iloc[i] = energy_expended
    
# Budget Results
current_budget = g
for i, row in df.iterrows():
    row["daily expense"] = np.sum(np.multiply(np.array(row[crop_names]), b))
    for j, crop in enumerate(crop_names):
        if i - t[j] >= 0:
            row["daily revenue"] += df[crop].iloc[i-t[j]] * s[j]
    if i == 0:
        row["cash on hand"] = 500 + row["daily revenue"] - row["daily expense"]
    else:
        row["cash on hand"] = df["cash on hand"].iloc[i-1] + row["daily revenue"] - row["daily expense"]

print(df)




print()
