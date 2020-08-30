import cvxpy as cp
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
# trees on average produce 50g worth of materials and take 30 energy to harvest
# assumptions:
    # you cannot hoe ground without planting something in it
    # you water crops every day
    # time is not a constraint

import cvxopt.solvers as cvxopt
cvxopt.options["maxiters"] = 1000

e = 270 # starting energy
g = 500 # starting gold

class Crop:
    def __init__(self, name, b, s, t, f=2, w=2, regrowth=sys.maxsize):
        self.name = name
        self.b = b
        self.s = s
        self.t = t
        self.f = f
        self.w = w
        self.regrowth = regrowth

crops = [
    Crop("jazz", 30, 50, 7),
    Crop("cauliflower", 80, 175, 12),
    Crop("garlic", 40, 60, 4),
    Crop("green bean", 60, 40, 10, regrowth=3),
    Crop("kale", 70, 110, 6),
    Crop("parsnip", 20, 35, 4),
    Crop("potato", 50, 80, 6),
    Crop("tulip", 20, 30, 6),
    Crop("foraging", 0, 50, 0, f=30, w=0)
]

crop_names = []
b = []
s = []
t = []
f = []
w = []
regrowth = []

for crop in crops:
    crop_names.append(crop.name)
    b.append(crop.b)
    s.append(crop.s)
    t.append(crop.t)
    f.append(crop.f)
    w.append(crop.w)
    regrowth.append(crop.regrowth)
print(crops)

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
                revenue_relevancy[k,l] = 1 + (i - k - t[l]) // crops[l].regrowth

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
            if k <= i and regrowth[l] != sys.maxsize and m - k >= regrowth[l]:
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

#print(selection.value)
#print("expenses: ", b @ selection.value)
#print("revenue: ",  s @ selection.value)
#print("profit: ", (s - b) @ selection.value)


# Results
df = pd.DataFrame(selection.value)
df.columns = crop_names
planted_names = ["p. " + x for x in crop_names] 
df = df.reindex(df.columns.tolist() + planted_names + ["energy expended", "daily expense", "daily revenue", "cash on hand"], axis=1)
df = df.fillna(0)

# Energy Results
for i, results_col in enumerate(planted_names):
    for j, row in enumerate(range(len(df))):
        if crops[i].regrowth == sys.maxsize:
            watering_start_day = j-t[i]+1
        else:
            watering_start_day = 0
        df[results_col].iloc[row] += df[df.columns[i]][max(0, watering_start_day):j+1].sum() #TODO: fix calculation for regrowing crops
for i, row in enumerate(df["energy expended"]):
    energy_expended = 0
    for j, crop in enumerate(crop_names):
        energy_expended += f[j] * df[crop].iloc[i]
    for j, planted_crop in enumerate(planted_names):
        energy_expended += w[j] * df[planted_crop].iloc[i]
    df["energy expended"].iloc[i] = energy_expended
    
# Benchmarking
crop_id = 0
crop_name = crop_names[crop_id]
def get_benchmark_planting_quantity(cash_budget, energy_budget):
    energy_constrained_count = energy_budget // (f[crop_id]+w[crop_id])
    cash_constrained_count = cash_budget // b[crop_id]
    return min(energy_constrained_count, cash_constrained_count)
def calculate_next_benchmark_row(i, df):
    if i == 0:
        previous_row = pd.Series().reindex_like(df.iloc[0]).fillna(0)
        previous_row["cash on hand"] = g
    else:
        previous_row = df.iloc[i-1]
    
    new_row = pd.Series().reindex_like(previous_row)


    if i % t[crop_id] == 0:
        crop_revenue = df[crop_name].iloc[i-t[crop_id]] * s[crop_id]
        cash_budget = crop_revenue + previous_row["cash on hand"]
        energy_budget = e
        new_row[crop_name] = get_benchmark_planting_quantity(cash_budget, energy_budget)
        remaining_energy = e - new_row[crop_name] * (w[crop_id] + f[crop_id])
        new_row["foraging"] = remaining_energy // f[-1]
        new_row["energy expended"] = new_row["foraging"] * f[-1] + new_row[crop_name] * (w[crop_id] + f[crop_id])
        new_row["daily expense"] = new_row[crop_name] * b[crop_id]
        new_row["daily revenue"] = crop_revenue + new_row["foraging"] * s[-1] 
        new_row["cash on hand"] = previous_row["cash on hand"] + new_row["daily revenue"] - new_row["daily expense"]
        new_row["p. " + crop_name] = new_row[crop_name]

    else:
        crop_revenue = 0
        cash_budget = crop_revenue + previous_row["cash on hand"]
        energy_budget = e
        new_row[crop_name] = 0
        new_row["p. " + crop_name] = previous_row["p. " + crop_name]
        remaining_energy = e - new_row[crop_name] * (w[crop_id] + f[crop_id]) - previous_row["p. " + crop_name] * w[crop_id]
        new_row["foraging"] = remaining_energy // f[-1]
        new_row["energy expended"] = new_row["foraging"] * f[-1] + new_row["p. " + crop_name] * (w[crop_id])
        new_row["daily expense"] = new_row[crop_name] * b[crop_id]
        new_row["daily revenue"] = crop_revenue + new_row["foraging"] * s[-1] 
        new_row["cash on hand"] = previous_row["cash on hand"] + new_row["daily revenue"] - new_row["daily expense"]
    pass
    return new_row


benchmark_df = pd.DataFrame().reindex_like(df).fillna(0)
for i, row in enumerate(df.iterrows()):
    benchmark_df.iloc[i] = calculate_next_benchmark_row(i, benchmark_df)
#print(benchmark_df[["jazz", "foraging", "p. jazz", "energy expended", "daily expense", "daily revenue", "cash on hand"]])
pass




# Budget Results
current_budget = g
for i, row in df.iterrows():
    row["daily expense"] = np.sum(np.multiply(np.array(row[crop_names]), b))
    for j, crop in enumerate(crop_names):
        if i - t[j] >= 0:
            k = i - t[j]
            while k >= 0:
                row["daily revenue"] += df[crop].iloc[k] * s[j] 
                k -= crops[j].regrowth
        # revenue_relevancy[k,l] = 1 + (i - k - t[l]) // crops[l].regrowth

    if i == 0:
        row["cash on hand"] = 500 + row["daily revenue"] - row["daily expense"]

    else:
        row["cash on hand"] = df["cash on hand"].iloc[i-1] + row["daily revenue"] - row["daily expense"]

print(df)

df[["cash on hand", "daily revenue", "daily expense"]].plot()
planted_crop_names = ["p. " + crop for crop in crop_names]
display_columns = list(set(planted_crop_names) - set(["p. foraging"])) + ["foraging"]
df[display_columns].plot()
plt.show()


print()
