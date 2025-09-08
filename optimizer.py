from ortools.linear_solver import pywraplp

def select_repairs(costs, budget):
    """Select a subset of repairs given costs and a budget.
    costs: list[float] (lower cost = cheaper to repair)
    budget: float
    returns: list[int] binary selection
    """
    n = len(costs)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError('OR-Tools solver not available')
    x = [solver.IntVar(0,1,f'x{i}') for i in range(n)]
    solver.Add(sum(costs[i]*x[i] for i in range(n)) <= budget)
    # objective: maximize number of repairs (could be changed)
    solver.Maximize(solver.Sum([x[i] for i in range(n)]))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return [0]*n
    return [int(x[i].solution_value()) for i in range(n)]
