import cplex
# 初始化模型
prob = cplex.Cplex()
prob.objective.set_sense(prob.objective.sense.maximize)
obj = [1.0, 2.0, 3.0, 1.0]
# 决策变量取值范围上界
ub = [40.0, cplex.infinity, cplex.infinity, 3.0]
# 决策变量取值范围下界
lb = [0.0, 0.0, 0.0, 2.0]
varnames = ["x1", "x2", "x3", "x4"]
# 决策变量类型，C是数值，I是整数
types = 'CCCI'
prob.variables.add(obj=obj, ub=ub, lb=lb, types=types, names=varnames)
print(prob.variables.get_lower_bounds())
print(prob.variables.get_upper_bounds("x2"))
print(prob.variables.get_names())
#约束中<=(L),=(E)
senses = "LLE"
rhs = [20.0, 30.0, 0.0]
rownames = ["r1", "r2", "r3"]
rows = [[["x1", "x2", "x3", "x4"], [-1.0, 1.0, 1.0, 10.0]],
        [["x1", "x2", "x3", "x4"], [1.0, -3.0, 1.0, 0.0]],
        [["x1", "x2", "x3", "x4"], [0.0, 1.0, 0.0, -3.5]]]
prob.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rownames)
prob.solve()
# 查看目标函数值
print("目标函数值：",prob.solution.get_objective_value())
# 查看最优解
print("最优解：",prob.solution.get_values())