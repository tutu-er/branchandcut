# Branch and Cut — 学习加速机组组合求解

本项目研究如何用**学习方法加速电力系统机组组合（Unit Commitment, UC）混合整数规划**的求解。测试系统为 IEEE 39 节点新英格兰系统（10 台机组）。

## 核心思路

1. **采样求解**：对随机扰动的负荷场景批量求解 UC，记录最优解的活跃约束集合（Active Set）。
2. **代理约束训练**：从历史最优解学习每台机组的三时段代理约束 `α·x[t] + β·x[t+1] + γ·x[t+2] ≤ δ`，压缩搜索空间。
3. **BCD 迭代**：用块坐标下降（BCD）交替优化二进制调度变量和对偶变量，配合神经网络更新代理参数。
4. **可行性泵**：从 LP 松弛出发，结合代理约束和子问题求解恢复整数可行解。

## 目录结构

```
src/
  case39_pypower.py       IEEE 39 节点系统数据
  case39_uc_data.py       UC 机组成本与运行参数
  uc_gurobipy.py          UC 完整 MIP 求解器（Gurobi）
  ed_gurobipy.py          经济调度 LP 求解器（Gurobi）
  ActiveSetLearner.py     活跃约束集采样与统计
  uc_NN_subproblem_v3.py  V3 三时段代理约束训练（主推）
  uc_NN_BCD.py            BCD + 神经网络主代理训练
  feasibility_pump.py     整数可行解恢复（可行性泵）
  sparse_surrogate_mining.py 稀疏参数化约束挖掘与可选软注入
  sparse_support_discovery.py x[g,t] 级别的稀疏支持集发现
  sparse_constraint_templates.py 支持集模板构造与软注入
tests/
  run_uc_case39.py        端到端 UC 测试
run_training.py           多模式训练入口
```

## 快速开始

```bash
# 1. 生成活跃约束样本数据
python src/ActiveSetLearner.py

# 2. 训练代理约束（三种模式）
#    MODE='surrogate'  仅训练 V3 代理约束
#    MODE='bcd'        仅 BCD 主代理训练
#    MODE='both'       BCD → lambda 注入 → surrogate 联合训练
#    RUN_FP=True       训练后运行可行性泵测试
python run_training.py

# 3. 运行基准测试
python tests/run_uc_case39.py
```

## 可选：稀疏参数化约束

`src/sparse_surrogate_mining.py` 提供一个与现有 BCD 主流程解耦的可选模块，用于：

- 从历史样本中的 `Pd`, `pg_lp`, `pg_true`, `x_lp`, `x_true` 挖掘候选约束；
- 用简单的 `Pd` 非线性特征拟合右端项 `rhs(Pd)`；
- 通过“真实解满足率 + LP 违反量 + greedy 去冗余”筛出少量高价值约束；
- 在 `feasibility_pump` / `UnifiedSurrogateManager` 中以软约束形式可选注入。

默认情况下该模块不会改变现有 `run_training.py`、`run_test.py` 等入口的行为，只有在显式传入稀疏约束库时才会启用。

推荐验证顺序：

1. 离线统计：先用 `mine_sparse_surrogate_library(...)` 或
   `mine_sparse_surrogate_library_from_agent(...)` 从已有样本中挖掘约束库。
2. 质量检查：用 `evaluate_library_on_samples(...)` 和 `summarize_library(...)`
   查看真实解满足率、LP 违反量差异和最终保留数量。
3. 在线注入：将得到的 `SparseSurrogateLibrary` 作为可选参数传入
   `feasibility_pump.recover_integer_solution(...)`，或在
   `UnifiedSurrogateManager(..., sparse_library=library)` 中启用软约束注入。

## 可选：稀疏支持集发现

如果目标不是直接学习完整约束参数，而是先找到高价值参与变量，
可以使用新的两阶段路径：

1. 用 `src/sparse_support_discovery.py` 从 `x_lp`, `x_true`, `Pd`
   中筛选高价值 `x[g,t]` 变量，并组合成少量跨时段支持集。
2. 用 `src/sparse_constraint_templates.py` 将支持集转成固定模板库。
3. 先把模板库作为软约束注入 `feasibility_pump` 或
   `UnifiedSurrogateManager` 做离线验证。
4. 如效果稳定，再把模板库通过 `Agent_NN_BCD(..., external_sparse_templates=...)`
   接回 BCD 参数学习阶段。

这条路径默认也不会改变现有入口行为，只有显式提供模板库时才启用。

## 依赖

| 包 | 用途 |
|----|------|
| `gurobipy` | MIP / LP 求解（需有效 license） |
| `pypower` | 电网数据与潮流计算 |
| `torch` | 神经网络训练 |
| `numpy` / `scipy` | 数值计算 |
| `cvxpy`（可选） | 备选建模接口 |
