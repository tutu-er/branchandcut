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

## 依赖

| 包 | 用途 |
|----|------|
| `gurobipy` | MIP / LP 求解（需有效 license） |
| `pypower` | 电网数据与潮流计算 |
| `torch` | 神经网络训练 |
| `numpy` / `scipy` | 数值计算 |
| `cvxpy`（可选） | 备选建模接口 |
