# 时序耦合约束实现总结

## 📝 修改概述

成功将 `uc_NN_subproblem.py` 从单一聚合约束改进为**时序耦合约束（Temporal Coupling Constraints）**形式。

---

## 🔄 主要修改

### 1. **约束形式对比**

**原始形式（单一聚合）**：
```
Σ(αₜ × xₜ) ≤ β  （1个约束/机组）
```

**新形式（时序耦合）**：
```
αₜ × x_t + βₜ × x_{t+1} ≤ γₜ  （T-1个约束/机组）
```

**优势**：
- ✅ 学习相邻时段的关系
- ✅ 模拟最小运行/停机时间的软约束
- ✅ 限制开关机频率
- ✅ 在整数性差的时段附近加强约束

---

## 🛠️ 代码修改详情

### 修改1: `SubproblemSurrogateNet` 类

**文件位置**: 第439-516行

**改动**:
- 输出从 `(alpha, beta)` 改为 `(alphas, betas, gammas)`
- 参数数量从 `(T+1)` 增加到 `3×(T-1)`
- 添加特征提取网络，三个独立的输出层

**代码结构**:
```python
class SubproblemSurrogateNet(nn.Module):
    def __init__(self, input_dim, T):
        self.num_coupling_constraints = T - 1
        self.feature_extractor = Sequential(...)
        self.alpha_net = Linear(...)  # 输出(T-1,)
        self.beta_net = Linear(...)   # 输出(T-1,)
        self.gamma_net = Sequential(..., Softplus())  # 输出(T-1,)非负
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return alphas, betas, gammas  # 三个(T-1,)张量
```

---

### 修改2: `SubproblemSurrogateTrainer.__init__`

**文件位置**: 第550-620行

**改动**:
- 存储结构从标量/向量改为矩阵
- `alpha_values`: `(n, T)` → `(n, T-1)`
- `beta_values`: `(n,)` → `(n, T-1)`
- 新增 `gamma_values`: `(n, T-1)`
- `mu`: `(n,)` → `(n, T-1)` - 每个时序约束一个对偶变量

---

### 修改3: `iter_with_primal_block`

**文件位置**: 第779-885行

**关键改动**:
```python
# 旧：单个约束
surrogate_lhs = Σ(alpha[t] * x[t])
model.addConstr(surrogate_lhs <= beta)

# 新：T-1个时序耦合约束
for t in range(T-1):
    coupling_lhs = alphas[t] * x[t] + betas[t] * x[t+1]
    model.addConstr(surrogate_viols[t] >= coupling_lhs - gammas[t])
```

**目标函数**:
```python
obj_primal = rho_primal * Σ(surrogate_viols[t])
obj_opt = rho_opt * Σ(surrogate_abs_vals[t] * mu_vals[t])
```

---

### 修改4: `iter_with_dual_block`

**文件位置**: 第886-940行

**改动**:
- 从单个对偶变量求解改为循环求解T-1个对偶变量
- 每个时序约束独立求解其对偶变量

```python
for t in range(T-1):
    coupling_viol = |alphas[t]*x[t] + betas[t]*x[t+1] - gammas[t]|
    min coupling_viol * mu_t
```

---

### 修改5: `loss_function_differentiable`

**文件位置**: 第945-1046行

**核心损失计算**:
```python
# obj_primal: 时序约束违反
for t in range(T-1):
    coupling_viol = ReLU(alphas[t]*x[t] + betas[t]*x[t+1] - gammas[t])
    obj_primal += coupling_viol

# obj_opt: 互补松弛
for t in range(T-1):
    obj_opt += |alphas[t]*x[t] + betas[t]*x[t+1] - gammas[t]| * mu[t]

# obj_dual: 对偶可行性
for t in range(T):
    dual_expr = cost - lambda[t]
    if t < T-1:
        dual_expr += alphas[t] * mu[t]  # 当前时段贡献
    if t > 0:
        dual_expr += betas[t-1] * mu[t-1]  # 下一时段贡献
```

---

### 修改6: `iter_with_surrogate_nn`

**文件位置**: 第1048-1087行

**改动**:
```python
# 前向传播
alphas, betas, gammas = self.surrogate_net(features)

# 计算loss
loss = self.loss_function_differentiable(
    sample_id, alphas, betas, gammas, device
)

# 更新存储
self.alpha_values[sample_id] = alphas.detach().cpu().numpy()
self.beta_values[sample_id] = betas.detach().cpu().numpy()
self.gamma_values[sample_id] = gammas.detach().cpu().numpy()
```

---

### 修改7: `cal_viol`

**文件位置**: 第1089-1132行

**改动**:
- 循环计算T-1个时序约束的违反量
- 对偶约束考虑相邻时段的贡献

---

### 修改8: `iter` (主训练循环)

**文件位置**: 第1136-1198行

**改动**:
- 调用新方法签名
- `mu[sample_id]` 现在是 `(T-1,)` 数组，使用 `np.maximum` 处理

---

### 修改9: 辅助方法

**`get_surrogate_params`** (第1200-1218行):
- 返回值从 `(alpha, beta)` 改为 `(alphas, betas, gammas)`

**`save/load`** (第1220-1258行):
- 添加 `gamma_values` 到state字典
- 添加 `num_coupling_constraints` 元数据

---

## 📊 参数数量对比

| 组件 | 原始 | 新版 | 增加量 |
|------|------|------|--------|
| 网络输出维度 | T+1 | 3×(T-1) | ~2×T |
| 存储空间（每样本） | T+1 | 3×(T-1) | ~2×T |
| 对偶变量（每样本） | 1 | T-1 | T-2 |
| 约束数量（优化） | 1 | T-1 | T-2 |

对于 T=8:
- 网络输出: 9 → 21 参数
- 约束数量: 1 → 7 个

---

## ✅ 语法检查结果

```bash
$ python3 -m py_compile src/uc_NN_subproblem.py
✓ 编译通过，无语法错误
```

---

## 🧪 测试建议

由于环境缺少依赖（numpy, torch, gurobipy），无法运行完整测试。建议在有完整环境的机器上：

### 测试1: 基本功能
```python
# 创建简单数据
ppc = case14.case14()
active_set_data = [...]  # 3个样本

# 训练对偶预测器
lambda_predictor = train_dual_predictor_from_data(ppc, active_set_data)

# 训练时序耦合约束
trainer = train_subproblem_surrogate_from_data(
    ppc, active_set_data, unit_id=0,
    max_iter=5, nn_epochs=5
)

# 验证形状
assert trainer.alpha_values.shape == (n_samples, T-1)
assert trainer.beta_values.shape == (n_samples, T-1)
assert trainer.gamma_values.shape == (n_samples, T-1)
```

### 测试2: 约束有效性
```python
# 检查约束违反量
for sample_id in range(n_samples):
    x = trainer.x[sample_id]
    for t in range(T-1):
        lhs = alphas[t] * x[t] + betas[t] * x[t+1]
        assert lhs <= gammas[t] + 1e-6, f"约束{t}违反"
```

### 测试3: 对比实验
```python
# 对比单约束 vs 时序耦合
# - 整数性间隙
# - 约束违反率
# - 训练时间
```

---

## 🐛 已知问题与修复

### 问题1: 重复代码行
**位置**: 第1125-1132行  
**修复**: 删除重复的 `return` 语句

---

## 🎯 下一步工作

1. **安装依赖环境**
   ```bash
   pip install numpy torch gurobipy pypower
   ```

2. **运行完整测试**
   ```bash
   python3 test_temporal_coupling.py
   ```

3. **对比实验**
   - 在真实UC数据集上对比效果
   - 测量整数性间隙改善幅度

4. **可能的进一步改进**
   - 添加注意力机制识别关键时段
   - 动态调整约束数量
   - 结合物理约束先验

---

## 📁 文件清单

- ✅ `src/uc_NN_subproblem.py` - 主实现（已修改）
- ✅ `src/uc_NN_subproblem_original.py` - 原始备份
- ✅ `test_temporal_coupling.py` - 测试脚本（已创建）
- ✅ `TEMPORAL_COUPLING_SUMMARY.md` - 本文档

---

## 💡 设计理念

时序耦合约束的核心思想：

1. **时间相关性**：相邻时段的启停状态通常不是独立的
2. **软约束替代**：用学习的约束代替硬编码的最小运行时间
3. **灵活性**：神经网络可以根据负荷模式调整约束强度
4. **表达能力**：T-1个约束远比1个约束更能刻画复杂可行域

---

**实现者**: AI Assistant  
**日期**: 2026-02-20  
**版本**: v1.0-temporal-coupling
