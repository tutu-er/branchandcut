# V3改进版实现计划

## 核心改进

### 1. 三时段时序耦合约束
**当前V1/V2**: `αₜ×x[t] + βₜ×x[t+1] ≤ γₜ`  
**改进V3**: `αₜ×x[t] + βₜ×x[t+1] + γₜ×x[t+2] ≤ δₜ`

### 2. 敏感时段动态选择
**当前**: 固定T-1=23个约束  
**改进**: 识别x ∈ [0.1, 0.9]的时段，动态生成5-15个约束

### 3. 自适应约束权重
为不同时段的约束学习不同权重

---

## 实现步骤

### Step 1: 修改网络架构
```python
class ThreeStepSurrogateNet(nn.Module):
    def __init__(self, input_dim, T, max_constraints=20):
        # 输出4个参数向量
        self.alpha_net = nn.Linear(hidden, max_constraints)
        self.beta_net = nn.Linear(hidden, max_constraints)
        self.gamma_net = nn.Linear(hidden, max_constraints)
        self.delta_net = nn.Sequential(
            nn.Linear(hidden, max_constraints),
            nn.Softplus()  # 确保δ非负
        )
```

### Step 2: 敏感时段识别
```python
def identify_sensitive_timesteps(x_vals, threshold_low=0.1, threshold_high=0.9):
    """
    识别整数性差的时段
    返回需要约束的时段索引列表
    """
    sensitive = []
    for t in range(len(x_vals) - 2):  # 三时段约束需要t+2
        # 检查t, t+1, t+2是否有整数性问题
        window = x_vals[t:t+3]
        if any(threshold_low < x < threshold_high for x in window):
            sensitive.append(t)
    return sensitive
```

### Step 3: 三时段约束生成
```python
# 在iter_with_primal_block中
for i, t in enumerate(sensitive_timesteps):
    model.addConstr(
        alphas[i] * x[t] + betas[i] * x[t+1] + gammas[i] * x[t+2] 
        <= deltas[i] + surrogate_viols[i]
    )
```

### Step 4: 对偶变量更新
每个敏感时段的约束独立对偶变量

---

## 文件结构
- `src/uc_NN_subproblem_v3.py` — 新版本实现
- `run_training_v3.py` — V3训练脚本
- `compare_v1_v2_v3.py` — 三版本对比

## 预期效果
- 更精准的约束（三时段关系）
- 更少的约束数量（5-15 vs 23）
- 更好的整数性改进（目标 < 0.5 平均）
