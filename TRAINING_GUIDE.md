# 时序耦合约束训练指南

## ✅ 已完成的工作

1. **代码修改完成**
   - ✓ `uc_NN_subproblem.py` 已改为时序耦合约束版本
   - ✓ 原始文件备份为 `uc_NN_subproblem_original.py`
   - ✓ 语法检查通过

2. **约束形式**
   ```
   旧：Σ(αₜ × xₜ) ≤ β         (1个约束/机组)
   新：αₜ×x_t + βₜ×x_{t+1} ≤ γₜ  (T-1个约束/机组)
   ```

3. **数据准备**
   - ✓ 有39个样本的active_sets数据
   - ✓ 包含负荷和机组承诺矩阵

4. **训练脚本**
   - ✓ `run_training.py` - 自动依赖检测
   - ✓ `test_temporal_coupling.py` - 基础测试

---

## 🚀 如何运行训练

### 方法1：在本地Python环境运行

```bash
# 1. 安装依赖（如果未安装）
pip install numpy torch gurobipy PYPOWER

# 2. 运行训练
cd /home/node/.openclaw/workspace/branchandcut
python3 run_training.py
```

### 方法2：使用Docker（需要安装依赖）

```bash
# 进入容器
docker exec -it <container_id> bash

# 安装依赖
pip3 install numpy torch gurobipy PYPOWER

# 运行训练
cd /home/node/.openclaw/workspace/branchandcut
python3 run_training.py
```

---

## 📊 训练参数

`run_training.py` 中的关键参数：

- **样本数量**：默认使用前10个样本（快速测试）
- **机组**：默认训练机组0
- **迭代次数**：5次BCD迭代
- **神经网络epochs**：每次迭代5个epoch
- **对偶预测器epochs**：10个epoch

修改这些参数可以调整训练速度和效果。

---

## 📁 输出文件

训练完成后会生成：

- `result/dual_predictor.pth` - 对偶变量预测器
- `result/temporal_coupling_unit0.pth` - 机组0的时序耦合模型

---

## 🔍 验证结果

训练脚本会自动：

1. 检查参数形状
2. 显示示例约束
3. 计算整数性指标
4. 保存模型

---

## ❌ 当前限制

Docker环境中**没有安装Python依赖**，需要：

1. 在本地安装依赖后运行
2. 或手动在Docker中安装依赖

---

## 📝 主要修改总结

详见 `TEMPORAL_COUPLING_SUMMARY.md`

---

## 🆘 如需帮助

如果遇到问题：

1. 检查依赖是否安装
2. 查看错误信息
3. 检查数据文件路径
4. 确认Gurobi许可证

---

**创建时间**: 2026-02-20  
**修改者**: AI Assistant
