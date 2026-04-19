# 对偶预测器（Dual Predictor）：新旧模型差异说明

本文说明 **电价型对偶** `lambda_pg_electricity_price` 监督预测器中，**旧版默认实现**与 **Case118 新设定** 在结构、训练目标与工程行为上的区别，便于复现与排障。

---

## 1. 相同部分（未改动的假设）

| 项目 | 说明 |
|------|------|
| **输入** | 仍仅使用负荷侧可观测量：`get_feature_vector_from_sample` 给出的 **`load_data` 与 `renewable_data` 展平拼接**（或退化为净负荷）。**不把机组启停作为输入特征。** |
| **输出** | 每台机、每个时段的电价型对偶，展平维度为 `ng × T`（如 case118：`39×24=936`）。 |
| **标签来源** | 与原先一致：优先读样本中的 `lambda_pg_electricity_price`；缺失时由 ED 在隐含 `x` 上求解得到（标签仍可能含「负荷不可分辨」的波动，见文末局限）。 |

---

## 2. 旧模型（典型：`DualVariablePredictorNet` + 纯 MSE）

### 2.1 网络

- **结构**：多层全连接 MLP（`Linear` → `LayerNorm` → `LeakyReLU` → `Dropout`），输入为 **一次性拍扁** 的高维向量。
- **归纳偏置**：不显式区分「母线 × 时段」矩阵结构，时段与空间维度混在同一向量中。

### 2.2 训练

- **损失**：`nn.MSELoss()`，在 **原始 λ 物理尺度** 上直接回归。
- **学习率调度**：`ReduceLROnPlateau` 作用于上述 MSE。
- **日志中的 `Loss`**：与验证脚本里的 **全样本、全输出维平均 MSE** 同量级（例如数百量级与 RMSE≈√MSE 一致）。

### 2.3 典型现象（为何容易像「列均值」）

在仅用负荷/可再生、而真值 λ 仍依赖数据里未作为特征给出的因素时，**条件期望**可能接近 **按输出维的全局均值**；纯 MSE 下网络易收敛到「整体平」的解，表现为：

- 验证集 MSE 与「用训练集列均值当预测」的基线 **差距很小**；
- 按维 R² 接近 0、但 MAE 不一定极端大（尺度与长尾共同决定）。

---

## 3. 新模型（Case118 预设：`temporal_conv` + 标准化 + 组合损失）

### 3.1 网络

- **结构**：`DualVariablePredictorNetTemporalConv`  
  - 将拼接后的负荷/可再生向量视为形状 **`(2×nb, T)`**（前 `nb` 通道为各母线负荷，后 `nb` 通道为可再生），沿 **时段维 `T` 做 `Conv1d`**，再展平接 MLP 头输出 `ng×T`。
- **归纳偏置**：显式利用 **同一母线跨时段** 的相关性，更贴合「曲线随时间变化」的电价型对偶。

### 3.2 训练

1. **目标标准化（可选，`dual_normalize_targets=True`）**  
   - 在 **训练集标签矩阵** 上，对每个输出维计算 `mean`、`std`（`std` 带下界，避免除零）。  
   - 网络在 **标准化空间** 上拟合；**推理**时在 `predict` 内做 **反标准化** 回到物理 λ。

2. **主拟合损失**：`SmoothL1`（Huber 风格，`beta=dual_smooth_l1_beta`），对异常大 λ 维更稳，减轻少数维主导梯度。

3. **形状损失（可选）**：`dual_cosine_loss_weight > 0` 时，在 **物理尺度** 上对预测与真值加 **`(1 - 余弦相似度)`** 的 batch 均值，鼓励整条 λ 向量与真值 **方向/波形** 一致，缓解退化为「只贴全局均值形状」。

4. **日志中的 `Loss`**：为 **`SmoothL1(标准化目标)` + 权重×余弦(物理空间)`** 的标量，**数值量级与旧版 MSE 不可直接对比**。

### 3.3 Checkpoint 额外字段

新权重文件除网络与优化器外，还可能包含：

- `dual_net_variant`、`dual_normalize_targets`、`dual_cosine_loss_weight`、`dual_smooth_l1_beta`
- `dual_y_mean`、`dual_y_std`（`numpy` 数组，用于反标准化）

**PyTorch 2.6+**：加载此类 checkpoint 时需 `weights_only=False`（仓库已在 `uc_NN_subproblem` 的对偶加载路径中处理）。

---

## 4. 如何对比「效果」

| 对比项 | 建议 |
|--------|------|
| **不要用训练日志里的 Loss 数值横向对比** | 旧：物理 MSE；新：标准化 SmoothL1 + 余弦，定义不同。 |
| **用同一 JSON、同一评估脚本** | 仓库根目录：`python validate_dual_predictor.py --model-dir <含 dual_predictor.pth 的目录> --active-set-json <同一文件> --val-fraction 0.2` |
| **看物理空间指标** | 脚本输出的 `mse_scalar_mean`、`rmse`、`mae_scalar_mean`、验证集 MSE 与 **列均值基线** 的相对关系。 |

在同一 case118 数据集上，新设定相对旧 MLP 的典型表现是：**物理 MSE/MAE 显著下降**，验证 MSE **明显低于** 列均值基线，按维 R² **明显升高**（具体数以你本机 `validate_dual_predictor` 输出为准）。

---

## 5. 配置入口

| 场景 | 做法 |
|------|------|
| **Case118 统一入口** | `run_training_case118.py` 顶部 **`CASE118_DUAL_PREDICTOR_*`** 常量 + `_configure_common()` 写入 `run_training.rt.DUAL_PREDICTOR_*`。 |
| **直接跑 `run_training.py`** | 修改模块级 `DUAL_PREDICTOR_NET_VARIANT`、`DUAL_PREDICTOR_NORMALIZE_TARGETS`、`DUAL_PREDICTOR_COSINE_LOSS_WEIGHT`、`DUAL_PREDICTOR_SMOOTH_L1_BETA`。 |

**`temporal_conv` 前提**：`input_dim == 2 × nb × T`，即样本需同时具备与 `(nb, T)` 一致的 `load_data` 与 `renewable_data` 拼接特征；否则实现会 **回退为 MLP** 并打印提示。

---

## 6. 局限与后续（与新旧无关的共性）

- 监督标签若在 ED 中依赖 **未作为网络输入的启停 `x`**，则存在 **负荷特征不可完全解释 λ** 的上限；新设定改善的是 **在可学信号下的拟合与优化**，不能从数学上消除所有歧义。
- 若在训练时用的 `mean/std` 覆盖 **全体样本**，而离线评估又做随机划分，则标准化统计量与「严格 K 折」理想流程略有差别；若需论文级严谨，可改为 **仅在训练折上估计 `mean/std` 并写入 checkpoint**。

---

## 7. 相关代码位置（便于维护）

| 内容 | 路径 |
|------|------|
| MLP 与 Temporal 网络、`train`/`predict`/`save`/`load` | `src/uc_NN_subproblem.py`（`DualVariablePredictorNet*`、`DualVariablePredictorTrainer` 及 `_dual_predictor_trainer_*`） |
| 训练入口聚合参数 | `src/uc_NN_subproblem.py`：`train_dual_predictor_from_data`；`run_training.py`：`run_surrogate` |
| Case118 预设常量 | `run_training_case118.py`：`CASE118_DUAL_PREDICTOR_*`、`_configure_common` |
| 离线指标脚本 | `validate_dual_predictor.py` |

---

*文档随实现演进可继续补充「消融实验建议」与「超参调优记录」。*
