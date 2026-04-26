# c_pg 与 NN-main：新旧训练方式差异说明

本文总结当前 `scripts/subproblem_loss_snapshot.py` 中新的 `c_pg` / `NN-main` 调参方式，与原来 `run_training.py` / `SubproblemSurrogateTrainer.iter()` 里完整 BCD 训练方式的区别。

## 1. 一句话区别

旧方式是“在线 BCD 联合训练”：每轮 BCD 中先解 primal / dual block，再马上用当前状态训练 `NN-main` 和 `c_pg`。

新方式是“冻结状态后的定向训练/调参”：先跑到指定 BCD 轮次前，保存同一个同步状态，然后分别从这个状态出发，单独测试 `NN-main` 或 `c_pg` 的训练策略。

## 2. 旧训练方式

旧流程入口通常是：

- `run_training.py` 的 `run_surrogate(...)`
- `src/uc_NN_subproblem.py` 的 `SubproblemSurrogateTrainer.iter(...)`
- 并行版本会从 `uc_NN_subproblem_parallel.py` 调同一套 trainer 逻辑

每个机组的 BCD 主循环大致是：

1. 用当前 `alpha/beta/gamma/delta/c_x/c_pg` 解 primal block，得到 `pg/x/coc/cpower`。
2. 用当前 primal 解 dual block，得到固有约束对偶 `lambda_inherent` 和代理约束对偶 `mu`。
3. 训练 `NN-main`，更新 `alpha/beta/gamma/delta/c_x`。
4. 训练 `c_pg`，更新 `pg_cost_net` 对应的 `c_pg`。
5. 刷新缓存输出，计算违反量并更新 `rho`。

旧方式中 `NN-main` 和 `c_pg` 都嵌在 BCD 迭代里，训练目标来自当前 BCD 状态下的可微 KKT 违反量：

- `NN-main` loss 只管 `alpha/beta/gamma/delta/c_x`，包括 `obj_primal`、`obj_dual_x`、`obj_opt` 和正则项。
- `c_pg` loss 只管 `pg` 驻点项，包括 `obj_dual_pg` 和 `c_pg` 自身正则。
- `c_pg` 一般要等 `iter_number >= pg_cost_start_round` 后才真正生效。

这种方式适合完整训练最终模型，但调参时有一个问题：每次试验都会受到前面 primal / dual 解、rho 更新、随机 batch 顺序、缓存状态的影响，很难只判断“这组 NN 参数本身好不好”。

## 3. 新训练方式

新入口统一在：

- `scripts/subproblem_loss_snapshot.py`
- `scripts/c_pg_loss_snapshot.py` 现在只是兼容包装，实际调用上面的脚本

新脚本有四个模式：

- `main_tune`：先完整跑若干 BCD 轮，然后停在指定轮次的 `NN-main` 之前，保存同步 bundle 和 `c_pg` snapshot，并可从同一状态跑若干 `NN-main` 小试验。
- `main_test`：加载 `main_tune` 保存的同步 bundle，直接调 `NN-main`。
- `test`：加载同一个同步 bundle 和 `c_pg` snapshot，只调 `c_pg`。
- `light_bake`：保留兼容用途，从头跑一段轻量完整训练并保存 snapshot / bundle。

关键变化是：`main_tune` 会先跑 `pre_iters = MAIN_TUNE_TARGET_ITER - 1` 个完整 BCD round，然后额外执行一次“只更新 primal / dual、不训练 NN”的 `_run_primal_dual_only_round(...)`，把状态固定在“下一次 NN-main 更新之前”。这个状态同时用于 `NN-main` 和 `c_pg`，所以两边调参看到的是同一组 `x/pg/mu/lambda_inherent/rho/缓存输出`。

## 4. 新 c_pg 训练

新 `c_pg` 调参在 `MODE="test"` 中进行。

它不是一上来就跑旧的 `iter_with_c_pg_nn(...)`，而是先构造解析目标：

```text
c_pg[t] = -pg_const[t]
```

其中 `pg_const[t]` 是 `pg` 驻点条件里除 `c_pg[t]` 以外的所有常数项，包括线性发电成本、电价型对偶 `lambda_vals`、`lambda_pg_lower/upper`、`lambda_ramp_up/down` 等。这样 `c_pg` 的目标非常直接：把 `pg_const[t] + c_pg[t]` 压到 0。

新 `c_pg` 训练的特点：

- 只训练 `pg_input_proj`、`pg_res_blocks`、`pg_cost_net`，主代理网络参数冻结。
- 可以先用 `direct_epochs` 对解析目标做监督式拟合。
- 支持 `MSE` 或 `SmoothL1` 风格目标、feature noise、AdamW、CosineAnnealingLR、梯度裁剪和 early stop。
- 可以选择性再跑一小段旧式 `iter_with_c_pg_nn(...)` 作为 `fine_epochs` 微调。
- 每个 trial 前都会恢复同一个 `base_state`，所以不同超参之间可比。
- 默认可跳过完整慢指标，用 `sum(abs(c_pg - analytic_target))` 作为快速 `obj_dual_pg` 代理指标选 best trial。

相比旧方式，新 `c_pg` 更像“先把 pg 驻点方程按解析答案拟合准”，旧方式则是在 BCD 循环里靠可微 KKT loss 慢慢推。

## 5. 新 NN-main 训练

新 `NN-main` 调参在 `MODE="main_test"` 中进行。

它同样从 `main_tune` 的冻结状态加载模型，但目标不是直接拿旧 loss 做很多轮，而是先构造 `alpha/beta/gamma/delta/c_x` 的直接目标。目标由 `_build_main_direct_targets(...)` 生成，大致思想是：

- 以当前网络输出为 anchor。
- 根据当前 `x`、`mu` 和活动/非活动代理约束，构造线性代理目标。
- 对 active 约束鼓励互补/贴边。
- 对 inactive 约束留 margin。
- 对系数、delta、cost 分别加 anchor 权重，避免解发散。
- 最后用加权最小二乘得到每个样本的目标 `alphas/betas/gammas/deltas/costs`。

新 `NN-main` 训练的特点：

- 训练 `main` 分支和 `x-cost` 分支，冻结 `c_pg`。
- 直接监督 `alpha/beta/gamma/delta/c_x` 到构造出的 proxy target。
- 可额外加入小权重 `proxy_kkt_loss_weight`，把原来的可微 KKT loss 混进去。
- `main` 参数和 `x-cost` 参数使用两个 AdamW optimizer，可分别设置 `direct_lr` 和 `direct_cost_lr`。
- 支持 mini-batch、feature noise、梯度裁剪、CosineAnnealingLR 和 direct MAE early stop。
- 可选择性再跑旧式 `iter_with_surrogate_nn(...)` 作为 `fine_epochs` 微调。
- 默认可跳过完整慢指标，用 `direct_mae` 和快速 KKT 组件做 trial 选择。

相比旧方式，新 `NN-main` 更像“先解一个局部 proxy target，再让网络拟合它”；旧方式则是直接在当前 BCD 状态上优化 `obj_primal + obj_dual_x + obj_opt + reg`。

## 6. 主要差异表

| 维度 | 旧方式：完整 BCD 在线训练 | 新方式：冻结状态定向训练 |
|---|---|---|
| 入口 | `run_training.py` / `trainer.iter()` | `scripts/subproblem_loss_snapshot.py` |
| 训练位置 | 每轮 BCD 内部 | 指定 BCD 轮次前的固定状态 |
| 状态是否固定 | 不固定，随 primal/dual/NN/rho 连续演化 | 固定，同一 bundle / snapshot 可重复加载 |
| `NN-main` 目标 | 可微 KKT loss：`primal + dual_x + opt + reg` | proxy target 监督拟合，可选少量 KKT loss |
| `c_pg` 目标 | 可微 KKT loss：`dual_pg + reg` | 解析目标 `c_pg = -pg_const`，可选旧式 fine tune |
| 参数冻结 | BCD 中按各函数内部逻辑分离参数 | 明确冻结另一支：调 `c_pg` 冻结 main，调 `NN-main` 冻结 `c_pg` |
| 调参可比性 | 较弱，不同试验会改变后续 BCD 轨迹 | 强，每个 trial 从同一个 base state 恢复 |
| 指标 | 完整违反量和日志 loss | 可用快速 proxy 指标，必要时再开完整 metrics |
| 主要用途 | 训练最终 surrogate 模型 | 定位 loss 问题、调学习率/正则/epoch、做消融 |

## 7. 该怎么用

如果目标是完整生成一套模型，仍然用旧的完整训练入口，因为它会覆盖 dual predictor、可选 unit predictor、所有机组 surrogate 和完整 BCD 轨迹。

如果目标是判断“为什么 `c_pg` 压不下去”或“NN-main 更新是否真的让指标变好”，优先用新脚本：

1. 先跑 `MODE="main_tune"`，得到同步 bundle 和 `c_pg` snapshot。
2. 调 `c_pg` 时跑 `MODE="test"`，只改 `TEST_C_PG_TRIALS`。
3. 调 `NN-main` 时跑 `MODE="main_test"`，只改 `TEST_MAIN_TRIALS`。
4. 找到稳定参数后，再把对应超参迁回完整 BCD 训练配置。

## 8. 注意事项

- 新方式不是完整训练的替代品，更像调参台和诊断工具。
- `main_test` / `test` 依赖 `main_tune` 产物；如果网络结构、样本数、case 或 feature 维度变了，应重新跑 `main_tune`。
- `TEST_C_PG_MODEL_SIZE_FOR_NEW_BUNDLE` 只对新建 bundle 有意义；加载已有 bundle 时网络宽度由 checkpoint 决定。
- `TEST_FULL_BASELINE_METRICS` / `TEST_FULL_FINAL_METRICS` 关闭时，结果里的指标是快速代理指标，不等同于完整 `cal_nn_logging_components()`。
- 如果最终要比较论文级效果，应回到同一完整训练配置下跑完整 BCD，再用统一评估脚本比较。
