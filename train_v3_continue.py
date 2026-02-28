#!/usr/bin/env python3
"""
V3 代理约束模型持续训练脚本

功能:
  1. 加载已有预训练模型（热启动）
  2. 对全部 6 个机组继续 BCD 迭代
  3. 每 CHECKPOINT_EVERY 轮保存一次检查点（覆盖 surrogate_unit_g.pth）
  4. 输出含时间戳的详细日志，同时写入文件

用法:
    python train_v3_continue.py
"""

import os
import sys
import time
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_JSON = ROOT / 'result' / 'active_sets_case30_20251223_002959.json'
MODEL_DIR = ROOT / 'result' / 'subproblem_models'

# ── 训练超参数 ────────────────────────────────────────────────────────────────
T_DELTA          = 1.0
BCD_ITERS_PER_UNIT = 300  # 每个机组继续 BCD 迭代次数（~1.75h）
NN_EPOCHS        = 20     # 每次 BCD 迭代内 NN 训练轮数
CHECKPOINT_EVERY = 10     # 每 N 次 BCD 迭代保存一次检查点

# ── 日志配置 ──────────────────────────────────────────────────────────────────
RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = ROOT / 'result' / f'train_v3_continue_{RUN_STAMP}.log'

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('train_v3')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    # 控制台 handler
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler
    os.makedirs(MODEL_DIR, exist_ok=True)
    fh = logging.FileHandler(str(LOG_FILE), encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = _setup_logger()


# ── 导入项目模块 ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))

import pypower.case30
from pypower.ext2int import ext2int
from src.uc_NN_subproblem_v3 import (
    DualVariablePredictorTrainer,
    SubproblemSurrogateTrainer,
)


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_raw_samples(json_path: Path):
    """
    加载原始 JSON 样本，保留 'active_set' 和 'lambda' 键。
    _initialize_solve 需要 active_set 来提取初始 x 值。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data['all_samples']
    for s in samples:
        s['pd_data'] = np.array(s['pd_data'])
    return samples


# ── 单机组继续训练 ────────────────────────────────────────────────────────────
def continue_train_unit(
    unit_id: int,
    ppc: dict,
    active_set_data: list,
    lambda_predictor,
) -> SubproblemSurrogateTrainer:
    """
    对单个机组热启动继续 BCD 训练。

    Returns:
        训练完成后的 SubproblemSurrogateTrainer
    """
    log.info(f"{'─'*55}")
    log.info(f"机组 {unit_id}  开始继续训练  "
             f"(BCD×{BCD_ITERS_PER_UNIT}, NN epochs×{NN_EPOCHS})")
    log.info(f"{'─'*55}")
    t_unit = time.time()

    # 1. 创建 trainer（_initialize_solve 用原始 active_set 提取初始 x）
    trainer = SubproblemSurrogateTrainer(
        ppc, active_set_data, T_DELTA, unit_id=unit_id,
        lambda_predictor=lambda_predictor,
        max_constraints=20,
    )

    # 2. 热启动：加载已有模型参数（NN 权重、rho、mu、alpha/beta/gamma/delta）
    model_path = MODEL_DIR / f'surrogate_unit_{unit_id}.pth'
    if model_path.exists():
        trainer.load(str(model_path))
        log.info(f"  热启动: 已加载 {model_path.name}")
        log.info(f"  加载后 rho_primal={trainer.rho_primal:.4f}, "
                 f"rho_dual={trainer.rho_dual:.4f}, rho_opt={trainer.rho_opt:.4f}")
    else:
        log.info(f"  警告: {model_path.name} 不存在，从头开始训练")

    # 3. 按批次继续 BCD 迭代，每批次结束后保存检查点
    remaining = BCD_ITERS_PER_UNIT
    batch_no  = 0
    while remaining > 0:
        batch_size = min(CHECKPOINT_EVERY, remaining)
        batch_no  += 1
        log.info(f"\n  [批次 {batch_no}] 机组{unit_id}  BCD 迭代 {batch_size} 次 ...")
        t_batch = time.time()

        trainer.iter(max_iter=batch_size, nn_epochs=NN_EPOCHS)

        elapsed = time.time() - t_batch

        # 保存检查点（覆盖同名文件）
        trainer.save(str(model_path))
        log.info(f"  [检查点] 机组{unit_id} 已保存 → {model_path.name}  "
                 f"(批次耗时 {elapsed:.1f}s, "
                 f"rho_dual={trainer.rho_dual:.2f}, rho_opt={trainer.rho_opt:.4f})")
        remaining -= batch_size

    total_unit = time.time() - t_unit
    log.info(f"机组 {unit_id} 全部训练完成, 耗时 {total_unit/60:.1f} min")
    return trainer


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    log.info("=" * 60)
    log.info(f"V3 代理约束模型持续训练")
    log.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"日志文件: {LOG_FILE.name}")
    log.info("=" * 60)

    # 加载数据
    log.info(f"\n>>> 加载数据: {DATA_JSON.name}")
    active_set_data = load_raw_samples(DATA_JSON)
    n_samples = len(active_set_data)
    T = active_set_data[0]['pd_data'].shape[1]
    log.info(f"  样本数={n_samples}, T={T}, T_delta={T_DELTA}")

    # 加载 PyPower 案例
    ppc     = pypower.case30.case30()
    ppc_int = ext2int(ppc)
    ng      = ppc_int['gen'].shape[0]
    log.info(f"  案例: case30, ng={ng}, nb={ppc_int['bus'].shape[0]}")

    # 加载对偶变量预测器
    log.info(f"\n>>> 加载对偶预测器 ...")
    lambda_predictor = DualVariablePredictorTrainer(ppc, active_set_data, T_DELTA)
    pred_path = MODEL_DIR / 'dual_predictor.pth'
    if pred_path.exists():
        lambda_predictor.load(str(pred_path))
        log.info(f"  已加载 {pred_path.name}")
    else:
        log.info(f"  警告: {pred_path.name} 不存在，将使用随机初始化的预测器")

    # 训练配置摘要
    log.info(f"\n{'='*60}")
    log.info(f"训练配置")
    log.info(f"{'='*60}")
    log.info(f"  机组数量:       {ng}")
    log.info(f"  BCD 迭代/机组:  {BCD_ITERS_PER_UNIT}")
    log.info(f"  NN epochs/BCD:  {NN_EPOCHS}")
    log.info(f"  检查点间隔:     每 {CHECKPOINT_EVERY} 次 BCD 迭代")
    log.info(f"  预计总 BCD:     {ng * BCD_ITERS_PER_UNIT} 次")
    log.info(f"{'='*60}\n")

    # 逐机组继续训练
    trainers   = {}
    unit_times = []

    for g in range(ng):
        try:
            trainer = continue_train_unit(g, ppc, active_set_data, lambda_predictor)
            trainers[g]  = trainer
            unit_times.append(time.time() - t_start)

            elapsed_total = time.time() - t_start
            log.info(f"\n>>> 进度: {g+1}/{ng} 机组完成, "
                     f"累计耗时 {elapsed_total/60:.1f} min")

        except Exception as e:
            log.error(f"机组 {g} 训练出错: {e}", exc_info=True)
            log.info("跳过该机组，继续下一个 ...")

    # 训练完成汇总
    total_elapsed = time.time() - t_start
    log.info(f"\n{'='*60}")
    log.info(f"全部训练完成!")
    log.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"总耗时:   {total_elapsed/60:.1f} min  ({total_elapsed/3600:.2f} h)")
    log.info(f"成功训练: {len(trainers)}/{ng} 个机组")
    log.info(f"模型保存: {MODEL_DIR}")
    log.info(f"{'='*60}")


if __name__ == '__main__':
    main()
