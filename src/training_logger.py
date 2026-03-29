#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练指标收集器，记录 BCD/Surrogate/Joint 三阶段迭代指标和快照。"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON 编码器：支持 numpy 数组和标量。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class TrainingLogger:
    """轻量级训练指标收集器。

    Attributes:
        metrics: 各阶段迭代指标列表。
        snapshots: x/pg/lambda 变量快照。
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {
            "bcd": [],
            "surrogate": {},  # unit_id -> list
            "joint": [],
        }
        self.snapshots: Dict[str, list] = {
            "x": [],
            "pg": [],
            "lambda": [],
        }
        self._start_time: float = time.time()

    # ------------------------------------------------------------------ #
    #  BCD 阶段
    # ------------------------------------------------------------------ #
    def log_bcd_iter(
        self,
        iter: int,
        obj_primal: float,
        obj_dual: float,
        obj_opt: float,
        rho_primal: float,
        rho_dual: float,
        rho_opt: float,
        nn_loss: Optional[float] = None,
    ) -> None:
        """记录 BCD 阶段单次迭代指标。"""
        record = {
            "iter": iter,
            "obj_primal": float(obj_primal),
            "obj_dual": float(obj_dual),
            "obj_opt": float(obj_opt),
            "rho_primal": float(rho_primal),
            "rho_dual": float(rho_dual),
            "rho_opt": float(rho_opt),
            "timestamp": time.time() - self._start_time,
        }
        if nn_loss is not None:
            record["nn_loss"] = float(nn_loss)
        self.metrics["bcd"].append(record)

    # ------------------------------------------------------------------ #
    #  Surrogate 阶段
    # ------------------------------------------------------------------ #
    def log_surrogate_iter(
        self,
        unit_id: int,
        iter: int,
        obj_primal: float,
        obj_dual: float,
        obj_opt: float,
        rho_primal: float,
        rho_dual: float,
        rho_opt: float,
        obj_dual_pg: Optional[float] = None,
        obj_dual_x: Optional[float] = None,
        obj_dual_coc: Optional[float] = None,
        rho_dual_pg: Optional[float] = None,
        rho_dual_x: Optional[float] = None,
        rho_dual_coc: Optional[float] = None,
        alpha_mean: Optional[float] = None,
        beta_mean: Optional[float] = None,
        gamma_mean: Optional[float] = None,
        delta_mean: Optional[float] = None,
        mu_mean: Optional[float] = None,
        nn_loss: Optional[float] = None,
    ) -> None:
        """记录 Surrogate 阶段单机组单次迭代指标。"""
        key = str(unit_id)
        if key not in self.metrics["surrogate"]:
            self.metrics["surrogate"][key] = []
        record = {
            "iter": iter,
            "obj_primal": float(obj_primal),
            "obj_dual": float(obj_dual),
            "obj_opt": float(obj_opt),
            "rho_primal": float(rho_primal),
            "rho_dual": float(rho_dual),
            "rho_opt": float(rho_opt),
            "timestamp": time.time() - self._start_time,
        }
        if obj_dual_pg is not None:
            record["obj_dual_pg"] = float(obj_dual_pg)
        if obj_dual_x is not None:
            record["obj_dual_x"] = float(obj_dual_x)
        if obj_dual_coc is not None:
            record["obj_dual_coc"] = float(obj_dual_coc)
        if rho_dual_pg is not None:
            record["rho_dual_pg"] = float(rho_dual_pg)
        if rho_dual_x is not None:
            record["rho_dual_x"] = float(rho_dual_x)
        if rho_dual_coc is not None:
            record["rho_dual_coc"] = float(rho_dual_coc)
        if alpha_mean is not None:
            record["alpha_mean"] = float(alpha_mean)
        if beta_mean is not None:
            record["beta_mean"] = float(beta_mean)
        if gamma_mean is not None:
            record["gamma_mean"] = float(gamma_mean)
        if delta_mean is not None:
            record["delta_mean"] = float(delta_mean)
        if mu_mean is not None:
            record["mu_mean"] = float(mu_mean)
        if nn_loss is not None:
            record["nn_loss"] = float(nn_loss)
        self.metrics["surrogate"][key].append(record)

    # ------------------------------------------------------------------ #
    #  Joint 阶段
    # ------------------------------------------------------------------ #
    def log_joint_iter(
        self,
        iter: int,
        obj_primal: float,
        obj_dual: float,
        obj_opt: float,
        rho_primal: float,
        rho_dual: float,
        rho_opt: float,
        integrality: Optional[float] = None,
        nn_loss: Optional[float] = None,
        surr_nn_loss: Optional[float] = None,
    ) -> None:
        """记录 Joint 阶段单次迭代指标。"""
        record = {
            "iter": iter,
            "obj_primal": float(obj_primal),
            "obj_dual": float(obj_dual),
            "obj_opt": float(obj_opt),
            "rho_primal": float(rho_primal),
            "rho_dual": float(rho_dual),
            "rho_opt": float(rho_opt),
            "timestamp": time.time() - self._start_time,
        }
        if integrality is not None:
            record["integrality"] = float(integrality)
        if nn_loss is not None:
            record["nn_loss"] = float(nn_loss)
        if surr_nn_loss is not None:
            record["surr_nn_loss"] = float(surr_nn_loss)
        self.metrics["joint"].append(record)

    # ------------------------------------------------------------------ #
    #  快照
    # ------------------------------------------------------------------ #
    def snapshot(
        self,
        stage: str,
        iter: int,
        x: Optional[np.ndarray] = None,
        pg: Optional[np.ndarray] = None,
        lambda_: Optional[Any] = None,
    ) -> None:
        """保存变量快照（仅 sample_id=0，每5轮记录一次）。

        Args:
            stage: 阶段名 ('bcd', 'surrogate', 'joint')
            iter: 当前迭代编号
            x: 二进制决策变量
            pg: 出力变量
            lambda_: 对偶变量
        """
        if iter % 5 != 0 and iter != 0:
            return
        meta = {"stage": stage, "iter": iter}
        if x is not None:
            self.snapshots["x"].append({**meta, "data": np.array(x, dtype=float)})
        if pg is not None:
            self.snapshots["pg"].append({**meta, "data": np.array(pg, dtype=float)})
        if lambda_ is not None:
            # lambda_ 可能是 dict 或 ndarray
            if isinstance(lambda_, dict):
                lam_data = lambda_.get("lambda_power_balance", None)
                if lam_data is not None:
                    lam_data = np.array(lam_data, dtype=float)
            else:
                lam_data = np.array(lambda_, dtype=float)
            if lam_data is not None:
                self.snapshots["lambda"].append({**meta, "data": lam_data})

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """保存全部指标和快照到 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 将 snapshots 中的 ndarray 转为 list
        snapshots_ser = {}
        for key, snap_list in self.snapshots.items():
            snapshots_ser[key] = []
            for snap in snap_list:
                entry = {k: v for k, v in snap.items() if k != "data"}
                if "data" in snap:
                    entry["data"] = snap["data"].tolist() if isinstance(snap["data"], np.ndarray) else snap["data"]
                snapshots_ser[key].append(entry)

        payload = {
            "metrics": self.metrics,
            "snapshots": snapshots_ser,
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=_NumpyEncoder, ensure_ascii=False, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """从 JSON 加载指标和快照。"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.metrics = payload.get("metrics", self.metrics)

        # 恢复 snapshots 中的 list → ndarray
        raw_snapshots = payload.get("snapshots", {})
        self.snapshots = {"x": [], "pg": [], "lambda": []}
        for key in ("x", "pg", "lambda"):
            for snap in raw_snapshots.get(key, []):
                entry = {k: v for k, v in snap.items() if k != "data"}
                if "data" in snap:
                    entry["data"] = np.array(snap["data"], dtype=float)
                self.snapshots[key].append(entry)
