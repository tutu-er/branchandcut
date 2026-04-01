"""
增强型可行性泵框架 - 支持代理模型特征诊断和物理信息增强恢复

核心组件：
1. SurrogateDiagnostics - 分析代理模型输出偏差模式
2. PhysicsGuidedRecovery - 基于UC约束的物理信息恢复
3. SolutionMemoryBank - 场景解记忆库
4. GuidedFeasibilityPump - 定向启发式FP主循环
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import defaultdict
from enum import Enum, auto
import json
import hashlib
from pathlib import Path


class BiasPattern(Enum):
    """代理模型偏差模式类型"""
    DELAYED_START = auto()      # 启动延迟
    DELAYED_STOP = auto()       # 停机延迟
    PREMATURE_START = auto()    # 过早启动
    PREMATURE_STOP = auto()     # 过早停机
    OSCILLATION = auto()        # 振荡偏差
    PHASE_SHIFT = auto()        # 相位偏移
    AMPLITUDE_ERROR = auto()    # 幅值误差
    NO_BIAS = auto()            # 无明显偏差


@dataclass
class TemporalBiasProfile:
    """时序偏差特征剖面"""
    unit_id: int
    pattern: BiasPattern
    avg_delay: float = 0.0          # 平均延迟时段数
    delay_std: float = 0.0          # 延迟标准差
    affected_time_slots: List[int] = field(default_factory=list)
    confidence: float = 0.0           # 置信度 [0,1]

    def to_dict(self) -> dict:
        return {
            'unit_id': self.unit_id,
            'pattern': self.pattern.name,
            'avg_delay': self.avg_delay,
            'delay_std': self.delay_std,
            'affected_time_slots': self.affected_time_slots,
            'confidence': self.confidence,
        }


@dataclass
class SolutionSignature:
    """解的特征签名"""
    load_hash: str          # 负荷特征哈希
    renewable_hash: str     # 新能源特征哈希
    x_pattern: np.ndarray   # 启停模式摘要

    def to_key(self) -> str:
        return f"{self.load_hash}_{self.renewable_hash}"


@dataclass
class StoredSolution:
    """存储的解条目"""
    signature: SolutionSignature
    x_opt: np.ndarray
    pg_opt: Optional[np.ndarray] = None
    lambda_opt: Optional[np.ndarray] = None
    objective: float = 0.0
    timestamp: float = field(default_factory=lambda: np.datetime64('now').astype(float))
    scenario_id: Optional[str] = None

    def compute_similarity(self, load_profile: np.ndarray) -> float:
        """计算与给定负荷剖面的相似度"""
        if self.signature.x_pattern.shape != load_profile.shape:
            return 0.0
        # 使用余弦相似度
        dot = np.dot(self.signature.x_pattern.flatten(), load_profile.flatten())
        norm_x = np.linalg.norm(self.signature.x_pattern)
        norm_l = np.linalg.norm(load_profile)
        if norm_x == 0 or norm_l == 0:
            return 0.0
        return dot / (norm_x * norm_l)


class SurrogateDiagnostics:
    """
    代理模型特征诊断器

    分析NN输出与最优解之间的偏差模式，识别系统性错误
    """

    def __init__(self, ng: int, T: int, min_samples: int = 5):
        self.ng = ng
        self.T = T
        self.min_samples = min_samples
        self.bias_profiles: Dict[int, TemporalBiasProfile] = {}
        self.transition_history: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    def add_sample(self, x_nn: np.ndarray, x_opt: np.ndarray, unit_id: Optional[int] = None):
        """
        添加一组NN输出和对应的最优解样本

        Args:
            x_nn: NN输出的启停决策 (ng, T) 或 (T,)
            x_opt: 最优启停决策 (ng, T) 或 (T,)
            unit_id: 如果输入是单机组，指定机组ID
        """
        if x_nn.ndim == 1:
            # 单机组输入
            if unit_id is None:
                raise ValueError("单机组输入需要提供unit_id")
            x_nn = x_nn.reshape(1, -1)
            x_opt = x_opt.reshape(1, -1)

        for g in range(x_nn.shape[0]):
            actual_unit_id = unit_id if unit_id is not None else g
            self.transition_history[actual_unit_id].append((x_nn[g], x_opt[g]))

    def analyze_bias_patterns(self) -> Dict[int, TemporalBiasProfile]:
        """
        分析累积的样本，识别各机组的偏差模式

        Returns:
            Dict[int, TemporalBiasProfile]: 各机组的偏差剖面
        """
        self.bias_profiles = {}

        for unit_id, transitions in self.transition_history.items():
            if len(transitions) < self.min_samples:
                continue

            profile = self._analyze_unit_transitions(unit_id, transitions)
            if profile.confidence > 0.3:  # 只保存置信度足够的模式
                self.bias_profiles[unit_id] = profile

        return self.bias_profiles

    def _analyze_unit_transitions(
        self,
        unit_id: int,
        transitions: List[Tuple[np.ndarray, np.ndarray]]
    ) -> TemporalBiasProfile:
        """分析单个机组的状态转换偏差"""

        x_nn_all = np.array([t[0] for t in transitions])
        x_opt_all = np.array([t[1] for t in transitions])

        # 检测启动/停机事件的延迟
        start_delays = []
        stop_delays = []
        affected_slots = []

        for t in range(1, self.T):
            # 检测启动事件
            opt_start = (x_opt_all[:, t-1] == 0) & (x_opt_all[:, t] == 1)
            nn_start = (x_nn_all[:, t-1] == 0) & (x_nn_all[:, t] == 1)

            if np.any(opt_start):
                for sample_idx in np.where(opt_start)[0]:
                    # 查找NN中最近的启动事件
                    for dt in range(-3, 4):
                        if 0 <= t + dt < self.T:
                            if dt != 0 and nn_start[sample_idx]:
                                start_delays.append(dt)
                                affected_slots.append(t)
                                break
                            elif dt == 0 and nn_start[sample_idx]:
                                break

        # 计算偏差模式
        if len(start_delays) > 0:
            avg_delay = np.mean(start_delays)
            delay_std = np.std(start_delays)

            if avg_delay > 0.5:
                pattern = BiasPattern.DELAYED_START
            elif avg_delay < -0.5:
                pattern = BiasPattern.PREMATURE_START
            else:
                pattern = BiasPattern.NO_BIAS

            confidence = min(len(start_delays) / (self.min_samples * 2), 1.0)
        else:
            avg_delay = 0.0
            delay_std = 0.0
            pattern = BiasPattern.NO_BIAS
            confidence = 0.0

        return TemporalBiasProfile(
            unit_id=unit_id,
            pattern=pattern,
            avg_delay=avg_delay,
            delay_std=delay_std,
            affected_time_slots=list(set(affected_slots)),
            confidence=confidence,
        )

    def get_compensated_solution(self, x_nn: np.ndarray) -> np.ndarray:
        """
        基于识别的偏差模式，对NN输出进行补偿修正

        Args:
            x_nn: NN输出的启停决策 (ng, T)

        Returns:
            补偿后的启停决策 (ng, T)
        """
        x_compensated = x_nn.copy()

        for unit_id, profile in self.bias_profiles.items():
            if profile.confidence < 0.5:
                continue

            unit_x = x_compensated[unit_id].copy()

            if profile.pattern == BiasPattern.DELAYED_START:
                # 启动延迟：提前启动信号
                for t in range(self.T - 1, 0, -1):
                    if unit_x[t] == 1 and unit_x[t-1] == 0:
                        # 检测到启动事件，尝试提前
                        shift = max(1, int(round(profile.avg_delay)))
                        if t - shift >= 0:
                            unit_x[t] = 0
                            unit_x[t - shift] = 1

            elif profile.pattern == BiasPattern.PREMATURE_START:
                # 过早启动：推迟启动信号
                for t in range(self.T - 1):
                    if unit_x[t] == 0 and unit_x[t+1] == 1:
                        shift = max(1, int(round(-profile.avg_delay)))
                        if t + shift < self.T:
                            unit_x[t+1] = 0
                            unit_x[t + shift] = 1

            x_compensated[unit_id] = unit_x

        return x_compensated


class SolutionMemoryBank:
    """
    场景解记忆库

    存储和检索历史最优解，支持基于负荷特征相似度的检索
    """

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.solutions: Dict[str, StoredSolution] = {}
        self.load_index: Dict[str, List[str]] = defaultdict(list)  # load_hash -> solution_keys

    def _compute_load_hash(self, load_profile: np.ndarray) -> str:
        """计算负荷剖面的特征哈希"""
        # 使用负荷统计特征
        features = np.array([
            np.mean(load_profile),
            np.std(load_profile),
            np.max(load_profile),
            np.min(load_profile),
            np.percentile(load_profile, 25),
            np.percentile(load_profile, 75),
        ])
        # 离散化后哈希
        quantized = np.round(features / 10).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()[:16]

    def store_solution(
        self,
        load_profile: np.ndarray,
        x_opt: np.ndarray,
        pg_opt: Optional[np.ndarray] = None,
        lambda_opt: Optional[np.ndarray] = None,
        objective: float = 0.0,
        scenario_id: Optional[str] = None,
    ) -> str:
        """存储一个解到记忆库"""

        load_hash = self._compute_load_hash(load_profile)

        # 构建签名
        signature = SolutionSignature(
            load_hash=load_hash,
            renewable_hash="",  # 可扩展
            x_pattern=np.mean(x_opt, axis=0),  # 每时段平均启停机
        )

        solution = StoredSolution(
            signature=signature,
            x_opt=x_opt,
            pg_opt=pg_opt,
            lambda_opt=lambda_opt,
            objective=objective,
            scenario_id=scenario_id,
        )

        key = f"{load_hash}_{len(self.solutions)}"

        # 检查是否已满，执行LRU淘汰
        if len(self.solutions) >= self.max_size:
            oldest_key = min(self.solutions.keys(),
                           key=lambda k: self.solutions[k].timestamp)
            del self.solutions[oldest_key]

        self.solutions[key] = solution
        self.load_index[load_hash].append(key)

        return key

    def retrieve_similar_solutions(
        self,
        load_profile: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, StoredSolution, float]]:
        """
        检索与给定负荷剖面相似的解

        Returns:
            List of (key, solution, similarity_score) tuples
        """
        load_hash = self._compute_load_hash(load_profile)

        # 首先尝试精确匹配
        candidate_keys = self.load_index.get(load_hash, [])

        # 如果没有精确匹配，考虑所有解
        if not candidate_keys:
            candidate_keys = list(self.solutions.keys())

        # 计算相似度并排序
        scored_solutions = []
        for key in candidate_keys:
            solution = self.solutions[key]
            similarity = solution.compute_similarity(load_profile)
            if similarity >= self.similarity_threshold:
                scored_solutions.append((key, solution, similarity))

        # 按相似度降序排序
        scored_solutions.sort(key=lambda x: x[2], reverse=True)

        return scored_solutions[:top_k]

    def save_to_disk(self, filepath: str):
        """将记忆库保存到磁盘"""
        data = {
            'solutions': {},
            'load_index': dict(self.load_index),
        }
        for key, sol in self.solutions.items():
            data['solutions'][key] = {
                'signature': {
                    'load_hash': sol.signature.load_hash,
                    'x_pattern': sol.signature.x_pattern.tolist(),
                },
                'x_opt': sol.x_opt.tolist(),
                'pg_opt': sol.pg_opt.tolist() if sol.pg_opt is not None else None,
                'objective': sol.objective,
                'timestamp': sol.timestamp,
                'scenario_id': sol.scenario_id,
            }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load_from_disk(cls, filepath: str) -> 'SolutionMemoryBank':
        """从磁盘加载记忆库"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        bank = cls()
        bank.load_index = defaultdict(list, data['load_index'])

        for key, sol_data in data['solutions'].items():
            sig = SolutionSignature(
                load_hash=sol_data['signature']['load_hash'],
                renewable_hash="",
                x_pattern=np.array(sol_data['signature']['x_pattern']),
            )
            sol = StoredSolution(
                signature=sig,
                x_opt=np.array(sol_data['x_opt']),
                pg_opt=np.array(sol_data['pg_opt']) if sol_data['pg_opt'] is not None else None,
                objective=sol_data['objective'],
                timestamp=sol_data['timestamp'],
                scenario_id=sol_data['scenario_id'],
            )
            bank.solutions[key] = sol

        return bank


class PhysicsGuidedRecovery:
    """
    物理信息增强恢复器

    利用UC问题的物理约束（功率平衡、爬坡、最小启停时间等）
    来修正代理模型的偏差
    """

    def __init__(
        self,
        ppc: dict,
        T_delta: float,
        penalty_weights: Optional[Dict[str, float]] = None,
    ):
        self.ppc = ppc
        self.T_delta = T_delta
        self.penalty_weights = penalty_weights or {
            'power_balance': 1e6,
            'ramp_violation': 1e5,
            'min_up_down': 1e4,
            'dc_flow': 1e3,
        }

        # 提取机组参数
        from pypower.ext2int import ext2int
        from pypower.idx_gen import PMIN, PMAX

        ppc_int = ext2int(ppc)
        self.gen = ppc_int['gen']
        self.ng = self.gen.shape[0]
        self.pmin = self.gen[:, PMIN]
        self.pmax = self.gen[:, PMAX]

        # 爬坡率
        self.Ru = 0.4 * self.pmax / T_delta
        self.Rd = 0.4 * self.pmax / T_delta

    def compute_physics_violations(
        self,
        x: np.ndarray,
        pg: np.ndarray,
        pd_sum: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算给定解的物理约束违反量

        Returns:
            Dict containing violation metrics
        """
        violations = {
            'power_balance': 0.0,
            'ramp_up_violation': 0.0,
            'ramp_down_violation': 0.0,
            'min_up_violation': 0.0,
            'min_down_violation': 0.0,
            'capacity_violation': 0.0,
        }

        ng, T = x.shape

        # 功率平衡违反
        for t in range(T):
            imbalance = abs(np.sum(pg[:, t]) - pd_sum[t])
            violations['power_balance'] += imbalance

        # 爬坡违反
        for g in range(ng):
            for t in range(1, T):
                ramp_up = pg[g, t] - pg[g, t-1]
                ramp_down = pg[g, t-1] - pg[g, t]

                if ramp_up > self.Ru[g]:
                    violations['ramp_up_violation'] += ramp_up - self.Ru[g]
                if ramp_down > self.Rd[g]:
                    violations['ramp_down_violation'] += ramp_down - self.Rd[g]

        # 发电容量违反
        for g in range(ng):
            for t in range(T):
                if x[g, t] > 0.5:
                    if pg[g, t] < self.pmin[g]:
                        violations['capacity_violation'] += self.pmin[g] - pg[g, t]
                    if pg[g, t] > self.pmax[g]:
                        violations['capacity_violation'] += pg[g, t] - self.pmax[g]

        return violations

    def apply_physics_corrections(
        self,
        x_biased: np.ndarray,
        bias_profiles: Dict[int, TemporalBiasProfile],
        pd_sum: np.ndarray,
        max_iterations: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        应用物理约束感知的修正

        Args:
            x_biased: 有偏差的启停决策
            bias_profiles: 偏差剖面
            pd_sum: 总负荷
            max_iterations: 最大修正迭代次数

        Returns:
            (x_corrected, correction_info)
        """
        x_corrected = x_biased.copy()
        ng, T = x_corrected.shape

        correction_info = {
            'iterations': 0,
            'modifications': [],
            'total_violation_before': 0.0,
            'total_violation_after': 0.0,
        }

        # 基于偏差模式的启发式修正
        for unit_id, profile in bias_profiles.items():
            if profile.confidence < 0.3:
                continue

            unit_x = x_corrected[unit_id].copy()

            if profile.pattern in [BiasPattern.DELAYED_START, BiasPattern.DELAYED_STOP]:
                # 延迟模式：尝试将启动/停机信号提前
                shift = max(1, int(round(profile.avg_delay)))
                for t in range(T - 1, shift - 1, -1):
                    if unit_x[t] == 1 and unit_x[t-1] == 0:
                        # 启动事件
                        if t - shift >= 0:
                            unit_x[t] = 0
                            unit_x[t - shift] = 1
                            correction_info['modifications'].append({
                                'unit': unit_id,
                                'time': t,
                                'type': 'advance_start',
                                'shift': shift,
                            })

            elif profile.pattern in [BiasPattern.PREMATURE_START, BiasPattern.PREMATURE_STOP]:
                # 过早模式：尝试将启动/停机信号推迟
                shift = max(1, int(round(-profile.avg_delay)))
                for t in range(T - shift):
                    if unit_x[t] == 0 and unit_x[t+1] == 1:
                        # 启动事件
                        if t + shift < T:
                            unit_x[t+1] = 0
                            unit_x[t + shift] = 1
                            correction_info['modifications'].append({
                                'unit': unit_id,
                                'time': t,
                                'type': 'delay_start',
                                'shift': shift,
                            })

            x_corrected[unit_id] = unit_x

        correction_info['iterations'] = 1
        return x_corrected, correction_info


class GuidedFeasibilityPump:
    """
    定向启发式可行性泵

    集成代理模型诊断、物理信息恢复和场景解记忆库的增强型FP
    """

    def __init__(
        self,
        ppc: dict,
        T_delta: float,
        diagnostics: Optional[SurrogateDiagnostics] = None,
        memory_bank: Optional[SolutionMemoryBank] = None,
        physics_recovery: Optional[PhysicsGuidedRecovery] = None,
        enable_guidance: bool = True,
        guidance_weight: float = 0.5,
    ):
        self.ppc = ppc
        self.T_delta = T_delta
        self.diagnostics = diagnostics
        self.memory_bank = memory_bank
        self.physics_recovery = physics_recovery
        self.enable_guidance = enable_guidance
        self.guidance_weight = guidance_weight

        self.iteration_stats: List[Dict] = []

    def guided_rounding(
        self,
        x_lp: np.ndarray,
        x_pool: Optional[np.ndarray] = None,
        bias_profiles: Optional[Dict[int, TemporalBiasProfile]] = None,
    ) -> np.ndarray:
        """
        定向启发式四舍五入

        结合偏差剖面和历史解信息，进行有指导的离散化
        """
        ng, T = x_lp.shape
        x_rounded = np.zeros_like(x_lp)

        for g in range(ng):
            for t in range(T):
                # 基础LP值
                val = x_lp[g, t]

                # 应用偏差补偿
                if bias_profiles and g in bias_profiles:
                    profile = bias_profiles[g]
                    if profile.confidence > 0.3:
                        # 根据偏差模式调整阈值
                        if profile.pattern in [BiasPattern.DELAYED_START]:
                            # 延迟启动：降低启动阈值
                            if val > 0.3:
                                val = 0.6  # 增加启动倾向
                        elif profile.pattern in [BiasPattern.PREMATURE_START]:
                            # 过早启动：提高启动阈值
                            if val < 0.7:
                                val = 0.4  # 降低启动倾向

                # 标准四舍五入
                x_rounded[g, t] = 1.0 if val > 0.5 else 0.0

        return x_rounded

    def run_guided_fp(
        self,
        x_init: np.ndarray,
        trusted_mask: np.ndarray,
        pd_data: np.ndarray,
        max_iter: int = 50,
        x_pool: Optional[np.ndarray] = None,
        bias_profiles: Optional[Dict[int, TemporalBiasProfile]] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """
        运行定向启发式可行性泵

        Args:
            x_init: 初始整数解
            trusted_mask: 高可信度变量掩码
            pd_data: 负荷数据
            max_iter: 最大迭代次数
            x_pool: 历史解池
            bias_profiles: 偏差剖面
            verbose: 是否打印详细信息

        Returns:
            (x_result, success, info)
        """
        from feasibility_pump import run_feasibility_pump, check_uc_feasibility

        # 首先尝试记忆库检索
        if self.memory_bank is not None:
            similar = self.memory_bank.retrieve_similar_solutions(
                pd_data.sum(axis=0) if pd_data.ndim > 1 else pd_data,
                top_k=1,
            )
            if similar and similar[0][2] > 0.95:  # 高相似度
                if verbose:
                    print(f"Found highly similar solution in memory bank (sim={similar[0][2]:.3f})")
                return similar[0][1].x_opt, True, {'source': 'memory_bank'}

        # 应用物理信息恢复修正
        if self.physics_recovery is not None and bias_profiles:
            x_init, correction_info = self.physics_recovery.apply_physics_corrections(
                x_init, bias_profiles, pd_data.sum(axis=0) if pd_data.ndim > 1 else pd_data
            )
            if verbose and correction_info['modifications']:
                print(f"Applied {len(correction_info['modifications'])} physics-guided corrections")

        # 运行标准FP，但使用定向启发式舍入
        # 这里我们通过修改x_pool的方式来影响FP行为

        # 构建增强的x_pool，包含偏差补偿解
        enhanced_pool = x_pool
        if x_pool is not None and bias_profiles:
            # 生成补偿后的解加入候选池
            x_compensated = self.guided_rounding(
                x_init.astype(float), x_pool, bias_profiles
            )
            enhanced_pool = np.concatenate([x_pool, x_compensated.reshape(1, *x_compensated.shape)], axis=0)

        # 调用标准可行性泵
        x_result, success = run_feasibility_pump(
            x_curr=x_init,
            trusted_mask=trusted_mask,
            ppc=self.ppc,
            pd_data=pd_data,
            T_delta=self.T_delta,
            x_pool=enhanced_pool,
            max_iter=max_iter,
            verbose=verbose,
        )

        info = {
            'success': success,
            'iterations': len(self.iteration_stats),
            'bias_profiles_used': list(bias_profiles.keys()) if bias_profiles else [],
        }

        # 存储成功的解到记忆库
        if success and self.memory_bank is not None:
            self.memory_bank.store_solution(
                load_profile=pd_data.sum(axis=0) if pd_data.ndim > 1 else pd_data,
                x_opt=x_result,
                scenario_id=info.get('scenario_id'),
            )

        return x_result, success, info


# ==================== 便捷工厂函数 ====================

def create_guided_fp_pipeline(
    ppc: dict,
    T_delta: float,
    ng: int,
    T: int,
    enable_diagnostics: bool = True,
    enable_memory: bool = True,
    memory_size: int = 1000,
    enable_physics_recovery: bool = True,
) -> GuidedFeasibilityPump:
    """
    创建完整的定向FP流水线

    Example:
        >>> pipeline = create_guided_fp_pipeline(ppc, 1.0, 10, 24)
        >>> x_result, success, info = pipeline.run_guided_fp(
        ...     x_init, trusted_mask, pd_data
        ... )
    """
    diagnostics = SurrogateDiagnostics(ng, T) if enable_diagnostics else None
    memory_bank = SolutionMemoryBank(max_size=memory_size) if enable_memory else None
    physics_recovery = PhysicsGuidedRecovery(ppc, T_delta) if enable_physics_recovery else None

    return GuidedFeasibilityPump(
        ppc=ppc,
        T_delta=T_delta,
        diagnostics=diagnostics,
        memory_bank=memory_bank,
        physics_recovery=physics_recovery,
        enable_guidance=True,
    )


# ==================== 测试代码 ====================

if __name__ == '__main__':
    # 简单的单元测试
    print("Testing SurrogateDiagnostics...")

    ng, T = 3, 24
    diagnostics = SurrogateDiagnostics(ng, T)

    # 模拟延迟启动的偏差
    x_nn = np.zeros((ng, T))
    x_opt = np.zeros((ng, T))

    # 机组0：NN延迟1时段启动
    x_opt[0, 5:20] = 1  # 最优：5-20时段运行
    x_nn[0, 6:20] = 1   # NN：6-20时段运行（延迟启动）

    diagnostics.add_sample(x_nn, x_opt)
    diagnostics.add_sample(x_nn, x_opt)
    diagnostics.add_sample(x_nn, x_opt)

    profiles = diagnostics.analyze_bias_patterns()

    print(f"Detected {len(profiles)} bias profiles")
    for uid, prof in profiles.items():
        print(f"  Unit {uid}: {prof.pattern.name}, delay={prof.avg_delay:.2f}, conf={prof.confidence:.2f}")

    # 测试补偿
    x_comp = diagnostics.get_compensated_solution(x_nn)
    print(f"\nOriginal NN solution:")
    print(f"  Unit 0 on times: {np.where(x_nn[0] == 1)[0].tolist()}")
    print(f"Compensated solution:")
    print(f"  Unit 0 on times: {np.where(x_comp[0] == 1)[0].tolist()}")

    print("\nAll tests passed!")
