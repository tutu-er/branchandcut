"""Resume case3lite main BCD training with theta enabled immediately.

This script is intended for server-side continuation from an existing
``bcd_model_case3lite_*.pth`` checkpoint.  It configures ``run_training``
directly instead of calling the case3lite preset wrapper, so it avoids
interactive shell/stdout issues and makes the resume overrides explicit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _repair_standard_streams() -> None:
    """Ensure print(..., flush=True) works even when launched from a bad shell."""

    if sys.stdout is None or getattr(sys.stdout, "closed", False):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None or getattr(sys.stderr, "closed", False):
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


_repair_standard_streams()

import run_training as rt  # noqa: E402
from src.uc_NN_BCD import Agent_NN_BCD  # noqa: E402

try:  # noqa: E402
    from src.uc_NN_BCD_parallel import ParallelAgent_NN_BCD
except Exception:  # pragma: no cover - optional import on some servers
    ParallelAgent_NN_BCD = None


CHECKPOINT = "result/bcd_models/bcd_model_case3lite_20260508_223657.pth"


def _force_theta_on_after_load(agent: object) -> None:
    """Override checkpoint-restored theta delay/curriculum settings."""

    agent.theta_constraint_delay_rounds = 0
    agent.theta_curriculum_delay_rounds = 0
    agent.theta_initial_scale = 0.65
    agent.theta_final_scale = 2.0
    agent.theta_curriculum_rounds = 20
    print(
        "[resume] forced theta on after checkpoint load: "
        "delay=0, scale=0.65->2.0 over 20 rounds",
        flush=True,
    )


_orig_agent_load = Agent_NN_BCD.load_model_parameters


def _agent_load_then_force_theta_on(self, *args, **kwargs):
    _orig_agent_load(self, *args, **kwargs)
    _force_theta_on_after_load(self)


Agent_NN_BCD.load_model_parameters = _agent_load_then_force_theta_on

if ParallelAgent_NN_BCD is not None:
    _orig_parallel_load = ParallelAgent_NN_BCD.load_model_parameters

    def _parallel_load_then_force_theta_on(self, *args, **kwargs):
        _orig_parallel_load(self, *args, **kwargs)
        _force_theta_on_after_load(self)

    ParallelAgent_NN_BCD.load_model_parameters = _parallel_load_then_force_theta_on


def configure() -> None:
    """Configure only main BCD resume training for case3lite."""

    rt.MODE = "bcd"
    rt.CASE_NAME = "case3lite"
    rt.RUN_FP = False
    rt.ACTIVE_SETS_FILE = (
        "result/active_set/active_sets_case3lite_T24_n1000_20260403_180137.json"
    )
    rt.MAX_SAMPLES = 100
    rt.UNIT_IDS = None

    rt.BCD_MODEL_FILE = CHECKPOINT
    rt.BCD_CONTINUE_TRAINING = True
    rt.BCD_RESTORE_RHO_FROM_CHECKPOINT = False

    rt.MAX_ITER = 40
    rt.BCD_MAX_ITER = 40
    rt.SUBPROBLEM_MAX_ITER = 60
    rt.NN_EPOCHS = 4

    rt.N_WORKERS_BCD = 4
    rt.BCD_LP_BACKEND = "cvxpy_highs"
    rt.BCD_HIGHS_THREADS = 1
    rt.BCD_GUROBI_THREADS = None

    # Continue from the checkpoint but rebalance dual-block priorities:
    # keep x-stationarity from being sacrificed for complementarity.
    rt.BCD_RHO_DUAL_X_INIT = 2e-2
    rt.BCD_RHO_OPT_INIT = 5e-4
    rt.BCD_LOSS_RATIO_DUAL_X = 5.0
    rt.BCD_DUAL_BLOCK_PROX_WEIGHT = 1e-2
    rt.BCD_PG_BLOCK_PROX_WEIGHT = 0.0

    # Theta is enabled immediately after checkpoint load.  These values are
    # also set on rt so the newly constructed agent receives them before load.
    rt.BCD_THETA_CONSTRAINT_DELAY_ROUNDS = 0
    rt.BCD_THETA_CURRICULUM_DELAY_ROUNDS = 0
    rt.BCD_THETA_CURRICULUM_ROUNDS = 20
    rt.BCD_THETA_INITIAL_SCALE = 0.65
    rt.BCD_THETA_FINAL_SCALE = 2.0
    rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT = 20
    rt.BCD_THETA_TRAINING_STAGES = [
        {"until_round": 10, "max_constraints_per_time_slot": 6},
        {"until_round": 20, "max_constraints_per_time_slot": 12},
        {"until_round": 40, "max_constraints_per_time_slot": 20},
    ]

    # Keep theta duals active but less aggressive than the failed run.
    rt.BCD_MU_DUAL_FLOOR_INIT = 0.5
    rt.BCD_ITA_DUAL_FLOOR_INIT = 0.0
    rt.DUAL_DECAY_ROUND = 10
    rt.BCD_DUAL_SIGN_RELAX_INTERVAL = 4

    # Reduce zeta/ita duals through a soft cap from the beginning.
    rt.BCD_ZETA_ITA_CAP_PENALTY_WEIGHT = 2.0
    rt.BCD_ZETA_ITA_CAP_INITIAL_WEIGHT = 0.5
    rt.BCD_ZETA_ITA_CAP_FINAL_WEIGHT = 2.0
    rt.BCD_ZETA_ITA_CAP_INITIAL = 2.0
    rt.BCD_ZETA_ITA_CAP_FINAL = 0.0
    rt.BCD_ZETA_ITA_CAP_START_ROUND = 0
    rt.BCD_ZETA_ITA_CAP_END_ROUND = 16

    # Avoid loading a mismatched standalone unit predictor during BCD resume.
    rt.USE_UNIT_PREDICTOR = False
    rt.BCD_USE_UNIT_PREDICTOR = False
    rt.UNIT_PREDICTOR_LOAD_PATH = None
    rt.UNIT_PREDICTOR_AUTO_LATEST_STANDALONE = False

    # This script is for main BCD only.
    rt.SURROGATE_MODEL_DIR = None
    rt.SURROGATE_CONTINUE_TRAINING = False
    rt.SURROGATE_DUAL_PREDICTOR_ONLY = False


def main() -> None:
    configure()
    print("=" * 72, flush=True)
    print(
        "case3lite BCD resume | "
        f"checkpoint={rt.BCD_MODEL_FILE} | max_iter={rt.BCD_MAX_ITER} | "
        f"backend={rt.BCD_LP_BACKEND} | workers={rt.N_WORKERS_BCD}",
        flush=True,
    )
    print(
        "theta immediate | floor=0.5 | scale=0.65->2.0 | "
        "zeta cap=2.0->0.0",
        flush=True,
    )
    print("=" * 72, flush=True)
    rt.main()


if __name__ == "__main__":
    main()
