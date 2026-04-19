"""Temporary config verification script – main_bcd."""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import run_training_case118 as case118_cfg
case118_cfg.TRAIN_TARGET = "main_bcd"
import run_training as rt

case118_cfg._configure_common()
case118_cfg._configure_main_bcd()

checks = {
    "MODE":                  rt.MODE,
    "BCD_LP_BACKEND":        rt.BCD_LP_BACKEND,
    "N_WORKERS_BCD":         rt.N_WORKERS_BCD,
    "BCD_GUROBI_THREADS":    rt.BCD_GUROBI_THREADS,
    "MAX_ITER":              rt.MAX_ITER,
    "BCD_MAX_ITER":          rt.BCD_MAX_ITER,
    "NN_EPOCHS":             rt.NN_EPOCHS,
    "DUAL_DECAY_ROUND":      rt.DUAL_DECAY_ROUND,
    "BCD_DUAL_SIGN_RELAX_INTERVAL": rt.BCD_DUAL_SIGN_RELAX_INTERVAL,
    "BCD_MAX_THETA":         rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT,
    "BCD_THETA_STAGES":      rt.BCD_THETA_TRAINING_STAGES,
    "THETA_HOT_START":       rt.THETA_HOT_START_STRATEGY,
    "ZETA_HOT_START":        rt.ZETA_HOT_START_STRATEGY,
    "LAMBDA_INIT":           rt.BCD_LAMBDA_INIT_STRATEGY,
    "NN_SIZE":               rt.BCD_NN_SIZE,
    "NN_BATCH":              rt.BCD_NN_BATCH_STRATEGY,
    "NN_LR":                 rt.BCD_NN_LR,
    "GAMMA_BASE":            rt.BCD_GAMMA_BASE,
    "RHO_PRIMAL":            rt.BCD_RHO_PRIMAL_INIT,
    "RHO_DUAL":              rt.BCD_RHO_DUAL_INIT,
    "RHO_DUAL_PG":           rt.BCD_RHO_DUAL_PG_INIT,
    "RHO_DUAL_X":            rt.BCD_RHO_DUAL_X_INIT,
    "RHO_DUAL_COC":          rt.BCD_RHO_DUAL_COC_INIT,
    "RHO_BINARY":            rt.BCD_RHO_BINARY_INIT,
    "RHO_OPT":               rt.BCD_RHO_OPT_INIT,
    "PG_PROX":               rt.BCD_PG_BLOCK_PROX_WEIGHT,
    "DUAL_PROX":             rt.BCD_DUAL_BLOCK_PROX_WEIGHT,
    "CASE_NAME":             rt.CASE_NAME,
    "RUN_FP":                rt.RUN_FP,
    "ENABLE_SPARSE":         rt.ENABLE_SPARSE_SUPPORTS,
    "ACTIVE_SETS_FILE":      rt.ACTIVE_SETS_FILE,
    "BCD_CONTINUE":          rt.BCD_CONTINUE_TRAINING,
    "SURROGATE_CONTINUE":    rt.SURROGATE_CONTINUE_TRAINING,
    "T_DELTA":               rt.T_DELTA,
    "MAX_SAMPLES":           rt.MAX_SAMPLES,
}
print("=== main_bcd configuration ===")
for k, v in checks.items():
    print(f"  {k:40s} = {v}")

# Theta schedule validation
try:
    case118_cfg._validate_main_bcd_theta_schedule(
        max_iter=rt.BCD_MAX_ITER,
        max_constraints_per_time_slot=rt.BCD_MAX_THETA_CONSTRAINTS_PER_TIME_SLOT,
        theta_training_stages=rt.BCD_THETA_TRAINING_STAGES,
    )
    print("\n[OK] theta schedule validation passed")
except Exception as e:
    print(f"\n[FAIL] theta schedule: {e}")

# Active-set file check
p = pathlib.Path(rt.ACTIVE_SETS_FILE)
if not p.is_absolute():
    p = pathlib.Path(__file__).parent / rt.ACTIVE_SETS_FILE
status = "OK" if p.exists() else "FAIL"
print(f"[{status}] Active set JSON: {p}")

# BCD_DUAL_SIGN_RELAX_INTERVAL vs BCD_DUAL_SIGN_RELAX_INTERVAL attribute name
try:
    _ = rt.BCD_DUAL_SIGN_RELAX_INTERVAL
    print("[OK] BCD_DUAL_SIGN_RELAX_INTERVAL attribute exists")
except AttributeError:
    print("[FAIL] BCD_DUAL_SIGN_RELAX_INTERVAL missing from run_training")

print("\nDone.")
