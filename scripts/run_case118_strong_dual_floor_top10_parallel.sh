#!/usr/bin/env bash
set -euo pipefail

# Run the top-10 high-variability case118 unit submodels in parallel on a server.
#
# Usage:
#   bash scripts/run_case118_strong_dual_floor_top10_parallel.sh
#   bash scripts/run_case118_strong_dual_floor_top10_parallel.sh --max-jobs 4
#   bash scripts/run_case118_strong_dual_floor_top10_parallel.sh --units 2,19,10 --dry-run
#   bash scripts/run_case118_strong_dual_floor_top10_parallel.sh --extra-args "--sub-iter 120 --sign4-delay-rounds 12"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-poweropt}"
MAX_JOBS="${MAX_JOBS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-366}"
UNITS_CSV="2,19,10,17,1,30,0,11,18,12"
ACTIVE_SETS="result/commitment_clustering/pattern_library_case118_K10_20260418_032025_active_set_like_refined_20260418_032025_price_only_clipped.json"
LOG_DIR="result/logs/case118_strong_dual_floor_top10_$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --units)
      UNITS_CSV="$2"
      shift 2
      ;;
    --active-sets)
      ACTIVE_SETS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --extra-args)
      # shellcheck disable=SC2206
      EXTRA_ARGS=($2)
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '1,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a UNITS <<< "$UNITS_CSV"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "repo_root=$REPO_ROOT"
echo "conda_env=$CONDA_ENV"
echo "active_sets=$ACTIVE_SETS"
echo "units=${UNITS[*]}"
echo "max_jobs=$MAX_JOBS"
echo "max_samples=$MAX_SAMPLES"
echo "log_dir=$LOG_DIR"
echo "extra_args=${EXTRA_ARGS[*]:-}"

run_unit() {
  local unit="$1"
  local log_file="$LOG_DIR/unit_${unit}.log"
  local cmd=(
    conda run --no-capture-output -n "$CONDA_ENV" python -u run_training_case118_strong_complex_dual_floor.py
    --target subproblem_bcd
    --solve-preset server
    --active-sets "$ACTIVE_SETS"
    --max-samples "$MAX_SAMPLES"
    --unit-ids "$unit"
    "${EXTRA_ARGS[@]}"
  )

  printf '[unit %s] %q ' "$unit" "${cmd[@]}" | tee "$log_file"
  printf '\n' | tee -a "$log_file"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}" >> "$log_file" 2>&1
  fi
}

failures=0
for unit in "${UNITS[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
  done
  run_unit "$unit" &
done

while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
done

echo "done; failures=$failures; logs=$LOG_DIR"
exit "$failures"
