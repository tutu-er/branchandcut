# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the main UC case39 test (from project root):
```bash
python tests/run_uc_case39.py
```

Run pytest tests:
```bash
pytest tests/
```

Run a single source module directly (most `src/` files have `if __name__ == '__main__'` sections):
```bash
python src/uc_NN.py
python src/uc_dfsm_bcd.py
```

## Architecture Overview

This is a **power systems Unit Commitment (UC) research project** investigating learning-based methods to accelerate solving UC mixed-integer programs (MIPs).

### Problem Domain
UC is a MIP that schedules generator on/off decisions (`x`, binary) and dispatch (`pg`, continuous) over a time horizon `T`, subject to power balance, ramp rate, minimum up/down time, and transmission constraints. The test system is the IEEE 39-bus New England system with 10 generators.

### Data Layer
- `src/case39_pypower.py`: Returns the 39-bus PyPower dict (`ppc`) via `get_case39_pypower()`
- `src/case39_uc_data.py`: UC-specific generator cost/operational data via `get_case39_uc()`
- `src/load.csv`: Load matrix of shape `(nb, T)` — nb buses, T time periods

### Solver Wrappers (baseline MIP solvers)
Each file wraps the same UC formulation for a different solver backend:
- `src/uc_gurobipy.py` — `UnitCommitmentModel` (Gurobi)
- `src/uc_cvxpy.py` — `UnitCommitmentModelCVXPY` (CVXPY + Gurobi backend)
- `src/uc_cplex.py` — `UnitCommitmentModelCplex` (CPLEX)
- `src/uc_scip.py` — SCIP backend

Economic Dispatch (continuous relaxation, LP) variants:
- `src/ed_gurobipy.py` — `EconomicDispatchGurobi`
- `src/ed_cvxpy.py` — `EconomicDispatchCVXPY`
- `src/ed_cplex.py` — CPLEX variant

### Learning-Based Methods

**Active Set Learning** (`src/ActiveSetLearner.py`):
`ActiveSetLearner` samples perturbed load scenarios, solves UC, and records which constraints are active at optimality. Uses a statistical sample size formula (beta distribution) to bound the probability of unseen active sets.

**DFSM + BCD** (`src/uc_dfsm_bcd.py`, `src/uc_dfsm_training.py`, `src/uc_dfsm_admm.py`):
Dual Feasibility Surrogate Method. Trains surrogate constraints from dual variables. Uses Block Coordinate Descent (BCD) or ADMM to alternately update binary and continuous variables.

**NN Hybrid Method** (`src/uc_NN.py`, `src/uc_NN_BCD.py`):
Main research contribution. Combines BCD iteration for binary variables `x` and dual variables with a PyTorch neural network that updates surrogate constraint parameters (`theta`, `zeta`). `uc_NN.py` is the refactored consolidation of `uc_NN_BCD.py`.

**Surrogate Constraints** (`src/uc_NN_subproblem.py`, `src/uc_NN_subproblem_v3.py`):
`SubproblemSurrogateTrainer` (v3) trains per-generator surrogate constraints of the form:
```
alpha[k]*x[t_k] + beta[k]*x[t_k+1] + gamma[k]*x[t_k+2] <= delta[k]
```
This 3-period form is the current canonical interface (V3). Surrogate constraint coefficients `(alphas, betas, gammas, deltas)` are passed to both subproblem solvers and `feasibility_pump.py`.

**Feasibility Pump** (`src/feasibility_pump.py`):
Heuristic to recover integer-feasible solutions from LP relaxations. Pipeline: solve global LP relaxation → collect integer solutions from per-generator subproblems → fix high-confidence variables → LP projection + rounding loop.

### Active Set Data Format
Solved UC samples are serialized to JSON files containing `all_samples`. Multiple `ActiveSetReader` classes exist (duplicated across `uc_dfsm_bcd.py`, `uc_NN_subproblem_v3.py`, `uc_NN.py`, `uc_NN_BCD.py`) — prefer the one in `uc_NN_subproblem_v3.py` for new code.

### Module Import Convention
All `src/` scripts resolve the project root and add it to `sys.path`, so cross-module imports use the `src.` prefix:
```python
from src.case39_pypower import get_case39_pypower
from src.ed_gurobipy import EconomicDispatchGurobi
```

### Dependencies
Required: `numpy`, `gurobipy`, `pypower`, `pandas`, `matplotlib`, `seaborn`, `scipy`
Optional: `torch` (NN features), `cvxpy` (CVXPY solvers), `cplex` (CPLEX solver)
All optional imports are guarded with try/except and `*_AVAILABLE` flags.
