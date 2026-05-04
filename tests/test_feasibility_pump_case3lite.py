import os
import sys
import types

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

if 'gurobipy' not in sys.modules:
    sys.modules['gurobipy'] = types.SimpleNamespace(
        Model=object,
        GRB=types.SimpleNamespace(OPTIMAL=2, MINIMIZE=1),
        quicksum=sum,
    )

if 'pypower' not in sys.modules:
    sys.modules['pypower'] = types.ModuleType('pypower')
    ext2int_mod = types.ModuleType('pypower.ext2int')
    ext2int_mod.ext2int = lambda ppc: ppc
    makeptdf_mod = types.ModuleType('pypower.makePTDF')
    makeptdf_mod.makePTDF = lambda *args, **kwargs: np.zeros((0, 0))
    idx_gen_mod = types.ModuleType('pypower.idx_gen')
    idx_gen_mod.GEN_BUS = 0
    idx_gen_mod.PMIN = 1
    idx_gen_mod.PMAX = 2
    idx_brch_mod = types.ModuleType('pypower.idx_brch')
    idx_brch_mod.RATE_A = 0
    idx_brch_mod.BR_STATUS = 1
    sys.modules['pypower.ext2int'] = ext2int_mod
    sys.modules['pypower.makePTDF'] = makeptdf_mod
    sys.modules['pypower.idx_gen'] = idx_gen_mod
    sys.modules['pypower.idx_brch'] = idx_brch_mod

dummy_subproblem_mod = types.ModuleType('src.uc_NN_subproblem')
dummy_subproblem_mod.SubproblemSurrogateTrainer = object
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL = "all"
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL_SINGLE_TIME = "all_single_time"
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4 = "all_templates_sign4"
dummy_subproblem_mod.SURROGATE_SINGLE_TIME_OFFSETS = ((0,),)
dummy_subproblem_mod.SURROGATE_TRIPLE_WINDOW_OFFSETS = ((0, 1, 2),)
dummy_subproblem_mod.build_surrogate_constraint_expression = lambda *args, **kwargs: 0.0
dummy_subproblem_mod.normalize_constraint_generation_strategy = lambda strategy: strategy
dummy_subproblem_mod.resolve_constraint_offsets_from_trainer = lambda trainer, strategy=None: ((0, 1, 2),)
sys.modules.setdefault('src.uc_NN_subproblem', dummy_subproblem_mod)
sys.modules.setdefault('uc_NN_subproblem', dummy_subproblem_mod)

dummy_sparse_mining = types.ModuleType('src.sparse_surrogate_mining')
dummy_sparse_mining.SparseSurrogateLibrary = object
dummy_sparse_mining.add_sparse_parameterized_constraints = lambda *args, **kwargs: None
sys.modules.setdefault('src.sparse_surrogate_mining', dummy_sparse_mining)
sys.modules.setdefault('sparse_surrogate_mining', dummy_sparse_mining)

dummy_sparse_templates = types.ModuleType('src.sparse_constraint_templates')
dummy_sparse_templates.SparseConstraintTemplateLibrary = object
dummy_sparse_templates.add_sparse_x_templates_to_model = lambda *args, **kwargs: None
sys.modules.setdefault('src.sparse_constraint_templates', dummy_sparse_templates)
sys.modules.setdefault('sparse_constraint_templates', dummy_sparse_templates)

from src import feasibility_pump_case3lite as fp_case3lite
from src import feasibility_pump_case118 as fp_case118
from src import feasibility_pump as fp


def _build_test_ppc(ng=3, min_up=None, min_down=None):
    gen = np.zeros((ng, 3), dtype=float)
    gen[:, 0] = np.arange(ng, dtype=float)
    if min_up is None:
        min_up = np.full(ng, 2.0, dtype=float)
    if min_down is None:
        min_down = np.full(ng, 2.0, dtype=float)
    return {
        "gen": gen,
        "uc_min_up_time_h": np.asarray(min_up, dtype=float),
        "uc_min_down_time_h": np.asarray(min_down, dtype=float),
    }


def test_build_g0_options_deduplicates_without_unpack_error():
    T = 6
    x_lp = np.array([0.9, 0.8, 0.7, 0.9, 0.8, 0.7], dtype=float)
    x_surr_lp = np.array([0.9, 0.8, 0.7, 0.9, 0.8, 0.7], dtype=float)
    x_init_k = np.array([1, 1, 1, 1, 1, 1], dtype=int)
    x_init_k_m = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    vote_majority = np.array([1, 1, 1, 1, 1, 1], dtype=int)
    trusted_mask = np.array([True, True, False, False, True, True], dtype=bool)
    support_reference = np.array([0.9, 0.9, 0.85, 0.88, 0.9, 0.9], dtype=float)

    options, trusted_override = fp_case3lite._build_g0_options(
        g=0,
        T=T,
        x_lp=x_lp,
        x_surr_lp=x_surr_lp,
        x_init_k=x_init_k,
        x_init_k_m=x_init_k_m,
        vote_majority=vote_majority,
        trusted_mask=trusted_mask,
        support_reference=support_reference,
        nearby_rows=None,
    )

    assert options
    assert isinstance(options[0]["name"], str)
    assert options[0]["row"].shape == (T,)
    assert trusted_override.shape == (T,)


def test_commitment_logic_feasibility_rejects_min_up_down_violation():
    ppc = _build_test_ppc(ng=1, min_up=[3.0], min_down=[2.0])

    is_valid, reason = fp.check_commitment_logic_feasibility(
        np.array([[0, 1, 0, 0]], dtype=int),
        ppc,
        1.0,
    )

    assert not is_valid
    assert "最小开机时间" in reason


def test_sanitize_named_commitment_candidates_outputs_logic_feasible_pool():
    ppc = _build_test_ppc(ng=2, min_up=[3.0, 2.0], min_down=[2.0, 2.0])
    candidate_specs = [
        ("bad", np.array([[0, 1, 0, 0], [1, 1, 0, 0]], dtype=int)),
        ("duplicate_good", np.array([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=int)),
    ]

    sanitized, rejected = fp._sanitize_named_commitment_candidates(candidate_specs, ppc, 1.0)

    assert sanitized
    assert len(sanitized) == 1
    assert rejected == []
    for _name, candidate in sanitized:
        is_valid, _reason = fp.check_commitment_logic_feasibility(candidate, ppc, 1.0)
        assert is_valid


def test_case3lite_x_pool_contains_only_logic_feasible_candidates():
    ppc = _build_test_ppc(ng=3, min_up=[3.0, 2.0, 2.0], min_down=[2.0, 2.0, 2.0])
    x_init_k = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )
    x_init_k_m = np.array(
        [
            [[0, 1, 0, 0], [0, 1, 1, 1]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
        ],
        dtype=int,
    )
    global_combinations = [
        {
            "candidate": np.array(
                [
                    [0, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                ],
                dtype=int,
            )
        }
    ]
    nearby_commitment_candidates = [
        (
            "nearby_1",
            np.array(
                [
                    [0, 1, 1, 1],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1],
                ],
                dtype=int,
            ),
        )
    ]

    x_pool = fp_case3lite._build_case3lite_x_pool(
        ppc,
        1.0,
        x_init_k,
        x_init_k_m,
        global_combinations,
        nearby_commitment_candidates,
    )

    assert x_pool is not None
    assert x_pool.shape[0] >= 1
    for candidate in x_pool:
        is_valid, _reason = fp.check_commitment_logic_feasibility(candidate, ppc, 1.0)
        assert is_valid


def test_case118_historical_candidates_rank_similar_commitments():
    ppc = _build_test_ppc(ng=2, min_up=[1.0, 1.0], min_down=[1.0, 1.0])
    target = {
        "load_data": np.full((2, 3), 10.0, dtype=float),
        "renewable_data": np.zeros((2, 3), dtype=float),
    }
    close_commitment = np.array([[1, 1, 1], [0, 0, 0]], dtype=int)
    far_commitment = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
    scenario_bank = [
        {
            "load_data": np.full((2, 3), 10.5, dtype=float),
            "renewable_data": np.zeros((2, 3), dtype=float),
            "unit_commitment_matrix": close_commitment,
        },
        {
            "load_data": np.full((2, 3), 40.0, dtype=float),
            "renewable_data": np.zeros((2, 3), dtype=float),
            "unit_commitment_matrix": far_commitment,
        },
    ]
    x_lp = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], dtype=float)
    x_surr_lp = x_lp.copy()
    x_init_k = close_commitment.copy()
    x_init_k_m = close_commitment[:, None, :].copy()

    candidates, records = fp_case118.build_case118_historical_candidates(
        target,
        trainers={},
        scenario_bank=scenario_bank,
        x_lp=x_lp,
        x_surr_lp=x_surr_lp,
        x_init_k=x_init_k,
        x_init_k_m=x_init_k_m,
        ppc=ppc,
        T_delta=1.0,
        max_candidates=2,
        candidate_pool_size=2,
    )

    assert candidates
    assert candidates[0][0] == "case118_history_1"
    np.testing.assert_array_equal(candidates[0][1], close_commitment)
    assert records[0]["subproblem_distance"] == 0.0


def test_case118_heuristic_candidates_repair_capacity_shortfall():
    ppc = _build_test_ppc(ng=2, min_up=[1.0, 1.0], min_down=[1.0, 1.0])
    ppc["gen"][:, 2] = np.array([40.0, 80.0], dtype=float)
    ppc["gencost"] = np.zeros((2, 7), dtype=float)
    ppc["gencost"][:, -2] = np.array([1.0, 2.0], dtype=float)

    pd_data = np.full((2, 3), 35.0, dtype=float)
    x_lp = np.zeros((2, 3), dtype=float)
    x_surr_lp = np.zeros((2, 3), dtype=float)
    x_init_k = np.zeros((2, 3), dtype=int)
    x_init_k_m = np.zeros((2, 1, 3), dtype=int)

    candidates = fp_case118.build_case118_heuristic_candidates(
        x_lp,
        x_surr_lp,
        x_init_k,
        x_init_k_m,
        ppc,
        pd_data,
        T_delta=1.0,
        reserve_margin=0.0,
    )

    capacity_candidates = [candidate for name, candidate in candidates if name == "case118_capacity_support"]
    assert capacity_candidates
    online_capacity = ppc["gen"][:, 2] @ capacity_candidates[0]
    assert np.all(online_capacity >= np.sum(pd_data, axis=0))
