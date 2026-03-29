import numpy as np
import os
import sys
import types

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
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL = 'all'
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL_SINGLE_TIME = 'all_single_time'
dummy_subproblem_mod.CONSTRAINT_STRATEGY_ALL_TEMPLATES_SIGN4 = 'all_templates_sign4'
dummy_subproblem_mod.SURROGATE_SINGLE_TIME_OFFSETS = (0,)
dummy_subproblem_mod.SURROGATE_TRIPLE_WINDOW_OFFSETS = (0, 1, 2)
dummy_subproblem_mod.normalize_constraint_generation_strategy = (
    lambda strategy: (
        'all_templates_sign4'
        if str(strategy).strip().lower() == 'all_templates_rhs3'
        else str(strategy).strip().lower()
    )
)
def _resolve_constraint_offsets_from_trainer(trainer, sample_id, n_constraints):
    offsets = getattr(trainer, 'surrogate_constraint_offsets', None)
    if (
        sample_id is not None
        and isinstance(offsets, list)
        and 0 <= sample_id < len(offsets)
    ):
        return list(offsets[sample_id])[:n_constraints]
    default = (0,) if getattr(trainer, 'constraint_generation_strategy', '') == 'all_single_time' else (0, 1, 2)
    return [default] * n_constraints
dummy_subproblem_mod.resolve_constraint_offsets_from_trainer = _resolve_constraint_offsets_from_trainer
def _build_surrogate_constraint_expression(x_values, timestep, offsets, alpha_value, beta_value, gamma_value, horizon):
    expr = 0.0
    if 0 in offsets and 0 <= timestep < horizon:
        expr += alpha_value * x_values[timestep]
    if 1 in offsets and 0 <= timestep + 1 < horizon:
        expr += beta_value * x_values[timestep + 1]
    if 2 in offsets and 0 <= timestep + 2 < horizon:
        expr += gamma_value * x_values[timestep + 2]
    return expr
dummy_subproblem_mod.build_surrogate_constraint_expression = _build_surrogate_constraint_expression
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

from src import feasibility_pump as fp


class _DummyTrainer:
    def __init__(
        self,
        unit_id,
        active_set_data,
        horizon=3,
        constraint_generation_strategy='all',
        sensitive_timesteps=None,
        all_mode_group_size=4,
    ):
        self.unit_id = unit_id
        self.active_set_data = active_set_data
        self.T = horizon
        self.generator_injection_sensitivity = np.array([[0.2, 0.6]], dtype=float)
        self.constraint_generation_strategy = constraint_generation_strategy
        self.sensitive_timesteps = (
            sensitive_timesteps
            if sensitive_timesteps is not None
            else [list(range(max(horizon - 2, 0))) for _ in active_set_data]
        )
        default_offsets = [(0,)] * horizon if constraint_generation_strategy == 'all_single_time' else None
        self.surrogate_constraint_offsets = (
            [default_offsets.copy() for _ in active_set_data]
            if default_offsets is not None
            else [[(0, 1, 2)] * len(timesteps) for timesteps in self.sensitive_timesteps]
        )
        self.all_mode_group_size = all_mode_group_size

    def get_surrogate_params(self, pd_data, lambda_val, renewable_data=None):
        sample = fp._coerce_scenario_sample(pd_data)
        load_sum = float(np.sum(fp.get_sample_load_data(sample)))
        renewable_sum = float(np.sum(fp.get_sample_renewable_data(sample)))
        lambda_sum = float(np.sum(lambda_val))
        base = 0.01 * load_sum + 0.02 * renewable_sum + 0.03 * lambda_sum + self.unit_id
        alphas = np.array([base, base + 0.1], dtype=float)
        betas = np.array([base + 0.2, base + 0.3], dtype=float)
        gammas = np.array([base + 0.4, base + 0.5], dtype=float)
        deltas = np.array([base + 0.6, base + 0.7], dtype=float)
        costs = np.zeros(self.T, dtype=float)
        pg_costs = np.zeros(self.T, dtype=float)
        return alphas, betas, gammas, deltas, costs, pg_costs


class _DummyLambdaPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, scenario_input):
        sample = fp._coerce_scenario_sample(scenario_input)
        load_sum = float(np.sum(fp.get_sample_load_data(sample)))
        self.calls.append(load_sum)
        lambda_pb = np.full(3, load_sum / 100.0, dtype=float)
        lambda_du = np.full((1, 3), load_sum / 500.0, dtype=float)
        lambda_dl = np.full((1, 3), load_sum / 1000.0, dtype=float)
        return {
            'lambda_power_balance': lambda_pb,
            'lambda_dcpf_upper': lambda_du,
            'lambda_dcpf_lower': lambda_dl,
        }


def test_extract_unit_lambda_projects_global_duals_with_dcpf():
    trainer = _DummyTrainer(1, active_set_data=[], horizon=3)
    lambda_payload = {
        'lambda_power_balance': np.array([1.0, 1.5, 2.0], dtype=float),
        'lambda_dcpf_upper': np.array([[0.3, 0.4, 0.5]], dtype=float),
        'lambda_dcpf_lower': np.array([[0.1, 0.1, 0.2]], dtype=float),
    }

    lambda_unit = fp._extract_unit_lambda(
        lambda_payload,
        T=3,
        unit_id=1,
        trainer=trainer,
    )

    expected = np.array([0.88, 1.32, 1.82], dtype=float)
    np.testing.assert_allclose(lambda_unit, expected)


def test_resolve_surrogate_constraint_timesteps_all():
    trainer = _DummyTrainer(0, active_set_data=[{}], horizon=5, constraint_generation_strategy='all')
    resolved = fp._resolve_surrogate_constraint_timesteps(trainer, {'sample_id': 0}, T=5, n_constraints=3)
    assert resolved == [0, 1, 2]


def test_resolve_surrogate_constraint_timesteps_all_templates_sign4():
    trainer = _DummyTrainer(
        0,
        active_set_data=[{}],
        horizon=5,
        constraint_generation_strategy='all_templates_sign4',
        all_mode_group_size=4,
    )
    resolved = fp._resolve_surrogate_constraint_timesteps(trainer, {'sample_id': 0}, T=5, n_constraints=12)
    assert resolved == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_resolve_surrogate_constraint_timesteps_all_single_time():
    trainer = _DummyTrainer(
        0,
        active_set_data=[{}],
        horizon=5,
        constraint_generation_strategy='all_single_time',
        sensitive_timesteps=[list(range(5))],
    )
    resolved = fp._resolve_surrogate_constraint_timesteps(trainer, {'sample_id': 0}, T=5, n_constraints=5)
    assert resolved == [0, 1, 2, 3, 4]


def test_resolve_surrogate_constraint_timesteps_sensitive_uses_sample_id():
    trainer = _DummyTrainer(
        0,
        active_set_data=[{}, {}],
        horizon=6,
        constraint_generation_strategy='sensitive',
        sensitive_timesteps=[[0, 2], [1, 3]],
    )
    resolved = fp._resolve_surrogate_constraint_timesteps(trainer, {'sample_id': 1}, T=6, n_constraints=2)
    assert resolved == [1, 3]


def test_collect_integer_solutions_combines_all_strategies(monkeypatch):
    active_set_data = [
        {
            'load_data': np.full((2, 3), 10.0),
            'renewable_data': np.full((2, 3), 1.0),
        },
        {
            'load_data': np.full((2, 3), 11.0),
            'renewable_data': np.full((2, 3), 1.0),
        },
        {
            'load_data': np.full((2, 3), 30.0),
            'renewable_data': np.full((2, 3), 5.0),
        },
    ]
    trainers = {
        0: _DummyTrainer(0, active_set_data),
        1: _DummyTrainer(1, active_set_data),
    }
    lambda_predictor = _DummyLambdaPredictor()

    def _fake_unit_lp(
        _trainer,
        _lambda_val,
        alphas,
        betas,
        gammas,
        deltas,
        costs=None,
        pg_costs=None,
        scenario_sample=None,
    ):
        score = float(np.sum(alphas + betas + gammas - deltas))
        x_sol = np.array(
            [
                0.9 if score >= 0 else 0.1,
                0.8 if np.mean(deltas) >= np.mean(alphas) else 0.2,
                0.7,
            ],
            dtype=float,
        )
        return x_sol, 2, {'status_name': 'OPTIMAL'}

    monkeypatch.setattr(fp, '_solve_unit_LP_with_surrogate', _fake_unit_lp)

    x_surr_lp, x_init_k, x_init_k_m = fp.collect_integer_solutions(
        pd_data=active_set_data[0],
        lambda_val=np.array([1.0, 1.0, 1.0], dtype=float),
        trainers=trainers,
        n_perturbations=2,
        n_similar_scenarios=1,
        similar_scenario_pool_size=2,
        n_load_perturbations=1,
        load_perturbation_scale=0.02,
        lambda_predictor=lambda_predictor,
        rng=np.random.default_rng(7),
    )

    assert x_surr_lp.shape == (2, 3)
    assert x_init_k.shape == (2, 3)
    assert x_init_k_m.shape == (2, 4, 3)
    assert len(lambda_predictor.calls) == 4
    assert np.all(np.logical_or(x_init_k_m == 0, x_init_k_m == 1))


def test_select_pool_restart_candidate_prefers_free_variable_difference():
    x_reference = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    trusted_mask = np.array([[True, False, True], [False, False, False]])
    x_pool = np.array(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 1]],
            [[1, 0, 1], [0, 0, 0]],
        ],
        dtype=int,
    )

    candidate = fp._select_pool_restart_candidate(
        x_reference=x_reference,
        x_pool=x_pool,
        trusted_mask=trusted_mask,
        rng=np.random.default_rng(3),
    )

    assert candidate is not None
    assert np.array_equal(candidate[trusted_mask], x_reference[trusted_mask])
    assert np.sum(candidate[~trusted_mask] != x_reference[~trusted_mask]) >= 1
