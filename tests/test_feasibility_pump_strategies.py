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
    def __init__(self, unit_id, active_set_data, horizon=3):
        self.unit_id = unit_id
        self.active_set_data = active_set_data
        self.T = horizon

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
        return alphas, betas, gammas, deltas, costs


class _DummyLambdaPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, scenario_input):
        sample = fp._coerce_scenario_sample(scenario_input)
        load_sum = float(np.sum(fp.get_sample_load_data(sample)))
        self.calls.append(load_sum)
        return np.full(3, load_sum / 100.0, dtype=float)


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

    def _fake_unit_lp(_trainer, _lambda_val, alphas, betas, gammas, deltas):
        score = float(np.sum(alphas + betas + gammas - deltas))
        return np.array(
            [
                0.9 if score >= 0 else 0.1,
                0.8 if np.mean(deltas) >= np.mean(alphas) else 0.2,
                0.7,
            ],
            dtype=float,
        )

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
