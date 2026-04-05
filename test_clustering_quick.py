"""Quick end-to-end test for commitment clustering (case118)."""
import sys
sys.path.insert(0, '.')
from src.numpy_compat import ensure_numpy_compat_for_pypower
ensure_numpy_compat_for_pypower()

from src.commitment_clustering import CommitmentClusterer
from src.mti118_data_loader import load_case118_ppc_with_mti_limits

JSON_PATH = r"result\active_set\active_sets_case118_T0_n366_20260322_063917.json"

print("Loading ppc...", flush=True)
ppc = load_case118_ppc_with_mti_limits(aggregate_thermal_by_bus=True)
ng = ppc["gen"].shape[0]
nb = ppc["bus"].shape[0]
print(f"  Generators: {ng}, Buses: {nb}, Branches: {ppc['branch'].shape[0]}", flush=True)

print("\nCreating clusterer (2 clusters, max 3 scenarios/cluster)...", flush=True)
clusterer = CommitmentClusterer(
    ppc=ppc,
    T_delta=1.0,
    case_name="case118",
    n_clusters=2,
    transition_penalty=1.0,
    lp_proximity_weight=0.0,
    max_cost_increase_ratio=None,
    max_scenarios_per_cluster=3,
    gurobi_time_limit=120.0,
    feature_mode="summary",
    verbose=False,
)

print("\nLoading samples from JSON...", flush=True)
samples = clusterer.load_samples_from_json(JSON_PATH)
print(f"  Total valid samples: {len(samples)}", flush=True)

if not samples:
    print("ERROR: No valid samples loaded!", flush=True)
    sys.exit(1)

s = samples[0]
print(f"  Sample 0: load={s['load_data'].shape}, x={s['unit_commitment_matrix'].shape}", flush=True)
ren = s.get("renewable_data")
if ren is not None:
    print(f"  renewable={ren.shape}", flush=True)

print("\nExtracting features...", flush=True)
features = clusterer.extract_features()
print(f"  Features shape: {features.shape}", flush=True)

print("\nClustering...", flush=True)
labels = clusterer.cluster(features)

print("\n--- Running full pipeline (solve + evaluate) ---", flush=True)
results = clusterer.run(JSON_PATH)

output = clusterer.save_results()
print(f"\nTest complete. Output: {output}", flush=True)
