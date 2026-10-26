"""
hac_threshold_search.py
Search for the threshold that best aligns HAC clusters with province membership,
with bootstrapped confidence intervals on the best threshold and its ARI score.

Strategy
--------
1. Load the cached linkage matrices (data/hac_linkage_cache.pkl).
2. Collect every unique merge height across all components as threshold candidates.
   Between two consecutive merge heights nothing changes, so this is the exact
   minimal set — no information is lost by only evaluating at merge heights.
3. Pre-compute a (n_thresholds × n_nodes) integer matrix of cluster labels.
   This costs ~6s and ~130MB once, then all subsequent operations are fast
   array lookups.
4. Find the best threshold on the full dataset (maximises ARI with provinces).
5. Bootstrap over nodes (with replacement) to build a distribution of best
   thresholds and best ARI scores, from which 95% CIs are derived.

Output: printed summary only.

Usage (from repo root):
  python results/HAC/src/hac_threshold_search.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

CACHE_PATH   = Path("results/HAC/data/hac_linkage_cache.pkl")
CLUSTERS_CSV = Path("results/HAC/data/hac_clusters.csv")
N_BOOTSTRAP  = 500
SEED         = 42
CI_LEVEL     = 95  # percent


# ── Load linkage cache ────────────────────────────────────────────────────────
print(f"Loading linkage cache from {CACHE_PATH}...")
with open(CACHE_PATH, "rb") as f:
    comp_results: list[tuple[list, np.ndarray]] = pickle.load(f)

total_nodes = sum(len(nids) for nids, _ in comp_results)
print(f"  {len(comp_results)} components, {total_nodes} nodes total")


# ── Load province labels ──────────────────────────────────────────────────────
print(f"Loading province labels from {CLUSTERS_CSV}...")
df = pd.read_csv(CLUSTERS_CSV)
df = df[df["province"] != "Unknown"].copy()
print(f"  {len(df)} nodes with known province")

# Build aligned arrays: node_id order matches what we'll compute cluster labels for
# Index node_ids present in linkage cache AND having a known province
cache_node_ids = {nid for nids, _ in comp_results for nid in nids}
df = df[df["node_id"].isin(cache_node_ids)].reset_index(drop=True)
province_labels = df["province"].values
node_id_set     = set(df["node_id"].tolist())
node_id_index   = {nid: i for i, nid in enumerate(df["node_id"].tolist())}
n_nodes         = len(df)
print(f"  {n_nodes} nodes used for alignment search")


# ── Collect all unique merge heights as threshold candidates ──────────────────
all_heights = np.unique(np.concatenate([Z[:, 2] for _, Z in comp_results]))
thresholds  = all_heights  # skip t=0 (all singletons, ARI=0 by definition)
n_thresh    = len(thresholds)
print(f"\n{n_thresh} threshold candidates  "
      f"({thresholds.min()/3600:.2f}h → {thresholds.max()/3600:.1f}h)")


# ── Pre-compute cluster label matrix ─────────────────────────────────────────
# Shape: (n_thresh, n_nodes)  dtype: int32
# Rows = thresholds, columns = nodes in the same order as df/province_labels.
# Only nodes with a known province (node_id_set) are included.
print("\nPre-computing cluster assignments for all thresholds...")
label_matrix = np.empty((n_thresh, n_nodes), dtype=np.int32)

# For each component, compute fcluster at every threshold, then scatter into matrix
for nids, Z in comp_results:
    # Only keep the subset of node_ids that appear in our filtered df
    col_indices = [node_id_index[nid] for nid in nids if nid in node_id_index]
    nids_filtered = [nid for nid in nids if nid in node_id_index]
    if not col_indices:
        continue

    # offset per component so labels don't collide across components
    comp_offset = int(label_matrix[:, col_indices].max() + 1) if label_matrix[:, col_indices].any() else 0

    for ti, t in enumerate(thresholds):
        labels = fcluster(Z, t=t, criterion="distance").astype(np.int32)
        # Map component-local node order to filtered global order
        for local_i, (nid, col_i) in enumerate(zip(nids_filtered, col_indices)):
            orig_local = nids.index(nid)
            label_matrix[ti, col_i] = labels[orig_local] + comp_offset

print(f"  Done  — matrix shape: {label_matrix.shape}  "
      f"({label_matrix.nbytes / 1e6:.0f} MB)")


# ── Full-data ARI sweep ───────────────────────────────────────────────────────
print("\nSearching for best threshold on full dataset...")
ari_full = np.array([
    adjusted_rand_score(province_labels, label_matrix[ti])
    for ti in range(n_thresh)
])

best_ti   = int(np.argmax(ari_full))
best_t_h  = thresholds[best_ti] / 3600
best_ari  = ari_full[best_ti]

# Also report top-5 so the user can see if there's a plateau
top5 = np.argsort(ari_full)[-5:][::-1]
print("\nTop 5 thresholds by ARI:")
print(f"  {'Threshold (h)':>16}  {'ARI':>8}")
for ti in top5:
    marker = " ◀ best" if ti == best_ti else ""
    print(f"  {thresholds[ti]/3600:>16.2f}  {ari_full[ti]:>8.4f}{marker}")


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
# Searching all 8,000+ thresholds in every bootstrap iteration is far too slow.
# Instead, narrow to the top-K candidates from the full sweep — the bootstrap
# only needs to determine which of the plausible winners is most stable, not
# re-examine the entire threshold space from scratch each time.
TOP_K = 100
top_k_idx = np.argsort(ari_full)[-TOP_K:]  # indices of top-K thresholds by ARI
top_k_matrix = label_matrix[top_k_idx]     # (TOP_K, n_nodes) — pre-sliced

print(f"\nBootstrapping ({N_BOOTSTRAP} iterations, seed={SEED}) "
      f"over top-{TOP_K} candidate thresholds...")
rng = np.random.default_rng(SEED)

boot_best_t   = np.empty(N_BOOTSTRAP)
boot_best_ari = np.empty(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    idx    = rng.integers(0, n_nodes, size=n_nodes)
    prov_b = province_labels[idx]

    ari_b  = np.array([
        adjusted_rand_score(prov_b, top_k_matrix[ki][idx])
        for ki in range(TOP_K)
    ])
    best_ki = int(np.argmax(ari_b))
    boot_best_t[b]    = thresholds[top_k_idx[best_ki]] / 3600
    boot_best_ari[b]  = ari_b[best_ki]

    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{N_BOOTSTRAP}...", flush=True)

alpha    = (100 - CI_LEVEL) / 2
t_lo, t_hi    = np.percentile(boot_best_t,   [alpha, 100 - alpha])
ari_lo, ari_hi = np.percentile(boot_best_ari, [alpha, 100 - alpha])


# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "═" * 52)
print("  HAC threshold search — province alignment")
print("═" * 52)
print(f"  Best threshold : {best_t_h:.2f}h  "
      f"({best_t_h/24:.2f} days)")
print(f"  {CI_LEVEL}% CI (threshold) : [{t_lo:.2f}h, {t_hi:.2f}h]  "
      f"([{t_lo/24:.2f}d, {t_hi/24:.2f}d])")
print(f"  ARI at best    : {best_ari:.4f}")
print(f"  {CI_LEVEL}% CI (ARI)       : [{ari_lo:.4f}, {ari_hi:.4f}]")
print("═" * 52)
