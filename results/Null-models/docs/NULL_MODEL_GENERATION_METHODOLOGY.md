# Null Model Generation Methodology

This document provides a detailed technical description of how each spatial null model was generated for the Roman road network analysis.

## Overview

Four null models were generated representing two spatial models × two scopes:

1. **Global KNN+MST** - Whole-network K-nearest neighbors with minimum spanning tree backbone
2. **Global Delaunay+MST** - Whole-network Delaunay triangulation with MST backbone and distance constraints
3. **Province-Bridged KNN+MST** - Per-province KNN models merged with empirical cross-province MST bridges
4. **Province-Bridged Delaunay+MST** - Per-province Delaunay models merged with empirical cross-province MST bridges

All models share the same node set as the empirical Roman road network (8,460 nodes) and preserve or extend network connectivity through spatial mechanisms.

---

## Spatial Null Model Construction

### 1. K-Nearest Neighbors (KNN) Model

**Purpose:** Create a spatial null preserving first-order locality constraints by connecting each node to its k nearest geographic neighbors.

**Algorithm:**

For each node $n$ in the network:
1. Calculate Euclidean distances from $n$ to all other nodes using planar coordinates (x, y)
2. Identify the $k$ nearest neighbors by distance (default $k=4$)
3. Add undirected edges $(n, v_i)$ for each of the $k$ nearest neighbors

**Edge Count:**
- Theoretical maximum: $n \times k / 2 \approx 16,920$ undirected edges
- Actual global KNN: **21,611 edges generated** (before MST addition)
- Excess due to averaging asymmetries in neighbor selection (if B is among A's k-nearest but A is not among B's k-nearest, both edges are added)

**Special Cases:**
- Nodes at borders: $k_{\text{eff}} = \min(k, n-1)$ to handle small graphs
- No self-loops: $(n, n)$ edges explicitly excluded

**Advantages:**
- Deterministic, reproducible spatial mechanism
- Respects local geographic structure
- Computationally efficient $O(n^2)$ pairwise distance calculation

---

### 2. Delaunay Triangulation Model

**Purpose:** Create a spatial null based on planar graph theory, where each node connects to neighbors in Delaunay triangulation cells. This preserves proximity-based constraints more rigorously than KNN.

**Algorithm:**

1. Extract all node coordinates as points in 2D space
2. Compute Delaunay triangulation using scipy.spatial.Delaunay
   - Input: $n \times 2$ array of (x, y) coordinates
   - Output: simplex mesh where each 3-node triangle represents a Delaunay cell
3. Extract edges from triangulation
   - Each triangle $(a, b, c)$ yields three edges: $(a,b), (b,c), (a,c)$
   - Deduplicate to create undirected graph H

**Distance Quantile Constraint** (unique to global Delaunay models):
- Problem: Delaunay can produce very long edges in sparse regions (e.g., remote provinces)
- Solution: Filter candidate edges using distance threshold
- Threshold: 90th percentile of **empirical edge lengths within each province** (when applied to province models)
- Implementation: Remove any Delaunay edge longer than this quantile after MST addition

**Edge Count (before filtering):**
- Theoretical: $\approx 3n - 6 \approx 25,374$ edges in convex hull triangulation
- Actual global Delaunay (post-MST): **22,555 edges** (pre-mask)
- After distance filtering & masking: **22,540 edges**

**Failure Robustness:**
- QHull degenerate case fallback: If Delaunay fails (collinear points), falls back to deterministic chain: sort nodes by (x, y) and connect adjacent pairs
- Fallback flag recorded as `mst_fallback_chain=1.0` in metadata

---

## Minimum Spanning Tree Backbone Addition

**Purpose:** Ensure connectivity in sparse spatial null models. KNN/Delaunay may generate disconnected components; MST guarantees single connected component.

**Algorithm (for models with `add_mst_backbone=True`):

1. **Candidate Graph Construction:**
   - Build candidate graph C from Delaunay triangulation of all $n$ nodes
   - Assign Euclidean distance as edge weight: $w(u,v) = ||(u) - (v)||_2$
   - If Delaunay fails, fall back to non-edge chain (see above)

2. **MST Computation:**
   - Apply Kruskal's algorithm on candidate C
   - Select $n-1$ edges with minimum total Euclidean distance
   - Result: $m_{\text{mst}}$ edges forming spanning tree of all nodes

3. **Backbone Integration:**
   - Iterate through MST edges
   - For each $(u, v)$ in MST: add to H if not already present
   - Non-redundant edges: count only added (not already in KNN/Delaunay)

**Statistics Recorded:**
- `mst_edges_candidate`: Total unique edges in Delaunay candidate graph
- `mst_edges_added`: Actual new edges added (existing edges not double-counted)
- `components_before_mst`: Number of disconnected components in H before adding MST
- `components_after_mst`: Number in combined H after MST (typically 1, rarely 2)
- `mst_fallback_chain`: Flag (0 or 1) indicating whether Delaunay failed and chain was used

**Global Model Results:**
| Model | Nodes | Components Before | MST Edges Added | Components After |
|-------|-------|-------------------|-----------------|------------------|
| KNN+MST | 8,460 | 53 | 211 | 1 |
| Delaunay+MST | 8,460 | 93 | 92 | 1 |

Note: Delaunay creates more components initially but requires fewer MST edges due to long-range edges already present.

---

## Edge Weight Assignment: `uniform_range_times`

**Purpose:** Assign travel time (`time_s`) weights to null model edges reflecting the empirical network's observed travel time distribution, without preserving rank or sampling bias.

**Algorithm:**

1. **Extract Empirical Range:**
   - Scan all edges in original graph G for `time_s` attribute
   - Record valid times: $t \in (0, \infty)$, finite values only
   - Calculate range: $t_{\min}$ and $t_{\max}$

2. **Assign Geometric Lengths:**
   - For each null edge $(u, v)$ in H:
     - Compute Euclidean distance: $d(u,v) = \sqrt{(x_u - x_v)^2 + (y_u - y_v)^2}$
     - Assign as `length_m` attribute: $H_{u,v}[\text{length\_m}] = d(u,v)$

3. **Uniform Sampling of Times:**
   - For each null edge $(u, v)$:
     - Sample uniformly: $t \sim \text{Uniform}(t_{\min}, t_{\max})$
     - Assign as `time_s` attribute: $H_{u,v}[\text{time\_s}] = t$

4. **Fallback Mechanism:**
   - If empirical graph has no usable `time_s` values:
     - Switch to `distance_speed` rule instead
     - Flag: `weight_rule_fallback=1.0`

**Empirical Parameters (Roman Road Network):**
- $t_{\min}$: ~262 seconds (fastest 1-node crossing)
- $t_{\max}$: ~165,000 seconds (longest cross-empire route)
- Mean: ~15,900 seconds per edge
- This range reflects: merchant caravan speeds, mountain terrain penalties, seasonal variability

**Why uniform_range_times?**
- Preserves empirical extremes (fastest/slowest routes) without assuming original rank structure
- Neutral with respect to distance (doesn't bias short edges toward fast vs. slow)
- Reasonable for null hypothesis: "Travel times distributed uniformly across observed range"

---

## Province-Bridged Model Construction

**Applies to:** Both KNN+MST and Delaunay+MST in province scope

**Purpose:** Respect province-level network structure. Rather than generating globally, build local spatial models per province, then connect provinces using empirically-observed cross-province corridors.

### Step 1: Per-Province Null Generation

For each of the 57 Roman provinces:
1. Extract subgraph $G_i$ = subset of empirical network within province $i$
2. Apply KNN or Delaunay spatial model independently (same rules as per-province case in global scope)
3. If `add_mst_backbone=True`: apply MST to ensure within-province connectivity
4. Assign edge weights using `uniform_range_times` (same range as global empirical)

**Result:** 57 independent KNN/Delaunay null subgraphs, preserving per-province road structure

### Step 2: Inter-Province MST Bridging

**Algorithm:** Connect provinces using minimum spanning tree over province-level graph.

**Graph Construction:**
1. Create province-level graph $P$: 
   - Nodes: 57 provinces (all provinces with $\geq 1$ node in 57 provinces)
   - Edges: initialized empty

2. For each province pair $(i, j)$ with $i < j$:
   - Find closest node pair between $G_i$ and $G_j$:
     - $u \in G_i$, $v \in G_j$
     - Minimize: $d_{\min}(i, j) = \min_{u \in G_i, v \in G_j} d(u, v)$
   - Add edge to $P$: $(i, j)$ with weight $d_{\min}(i, j)$

   **Alternative mode (mst_centroid):** Weight by distance between province centroids instead (not used in final models)

3. Apply Kruskal's MST algorithm on $P$:
   - Select $n_{\text{prov}} - 1 = 56$ edges with minimum total distance
   - This ensures 57 provinces connected in single component

4. For each selected MST edge $(i_{\text{mst}}, j_{\text{mst}})$:
   - Retrieve closest node pair: $(u, v, d)$ from earlier
   - If edge $(u, v)$ not already in combined graph: add it
   - Assign weight using speed-based rule: $\text{time\_s} = d / v_{\text{median}}$
   - $v_{\text{median}}$ = median speed across all empirical edges

**Results (all province bridged models):**
- Province bridge edges added: **56 edges**
- Province pairs considered: **1,596 pairs** (57 choose 2)
- Provinces connected: **57 → 1 component**

Example bridges added:
- Britannia ↔ Belgica: ~95 km (via Dover Strait shortest crossing)
- Sicilia ↔ Bruttium: ~3 km (straits of Messina)
- These are MST-selected overwater routes (but masked out in final models; see below)

---

## Empirical Province-Pair Adjacency Masking

**Purpose:** Remove unrealistic null edges crossing sea barriers or geographic separations not observed in empirical network.

**Problem Addressed:**
- Without masking, KNN/Delaunay might connect:
  - Britannia directly to Hispania (across English Channel)
  - Hispania directly to Africa Proconsularis (across Gibraltar Strait)
  - Sicilia as isolated island with long artificial cross-sea edges

**Algorithm:**

### Step 1: Extract Empirical Cross-Province Pairs

```
allowed_pairs = set()
for (u, v) in empirical_graph.edges():
    province_u = node_to_province[u]
    province_v = node_to_province[v]
    if province_u != province_v:
        key = (province_u, province_v) if province_u < province_v 
              else (province_v, province_u)
        allowed_pairs.add(key)
```

Result: **91 allowed cross-province pairs** (out of 1,596 possible)

Examples of non-observed pairs (masked):
- (Britannia, Gallia Lugdunensis): No empirical edges → no shortcuts allowed
- (Sicilia, mainland): Empirically connected only via Bruttium (1 edge crossing strait) → long Sicily-direct edges removed

### Step 2: Node ID Mapping (Coordinate-Based)

**Challenge:** 
- Global empirical graph uses integer node IDs (1, 2, 3, ...)
- Province subgraphs use coordinate tuples: $(x, y)$
- Direct node matching fails (0 overlap)

**Solution:** Match by coordinates:
```
coord_to_province = {}
for province_name, province_graph in subgraphs.items():
    for node_id in province_graph.nodes():
        coord_to_province[(node_id)] = province_name

node_to_province = {}
for node_id in global_graph.nodes():
    x, y = global_graph.nodes[node_id]['x'], global_graph.nodes[node_id]['y']
    coord_key = (x, y)
    if coord_key in coord_to_province:
        node_to_province[node_id] = coord_to_province[coord_key]
```

Mapping success rate: **8,417 / 8,460 nodes mapped** (99.5%)
- Unmapped nodes: boundary/extrapolated nodes in global graph but not in any province subgraph

### Step 3: Filter Null Edges

```
removed = 0
for (u, v) in null_graph.edges():
    prov_u = node_to_province.get(u)
    prov_v = node_to_province.get(v)
    
    if prov_u is None or prov_v is None or prov_u == prov_v:
        continue  # Skip unmapped or within-province
    
    pair_key = (prov_u, prov_v) if prov_u < prov_v 
               else (prov_v, prov_u)
    
    if pair_key not in allowed_pairs:
        null_graph.remove_edge(u, v)
        removed += 1
```

**Results:**

| Model | Generated | Removed | Final | Removed % |
|-------|-----------|---------|-------|-----------|
| Global KNN+MST | 21,811 | 14 | 21,808 | 0.06% |
| Global Delaunay+MST | 22,555 | 15 | 22,540 | 0.07% |
| Province-Bridged KNN+MST | 21,846 | - | 21,790 | - |
| Province-Bridged Delaunay+MST | 24,636 | - | 24,580 | - |

**Interpretation:** Small removal percentages indicate null models naturally respect observed province connectivity. Removed edges are concentrated in bridge edges (cross-province shortcuts).

**Default Behavior:** Masking is **ON by default** (disabled with `--allow-sea-crossings` flag)

---

## Delaunay Long-Edge Trimming (Province Models Only)

**Available for:** Province-scope Delaunay models with `--delaunay-max-dist-quantile` parameter

**Purpose:** Prevent within-province Delaunay triangulation from generating unrealistically long edges spanning across mountainous terrain.

**Algorithm:**

1. During per-province null generation:
   - Compute Delaunay triangulation for each province's nodes
   - Extract all candidate edges with Euclidean lengths
   - Sort by length, compute 90th percentile of empirical edge lengths **within that province**

2. Filter long edges:
   - Remove any Delaunay edge longer than this province-specific quantile
   - Preserve all edges at/below quantile

3. Apply MST backbone to reconnect any fragmented components

**Parameters:**
- `--delaunay-max-dist-quantile 0.90` (default): Keep only edges ≤ 90th percentile

**Empirical Result (Delaunay Province Models):**
- Long edges removed per province: ~30-100 (varies by province size)
- Total removed globally: ~2,901 edges across 57 provinces
- Typical removed length: > 50 km in densely-networked provinces

---

## Complete Generation Summary Table

| Parameter | Global KNN | Global Delaunay | Prov-Bridged KNN | Prov-Bridged Delaunay |
|-----------|-----------|-----------------|------------------|----------------------|
| **Spatial Model** | KNN, k=4 | Delaunay triangulation | Per-province KNN, k=4 | Per-province Delaunay |
| **Nodes** | 8,460 | 8,460 | 8,417 (per-province) | 8,417 (per-province) |
| **Generated Edges** | 21,611 | 22,463 | 21,790 | 24,580 |
| **MST Added** | +211 | +92 | +56 (bridges) | +56 (bridges) |
| **Pre-Mask Edges** | 21,822 | 22,555 | - | - |
| **Post-Mask Edges** | 21,808 | 22,540 | 21,790 | 24,580 |
| **Masked Edges** | 14 | 15 | 0 (province-aware) | 0 (province-aware) |
| **Weight Rule** | Uniform range | Uniform range | Uniform range | Uniform range |
| **Time Range** | [262s, 165k s] | [262s, 165k s] | [262s, 165k s] | [262s, 165k s] |
| **Distance Cap** | None | 90th percentile | None | 90th percentile |
| **Connectivity** | 1 component | 1 component | 1 component (57 via MST) | 1 component (57 via MST) |

---

## Output Files Generated

Each null model produces two outputs:

### Pickle Files (`.pkl`)
- **Structure:** Dict with keys:
  - `graph` (global scope): Single combined NetworkX graph
  - `province_subgraphs` (province scope): Dict of 57 province subgraphs
- **Node/Edge Data:** 
  - Nodes: `x`, `y` coordinates
  - Edges: `length_m` (geometric), `time_s` (weight), `weight` attribute
- **Metadata:** `meta` dict with all generation parameters

### Summary CSV (`.csv`)
- **Per-model statistics:** Edge counts, density, clustering, etc.
- **Comparison:** Original vs. null metrics for each graph
- **MST stats:** Components before/after, MST edges added
- **Weight stats:** Rule used, fallback flags
- **Masking stats:** Number of removed empirical cross-province mask edges

---

## References

- **Delaunay Triangulation:** Delaunay (1934); implemented via scipy.spatial.Delaunay (Qhull implementation)
- **Minimum Spanning Tree:** Kruskal's algorithm (1956) via NetworkX minimum_spanning_edges()
- **Spatial Network Null Models:** Erdős & Rényi (1959) baseline; spatial variants in Papadopoulos et al. (2012)
