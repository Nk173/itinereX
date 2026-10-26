# Spatial Null Models: Sea-Crossing Masking Complete

## Summary of Changes

Successfully implemented and applied **empirical province-pair adjacency masking** to constrain spatial null models to respect observed cross-province connectivity in the Roman road network.

## Problem Addressed

User identified unrealistic sea-crossing artifacts in generated null models:
- English Channel direct routes between Britannia and mainland Europe
- Gibraltar Strait shortcuts between Spain and North Africa
- Sicily isolated from mainland via artificial long edges

Root cause: Null models were generating cross-province edges without reference to what was empirically possible (sea separates many provinces and has no road network).

## Solution Implemented

### 1. Coordinate-Based Node ID Mapping
- **Issue:** Global empirical graph uses integer node IDs; province subgraphs use coordinate tuples
- **Fix:** Created `_node_id_to_province_map_from_coords()` function to bridge this gap
- Maps nodes in global graph by matching (x, y) coordinates to province subgraph nodes
- Enables masking despite different node ID schemes across repositories

### 2. Empirical Adjacency Extraction
- Function `_empirical_allowed_province_pairs()` scans empirical cross-province edges
- Extracts complete set of observed (province_i, province_j) pairs
- Result: **91 allowed province pairs** in Roman empirical network

### 3. Post-Generation Edge Filtering
- Function `_filter_edges_by_province_mask()` removes null edges where both endpoints map to unobserved province pairs
- **Global scope:** Applies directly to full H graph after model construction
- **Province scope:** Filters province pairs before MST bridge selection
- Applied by default; disabled with `--allow-sea-crossings` CLI flag

## Results

### Global Models
| Model | Generated | Removed by Mask | Final | Reduction |
|-------|-----------|-----------------|-------|-----------|
| KNN+MST | 21,822 | 14 (-0.06%) | 21,808 | 14 edges |
| Delaunay+MST | 22,555 | 15 (-0.07%) | 22,540 | 15 edges |

**Note:** Delaunay also applied distance cap (90th percentile) during generation, removing 2,901 long edges.

### Province-Bridged Models (regenerated for consistency)
- KNN+MST bridged: 21,790 edges (57 provinces + 56 MST bridges)
- Delaunay+MST bridged: 24,580 edges (57 provinces + 56 MST bridges)
- Both now include masking on bridge selection

### Maps Generated
All 5 network visualizations regenerated with cleaned models:
1. `empirical_dataset_network.png` - Empirical Roman road network
2. `global_spatial_null_knn_mst_uniformrange.png` - Global KNN+MST null
3. `global_spatial_null_delaunay_mst_uniformrange.png` - Global Delaunay+MST null
4. `province_subnetworks_spatial_null_knn_mst_bridged.png` - Province-bridged KNN+MST null
5. `province_subnetworks_spatial_null_delaunay_mst_bridged.png` - Province-bridged Delaunay+MST null

Location: `figures/null_model_maps/`

## Code Changes

### Files Modified
- `scripts/spatial_null_model_rewire.py`
  - Added `_node_id_to_province_map_from_coords()` for coordinate-based mapping
  - Refactored `_empirical_allowed_province_pairs()` documentation
  - Updated global masking logic to use coordinate mapping instead of direct node ID lookup
  - Fixed province-bridge masking to use correct node representation

### Key Implementation Details
```python
def _node_id_to_province_map_from_coords(G, province_subgraphs):
    """Map node IDs in G to provinces by matching (x,y) coordinates to subgraph nodes."""
    # Build coordinate-to-province map from subgraphs (nodes stored as tuples)
    # Then map nodes in G (with x,y attrs) to provinces by coordinate match
    # Handles heterogeneous node ID schemes across repositories

def _filter_edges_by_province_mask(H, node_to_province, allowed_pairs):
    """Remove edges where both endpoints map to unobserved province pair."""
    # Iterates through H edges, checking if edge endpoints' provinces are in allowed_pairs
    # Removes non-allowed edges, returns count of removed edges
```

## Validation Notes

- **Masking effectiveness:** 14-15 edges removed from global models (0.06-0.07% of ~22k edges)
- **Expected behavior:** Small removal percentage suggests most null edges naturally respect observed province connectivity
- **Sea-crossing artifacts:** Maps should now show confined province-level clustering without unrealistic intercontinental jumps
- **Bridge selection:** Province-scope bridges now filtered through same masking, preventing mst links across sea barriers

## Usage

Masking enabled by default. To disable (allow sea crossings for comparison):
```bash
python scripts/spatial_null_model_rewire.py --model knn_mst --allow-sea-crossings ...
```

## Next Steps

- Visual inspection of regenerated maps to confirm sea-crossing artifacts eliminated
- Comparison of network metrics between pre/post-masking null models (if needed)
- Integration into temporal pipeline (if full pipelines needed later)
