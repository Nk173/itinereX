
# itinereX

Builds a cleaned, analyzable road network graph from the itiner-E road dataset.

The goal of this repository is to take the source linework (often with near-miss junctions, duplicate endpoints, and degree-2 chains), rebuild a consistent topology, and export a weighted NetworkX graph that can be used for downstream analysis (routing, centrality, “important nodes”, etc.).

## What this repo produces

- An interactive **before/after** visualization of the network cleaning.
- A cleaned, noded topology with degree-2 simplification.
- A **time-weighted** graph export where each edge has `weight = time_s` (seconds) plus stable `node_id` / `edge_id` mappings.

## Cleaning schematic

![Network cleaning schematics](itinere_network_cleaning_schematics.png)

## Interactive map (after cleaning)

GitHub does not render embedded iframes in README files, so an embedded panel won’t work here.

Open the hosted map on GitHub Pages instead:

[![Open the interactive after-cleaning map](itinere_network_cleaning_schematics.png)](https://nk173.github.io/itinereX/roads_after_cleaning.html)

- Map: https://nk173.github.io/itinereX/roads_after_cleaning.html

## Network statistics

| Metric | Value |
|---|---:|
| nodes | 8460 |
| edges | 12849 |
| avg_degree | 3.0375886524822695 |
| avg_weighted_degree (sum of `weight` per node) | 167930.63592041272 (in seconds taken to travel from one node to another)|
| connected_components | 12 |
| largest_component_nodes | 7612 |
| largest_component_edges | 11693 |
| largest_component_fraction_of_nodes | 0.8997635933806146 |


## How to run

1. Open the notebook: [itinere-X_network.ipynb](itinere-X_network.ipynb)
2. Run the cells top-to-bottom.

Outputs (written to repo root):
- [roads_before_cleaning.html](roads_before_cleaning.html)
- [roads_after_cleaning.html](roads_after_cleaning.html)
- `G_clean_weighted_with_ids.pkl`



