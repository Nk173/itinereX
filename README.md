
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

### Option A (recommended): GitHub Pages

This repo includes a small GitHub Pages site in `docs/` that embeds the interactive map:

- Open the Pages site: https://<your-username>.github.io/<your-repo>/

To enable it:
- Repository **Settings → Pages → Build and deployment**
- **Source**: Deploy from a branch
- **Branch**: `main` (or your default branch), **Folder**: `/docs`

### Option B: Direct HTML link

- Open the HTML map: [roads_after_cleaning.html](roads_after_cleaning.html)

## How to run

1. Open the notebook: [itinere-X_network.ipynb](itinere-X_network.ipynb)
2. Run the cells top-to-bottom.

Outputs (written to repo root):
- [roads_before_cleaning.html](roads_before_cleaning.html)
- [roads_after_cleaning.html](roads_after_cleaning.html)
- `G_clean_weighted_with_ids.pkl`

