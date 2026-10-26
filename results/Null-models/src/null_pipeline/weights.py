from __future__ import annotations

import networkx as nx

from .time_estimation import TravelTimePredictor


def assign_predicted_weights(
    G: nx.Graph,
    predictor: TravelTimePredictor,
) -> nx.Graph:
    """Assign dist_m and time_s to every null edge using the shared predictor."""
    H = G.copy()
    for u, v in H.edges():
        dist_m = predictor.pair_distance_m(int(u), int(v))
        time_s = predictor.predict_pair_travel_time(int(u), int(v))
        H[u][v]["dist_m"] = float(dist_m)
        H[u][v]["time_s"] = float(time_s)
    return H
