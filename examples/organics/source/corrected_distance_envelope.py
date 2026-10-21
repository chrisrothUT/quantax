"""Opt-in minimum-image envelope for new TriangularB experiments.

Only static distances are changed on a copy of the model. Existing installed
Quantax sources and model parameters are unchanged.
"""

from pathlib import Path
import copy, hashlib
import numpy as np
import quantax as qtx
from dimer_correlations import minimum_distances

POLICY = "true_minimum_image_skew_cell_v1"


def metadata():
    return dict(
        distance_envelope_policy=POLICY,
        distance_envelope_helper_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )


def configure(model, config):
    if config.get("distance_envelope_policy") != POLICY:
        raise ValueError("Corrected distance policy must be explicit")
    if (
        config["distance_envelope_helper_sha256"]
        != metadata()["distance_envelope_helper_sha256"]
    ):
        raise ValueError("Distance-envelope helper changed")
    lat = qtx.get_lattice()
    if lat.shape[0] != 1 or model._nlabels != 2:
        raise ValueError(
            "This policy currently supports one-site-cell spinful lattices"
        )
    distances = minimum_distances(lat.coord, lat.shape[1:], lat.basis_vectors)
    extent = np.array(lat.shape[1:])
    xyz = lat.xyz_from_index[:, 1:]
    inverse = np.ravel_multi_index(((-xyz) % extent).T, tuple(extent))
    np.testing.assert_allclose(distances, distances[inverse], atol=1e-12)
    result = copy.copy(model)
    object.__setattr__(
        result,
        "_distances",
        tuple(np.broadcast_to(distances, (2, 2, len(distances))).ravel().tolist()),
    )
    return result
