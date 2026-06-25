"""
GPS-based neighbor selection and flight-grid overlap estimation.

Public API:
  haversine_distance()   - ground distance between two GPS coords (meters)
  estimate_overlap()     - analyse capture GPS to estimate forward/side overlap %
                           and derive safe SfM parameters (keyframe interval,
                           n_neighbors) automatically.
  build_neighbor_pairs() - return unique matching pairs for feature matching
"""

import math
from typing import List, NamedTuple, Tuple

from ..ingestion.capture import Capture


# -- Haversine distance --------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Ground distance in meters between two WGS84 GPS coordinates.
    Public - also used by src.sfm.keyframes for flight-path ordering.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Backwards-compatible private alias used by some internal callers
_haversine_distance = haversine_distance


# -- Overlap estimation --------------------------------------------------------

class FlightOverlapEstimate(NamedTuple):
    """
    Estimated overlap statistics for a drone mission.

    forward_spacing_m  : median along-track distance between sequential shots (m)
    side_spacing_m     : median cross-track distance between adjacent flight lines (m)
    forward_overlap    : estimated forward overlap fraction  (0.0 to 1.0)
    side_overlap       : estimated side overlap fraction     (0.0 to 1.0)
    footprint_m        : estimated ground footprint size (m) derived from altitude + FOV
    n_neighbors        : recommended n_neighbors for LightGlue matching
    keyframe_interval  : recommended keyframe interval for SfM
    """
    forward_spacing_m : float
    side_spacing_m    : float
    forward_overlap   : float
    side_overlap      : float
    footprint_m       : float
    n_neighbors       : int
    keyframe_interval : int


def estimate_overlap(captures: List[Capture]) -> FlightOverlapEstimate:
    """
    Estimate flight overlap from GPS metadata alone, then recommend SfM parameters.

    Strategy
    --------
    1. Forward spacing  -- median distance between consecutive captures (along-track).

    2. Footprint        -- estimated from median altitude + typical 80 deg horizontal FOV:
                              footprint = 2 * altitude * tan(FOV/2)
                          Falls back to 120 m altitude if EXIF altitude is absent.

    3. Side spacing     -- for each sampled capture, find the nearest neighbor that is
                          at least `min_skip` sequential indices away (to avoid same-line
                          images). min_skip = images_per_line = footprint / forward_spacing.

    4. Overlap fractions:
                          forward_overlap = 1 - forward_spacing / footprint
                          side_overlap    = 1 - side_spacing   / footprint
                          Both clamped to [0.0, 0.99].

    5. SfM parameters derived from overlap:
         keyframe_interval:
           forward_overlap >= 0.85  ->  3  (dense: 8+ views per ground point)
           forward_overlap >= 0.70  ->  2  (moderate)
           otherwise                ->  1  (sparse: cannot skip any image)

         n_neighbors:
           Must bridge at least one full flight line.
           n_neighbors = images_per_line + 5 (safety buffer), minimum 10.

    Returns FlightOverlapEstimate with all-zero / conservative defaults when
    fewer than 20 GPS captures exist (not enough data to estimate reliably).
    """
    import numpy as np

    gps = [c for c in captures if c.latitude is not None and c.longitude is not None]
    n   = len(gps)

    _default = FlightOverlapEstimate(
        forward_spacing_m=0.0, side_spacing_m=0.0,
        forward_overlap=0.0, side_overlap=0.0, footprint_m=0.0,
        n_neighbors=20, keyframe_interval=1,
    )

    if n < 20:
        return _default

    # 1. Forward spacing
    fwd_dists = [
        haversine_distance(
            gps[i].latitude,  gps[i].longitude,
            gps[i+1].latitude, gps[i+1].longitude,
        )
        for i in range(n - 1)
    ]
    median_fwd = float(np.median(fwd_dists))
    if median_fwd < 1e-3:
        return _default  # all images at same GPS point

    # 2. Footprint from altitude + typical FOV
    altitudes  = [c.altitude for c in gps if c.altitude is not None]
    median_alt = float(np.median(altitudes)) if altitudes else 120.0
    fov_rad    = math.radians(80)   # typical DJI / Autel horizontal FOV
    footprint  = 2.0 * median_alt * math.tan(fov_rad / 2.0)

    # 3. Side (cross-track) spacing
    # Estimate how many sequential shots cover one footprint length (one line width)
    imgs_per_line = max(5, int(round(footprint / max(median_fwd, 1.0))))
    min_skip      = imgs_per_line

    side_dists: list = []
    for i in range(0, n, 4):       # sample every 4th capture for speed
        best  = math.inf
        lat_i = gps[i].latitude
        lon_i = gps[i].longitude
        for j in range(n):
            if abs(i - j) <= min_skip:
                continue
            d = haversine_distance(lat_i, lon_i, gps[j].latitude, gps[j].longitude)
            if d < best:
                best = d
        if best < math.inf:
            side_dists.append(best)

    median_side = float(np.median(side_dists)) if side_dists else footprint

    # 4. Overlap fractions
    fwd_overlap  = max(0.0, min(0.99, 1.0 - median_fwd  / footprint))
    side_overlap = max(0.0, min(0.99, 1.0 - median_side / footprint))

    # 5. SfM parameters
    if fwd_overlap >= 0.85:
        kf_interval = 3
    elif fwd_overlap >= 0.70:
        kf_interval = 2
    else:
        kf_interval = 1

    n_neighbors = max(10, imgs_per_line + 5)

    return FlightOverlapEstimate(
        forward_spacing_m = round(median_fwd,  1),
        side_spacing_m    = round(median_side, 1),
        forward_overlap   = round(fwd_overlap,  2),
        side_overlap      = round(side_overlap, 2),
        footprint_m       = round(footprint,    1),
        n_neighbors       = n_neighbors,
        keyframe_interval = kf_interval,
    )


# -- Neighbor pair builder -----------------------------------------------------

def build_neighbor_pairs(
    captures: List[Capture],
    n_neighbors: int = 20,
) -> List[Tuple[str, str]]:
    """
    For each capture, find the N nearest captures by GPS distance.
    Returns sorted unique pairs (capture_id_a, capture_id_b).

    Reduces matching from O(n^2) exhaustive to O(n * n_neighbors).
    Example: 1000 images -> 499,500 exhaustive vs ~10,000 GPS-filtered pairs.
    """
    with_gps    = [c for c in captures if c.latitude is not None and c.longitude is not None]
    without_gps = [c for c in captures if c.latitude is None or c.longitude is None]

    if without_gps:
        print(f"[neighbors] {len(without_gps)} captures have no GPS -- excluded from neighbor pairs.")

    if not with_gps:
        raise ValueError("No captures with GPS coordinates. Cannot build neighbor pairs.")

    pairs: set = set()

    for i, cap_a in enumerate(with_gps):
        distances = [
            (haversine_distance(cap_a.latitude, cap_a.longitude, cap_b.latitude, cap_b.longitude),
             cap_b.capture_id)
            for j, cap_b in enumerate(with_gps) if i != j
        ]
        for _, neighbor_id in sorted(distances)[:n_neighbors]:
            pairs.add(tuple(sorted([cap_a.capture_id, neighbor_id])))

    pair_list  = sorted(pairs)
    exhaustive = len(with_gps) * (len(with_gps) - 1) // 2
    print(f"[neighbors] {len(with_gps)} captures -> {len(pair_list)} GPS-neighbor pairs "
          f"(n_neighbors={n_neighbors}, exhaustive would be {exhaustive})")
    return pair_list
