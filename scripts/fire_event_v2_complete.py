"""
FireEvent V3 generation for West Bengal VIIRS detections.

V3 event definition (revised from V2)
--------------------------------------
A VIIRS detection is assigned to an existing FireEvent when:

1. It is within an adaptive spatial threshold of the NEAREST recent
   detection already in that event (not the event's running centroid
   average — see WHY V2 CHANGED below).
2. It occurs within 48 hours of the event's most recent detection.
3. The resulting event is no more than MAX_EVENT_DURATION_DAYS from
   its first detection (a generous safety valve, not a feature-
   shaping cutoff — see WHY V2 CHANGED below).

Otherwise, a new FireEvent is created.

WHY V2 CHANGED
--------------
Running V2 on the real West Bengal 2024 data surfaced two concrete
problems:

1. THE 7-DAY CAP WAS FRAGMENTING PERSISTENT SOURCES.
   274 events ran right up against the old 7-day cap, and 243 of
   those (89%) had another event start within 5 days afterward,
   within 1 km of the same centroid — i.e. the same physical
   persistent thermal source (almost certainly industrial: a flare,
   kiln, or plant) sliced into repeated ~7-day chunks. This directly
   corrupts the "persistence over weeks/months" and "historical
   behaviour" features the project depends on to tell a normal
   persistent industrial source apart from an abnormal one.

   FIX: the duration cap is now a generous safety valve
   (MAX_EVENT_DURATION_DAYS, default 120 days) rather than a
   feature-defining boundary, events are tagged "chronic" when they
   run long, and a post-processing pass (link_chained_events) still
   explicitly links any event that DOES hit the cap to its likely
   continuation, so persistence can be reconstructed even in the
   rare case a source outlives the safety valve.

2. CENTROID-DISTANCE MATCHING CAN SPLIT SPREADING FIRES.
   V2 checked distance from a new detection to the event's running
   MEAN position. For a genuinely spreading/elongated wildfire, a
   detection near one end of the fire front can end up farther than
   the threshold from the average of all points seen so far, even
   though it sits right next to the fire's actual edge — this can
   fragment one real wildfire into several artificial events and
   corrupt area-growth / spread-direction features.

   FIX: matching now checks distance to the NEAREST of the event's
   recent detections (single-linkage / density-reachability, closer
   to how DBSCAN would treat it), not distance to the mean.

3. A FIXED 1 KM THRESHOLD IGNORES VIIRS PIXEL FOOTPRINT GROWTH.
   VIIRS pixels are ~375 m at nadir but grow toward the swath edge
   (up to ~1-2 km). The dataset's own scan/track columns report this
   per-detection footprint. A fixed 1 km threshold is too generous
   for compact nadir pixels and too strict for large edge pixels.

   FIX: the spatial threshold now adds a small, capped buffer based
   on the average scan/track footprint of the two points being
   compared, falling back to the fixed 1 km threshold when scan/track
   aren't available.

Inputs
------
viirs_west_bengal_2024_clean_authoritative.csv

Outputs
-------
fire_events_west_bengal_2024_v2.csv
viirs_west_bengal_2024_detection_to_event_v2.csv

Required packages
-----------------
pip install pandas numpy shapely pyproj
"""

import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
from pyproj import Geod
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "viirs_west_bengal_2024_clean_authoritative.csv"
)

EVENT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fire_events_west_bengal_2024_v2.csv"
)

MAPPING_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "viirs_west_bengal_2024_detection_to_event_v2.csv"
)

# ------------------------------------------------------------
# FireEvent V3 thresholds
# ------------------------------------------------------------

# Base spatial threshold for a near-nadir VIIRS pixel (~375 m).
MAX_SPATIAL_KM = 1.0

# Gap threshold: the real event boundary. Two detections separated
# by more than this at the same/nearby location are treated as
# unrelated re-ignitions, not one continuous event.
MAX_GAP_HOURS = 48.0

# Safety valve only — NOT a feature-shaping cutoff (see module
# docstring, point 1). Generous enough that a genuine multi-month
# industrial source is captured as one event almost all the time.
MAX_EVENT_DURATION_DAYS = 120.0

# An event is tagged "persistent" once it exceeds this duration...
PERSISTENT_THRESHOLD_HOURS = 24.0

# ...and "chronic" once it exceeds this one. Chronic events are the
# strongest candidates for a normal/stable persistent industrial
# source rather than a wildfire.
CHRONIC_THRESHOLD_DAYS = 30.0

# How many of an event's most recent detections are kept for the
# nearest-point spatial check. Bounded so the check stays cheap even
# for very long chronic events; large enough to represent a fire
# front's actual current extent rather than just its last point.
LINKAGE_WINDOW_POINTS = 25

# Extra spatial buffer added per point based on its VIIRS scan/track
# footprint (km), capped so edge-of-swath pixels don't over-merge.
PIXEL_FOOTPRINT_BUFFER_FRACTION = 0.5
MAX_ADAPTIVE_BUFFER_KM = 0.5

# Post-processing chain-linking (see link_chained_events): how far
# apart (in time/space) two events can be while still being flagged
# as "probably the same physical source, split by the safety valve".
CHAIN_LINK_MAX_DAYS = 10.0
CHAIN_LINK_MAX_KM = 1.5

EARTH_RADIUS_KM = 6371.0088



# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two latitude/longitude
    points using the Haversine formula.

    Returns distance in kilometres.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# LOAD AND CLEAN VIIRS DATA
# ============================================================

def load_viirs_data(input_file):
    print("=" * 70)
    print("LOADING WEST BENGAL VIIRS DATA")
    print("=" * 70)

    df = pd.read_csv(input_file)

    print(f"Rows loaded: {len(df):,}")

    # Standardize important column types
    df["acq_datetime"] = pd.to_datetime(
        df["acq_datetime"],
        errors="coerce"
    )

    for column in [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track"
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove records that cannot participate in event creation
    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime"
        ]
    ).copy()

    # Validate geographic coordinates
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    # Sort chronologically
    df = (
        df.sort_values("acq_datetime")
        .reset_index(drop=True)
    )

    print(f"Valid detections: {len(df):,}")
    print(
        f"Date range: "
        f"{df['acq_datetime'].min()} "
        f"to "
        f"{df['acq_datetime'].max()}"
    )

    return df


# ============================================================
# ADAPTIVE SPATIAL THRESHOLD
# ============================================================

def pixel_footprint_km(scan, track):
    """
    Approximate ground footprint size (km) from VIIRS scan/track.
    Returns None if either value is missing/invalid.
    """

    if scan is None or track is None:
        return None

    if not (np.isfinite(scan) and np.isfinite(track)):
        return None

    return (float(scan) + float(track)) / 2.0


def spatial_threshold_km(footprint_a, footprint_b):
    """
    Base threshold plus a small, capped buffer for larger
    (typically swath-edge) VIIRS pixels. Falls back to the fixed
    base threshold when footprint info isn't available for either
    point being compared.
    """

    sizes = [
        size
        for size in (footprint_a, footprint_b)
        if size is not None
    ]

    if not sizes:
        return MAX_SPATIAL_KM

    avg_size = sum(sizes) / len(sizes)

    buffer_km = min(
        MAX_ADAPTIVE_BUFFER_KM,
        PIXEL_FOOTPRINT_BUFFER_FRACTION * avg_size
    )

    return MAX_SPATIAL_KM + buffer_km


# ============================================================
# CREATE FIRE EVENTS
# ============================================================

def create_fire_events(df):
    """
    Greedy temporal-spatial FireEvent construction (V3).

    Each incoming detection is compared with currently active
    events using NEAREST-POINT distance (to any of the event's most
    recent LINKAGE_WINDOW_POINTS detections, not the running mean),
    with an adaptive spatial threshold based on VIIRS pixel
    footprint, provided that:
        - time gap <= MAX_GAP_HOURS
        - total event age <= MAX_EVENT_DURATION_DAYS (safety valve)

    If no event qualifies, a new event is started.
    """

    print()
    print("=" * 70)
    print("CREATING FIRE EVENTS - V3")
    print("=" * 70)

    print(f"Base spatial threshold:  {MAX_SPATIAL_KM} km (+ adaptive buffer)")
    print(f"Temporal gap threshold:  {MAX_GAP_HOURS} hours")
    print(
        f"Duration safety valve:   "
        f"{MAX_EVENT_DURATION_DAYS} days"
    )
    print(f"Linkage window:          last {LINKAGE_WINDOW_POINTS} points/event")

    has_footprint = (
        "scan" in df.columns
        and "track" in df.columns
    )

    if not has_footprint:
        print(
            "[NOTE] scan/track columns not found — using fixed "
            f"{MAX_SPATIAL_KM} km threshold for all detections."
        )

    # Active events are events that can still accept new detections.
    active_events = []

    # All events ever created.
    event_records = []

    for index, row in df.iterrows():

        current_time = row["acq_datetime"]
        current_lat = float(row["latitude"])
        current_lon = float(row["longitude"])

        current_footprint = (
            pixel_footprint_km(
                row.get("scan"),
                row.get("track")
            )
            if has_footprint
            else None
        )

        # ----------------------------------------------------
        # Remove events that can no longer accept this point.
        # ----------------------------------------------------

        still_active = []

        for event in active_events:

            gap_hours = (
                current_time - event["last_time"]
            ).total_seconds() / 3600.0

            age_days = (
                current_time - event["start_time"]
            ).total_seconds() / 86400.0

            if (
                gap_hours <= MAX_GAP_HOURS
                and age_days <= MAX_EVENT_DURATION_DAYS
            ):
                still_active.append(event)

        active_events = still_active

        # ----------------------------------------------------
        # Find the best spatially-compatible active event using
        # nearest-point (not centroid) distance.
        # ----------------------------------------------------

        best_event = None
        best_distance = None

        for event in active_events:

            recent_lats = event["recent_lats"]
            recent_lons = event["recent_lons"]
            recent_footprints = event["recent_footprints"]

            distances_km = haversine_km(
                current_lat,
                current_lon,
                recent_lats,
                recent_lons
            )

            if current_footprint is None:
                thresholds_km = np.full(
                    len(recent_lats), MAX_SPATIAL_KM
                )
            else:
                thresholds_km = np.array([
                    spatial_threshold_km(current_footprint, fp)
                    for fp in recent_footprints
                ])

            compatible = distances_km <= thresholds_km

            if not np.any(compatible):
                continue

            event_min_distance = float(
                distances_km[compatible].min()
            )

            if (
                best_distance is None
                or event_min_distance < best_distance
            ):
                best_distance = event_min_distance
                best_event = event

        # ----------------------------------------------------
        # Assign to the best compatible event, or start a new one.
        # ----------------------------------------------------

        if best_event is not None:

            event = best_event

            event["indices"].append(index)
            event["last_time"] = current_time

            event["recent_lats"] = np.append(
                event["recent_lats"], current_lat
            )[-LINKAGE_WINDOW_POINTS:]

            event["recent_lons"] = np.append(
                event["recent_lons"], current_lon
            )[-LINKAGE_WINDOW_POINTS:]

            event["recent_footprints"] = (
                event["recent_footprints"]
                + [current_footprint]
            )[-LINKAGE_WINDOW_POINTS:]

        else:

            event = {
                "start_time": current_time,
                "last_time": current_time,
                "recent_lats": np.array([current_lat]),
                "recent_lons": np.array([current_lon]),
                "recent_footprints": [current_footprint],
                "indices": [index],
            }

            active_events.append(event)
            event_records.append(event)

    print(
        f"FireEvents created: {len(event_records):,}"
    )

    return event_records


# ============================================================
# ASSIGN EVENT IDS
# ============================================================

def assign_event_ids(df, event_records):
    """
    Give events stable human-readable IDs based on start time.
    """

    print()
    print("=" * 70)
    print("ASSIGNING EVENT IDS")
    print("=" * 70)

    # Ensure events are ordered chronologically.
    event_records = sorted(
        event_records,
        key=lambda event: event["start_time"]
    )

    index_to_event = {}

    for event_number, event in enumerate(
        event_records,
        start=1
    ):

        event_id = (
            f"WB2024_EVT_{event_number:06d}"
        )

        for index in event["indices"]:
            index_to_event[index] = event_id

    # Every detection must map to exactly one event.
    df["event_id"] = [
        index_to_event[index]
        for index in range(len(df))
    ]

    return df


# ============================================================
# BUILD EVENT-LEVEL FEATURES
# ============================================================

def build_event_dataset(df):
    """
    Convert detection-level data into one row per FireEvent.
    """

    print()
    print("=" * 70)
    print("BUILDING EVENT-LEVEL DATASET")
    print("=" * 70)

    geod = Geod(ellps="WGS84")

    events = []

    for event_id, group in df.groupby(
        "event_id",
        sort=False
    ):

        group = group.sort_values(
            "acq_datetime"
        )

        # ----------------------------------------------------
        # Temporal features
        # ----------------------------------------------------

        start_time = group["acq_datetime"].min()
        end_time = group["acq_datetime"].max()

        duration_hours = (
            end_time - start_time
        ).total_seconds() / 3600.0

        detection_count = len(group)

        # ----------------------------------------------------
        # Spatial features
        # ----------------------------------------------------

        latitudes = group["latitude"].to_numpy()
        longitudes = group["longitude"].to_numpy()

        centroid_lat = float(latitudes.mean())
        centroid_lon = float(longitudes.mean())

        # Convex hull of detection points.
        # For one point this is a POINT.
        # For two points this is a LINESTRING.
        # For >=3 non-collinear points this is a POLYGON.
        multipoint = MultiPoint(
            list(zip(longitudes, latitudes))
        )

        footprint_wkt = (
            multipoint.convex_hull.wkt
        )

        # First-to-last displacement.
        first_lon = float(
            group.iloc[0]["longitude"]
        )
        first_lat = float(
            group.iloc[0]["latitude"]
        )

        last_lon = float(
            group.iloc[-1]["longitude"]
        )
        last_lat = float(
            group.iloc[-1]["latitude"]
        )

        _, _, displacement_m = geod.inv(
            first_lon,
            first_lat,
            last_lon,
            last_lat
        )

        displacement_km = (
            displacement_m / 1000.0
        )

        # ----------------------------------------------------
        # Thermal features
        # ----------------------------------------------------

        frp = group["frp"].dropna()

        bright_ti4 = group[
            "bright_ti4"
        ].dropna()

        bright_ti5 = group[
            "bright_ti5"
        ].dropna()

        event = {
            "event_id": event_id,

            # Temporal
            "start_time": start_time,
            "end_time": end_time,
            "duration_hours": round(
                duration_hours,
                3
            ),
            "detection_count": int(
                detection_count
            ),

            # Spatial
            "centroid_lat": round(
                centroid_lat,
                6
            ),
            "centroid_lon": round(
                centroid_lon,
                6
            ),
            "footprint_wkt": footprint_wkt,
            "displacement_km": round(
                displacement_km,
                4
            ),

            # Thermal
            "mean_frp": (
                float(frp.mean())
                if len(frp)
                else np.nan
            ),

            "max_frp": (
                float(frp.max())
                if len(frp)
                else np.nan
            ),

            "frp_variance": (
                float(frp.var(ddof=0))
                if len(frp)
                else np.nan
            ),

            "mean_brightness_ti4": (
                float(bright_ti4.mean())
                if len(bright_ti4)
                else np.nan
            ),

            "max_brightness_ti4": (
                float(bright_ti4.max())
                if len(bright_ti4)
                else np.nan
            ),

            "mean_brightness_ti5": (
                float(bright_ti5.mean())
                if len(bright_ti5)
                else np.nan
            ),

            "max_brightness_ti5": (
                float(bright_ti5.max())
                if len(bright_ti5)
                else np.nan
            ),
        }

        events.append(event)

    events_df = pd.DataFrame(events)

    # --------------------------------------------------------
    # Event type
    # --------------------------------------------------------
    #
    # "chronic" events (running well beyond a typical wildfire's
    # lifetime) are the strongest candidates for a normal, stable
    # persistent industrial source rather than a wildfire — this is
    # the signal the project's normal-vs-abnormal-industrial
    # extension needs.

    events_df["event_type"] = np.select(
        [
            events_df["detection_count"].eq(1),
            events_df["duration_hours"].gt(
                CHRONIC_THRESHOLD_DAYS * 24
            ),
            events_df["duration_hours"].gt(
                PERSISTENT_THRESHOLD_HOURS
            ),
        ],
        [
            "single_detection",
            "chronic",
            "persistent",
        ],
        default="multi_detection"
    )

    # --------------------------------------------------------
    # Quality/review flag
    # --------------------------------------------------------
    #
    # These are NOT additional event-definition rules.
    # They identify events worth inspecting manually.
    #
    # near_duration_cap specifically flags events that ran into the
    # MAX_EVENT_DURATION_DAYS safety valve — these are exactly the
    # events link_chained_events() below tries to reconnect to their
    # likely continuation.

    events_df["near_duration_cap"] = (
        events_df["duration_hours"]
        >= (MAX_EVENT_DURATION_DAYS * 24 - 24)
    )

    events_df["review_flag"] = (
        events_df["near_duration_cap"]
        |
        (
            events_df["displacement_km"] > 5
        )
    )

    return events_df


# ============================================================
# LINK CHAINED EVENTS
# ============================================================

def link_chained_events(events_df):
    """
    For events that ran into the MAX_EVENT_DURATION_DAYS safety
    valve (near_duration_cap), look for a spatially-close event that
    starts shortly after this one ends, and link them.

    This is exactly the check that surfaced the V2 fragmentation
    problem in the first place (89% of long-duration V2 events had
    such a follow-up within 5 days / 1 km) — running it here means
    downstream feature-engineering can reconstruct true multi-month
    persistence even on the rare occasion a source outlives the
    safety valve, instead of silently treating each chunk as an
    unrelated short-lived event.

    Adds two columns:
        linked_next_event_id : the likely continuation of this event
        linked_prev_event_id : the likely predecessor of this event
    """

    events_df = events_df.copy()

    events_df["linked_next_event_id"] = None
    events_df["linked_prev_event_id"] = None

    near_cap = events_df[events_df["near_duration_cap"]]

    if near_cap.empty:
        return events_df

    for _, row in near_cap.iterrows():

        window_start = row["end_time"]
        window_end = row["end_time"] + pd.Timedelta(
            days=CHAIN_LINK_MAX_DAYS
        )

        candidates = events_df[
            (events_df["event_id"] != row["event_id"])
            & (events_df["start_time"] >= window_start)
            & (events_df["start_time"] <= window_end)
        ]

        if candidates.empty:
            continue

        distances_km = haversine_km(
            row["centroid_lat"],
            row["centroid_lon"],
            candidates["centroid_lat"].to_numpy(),
            candidates["centroid_lon"].to_numpy()
        )

        close_enough = distances_km <= CHAIN_LINK_MAX_KM

        if not np.any(close_enough):
            continue

        close_candidates = candidates.loc[close_enough].copy()
        close_candidates["_distance_km"] = distances_km[close_enough]

        best = close_candidates.sort_values(
            ["_distance_km", "start_time"]
        ).iloc[0]

        events_df.loc[
            events_df["event_id"] == row["event_id"],
            "linked_next_event_id"
        ] = best["event_id"]

        events_df.loc[
            events_df["event_id"] == best["event_id"],
            "linked_prev_event_id"
        ] = row["event_id"]

    linked_count = events_df["linked_next_event_id"].notna().sum()

    print(
        f"\nChain-linked events: {linked_count:,} "
        f"of {near_cap.shape[0]:,} events near the duration cap "
        "were linked to a likely continuation."
    )

    return events_df


# ============================================================
# CREATE DETECTION -> EVENT MAPPING
# ============================================================

def create_detection_mapping(df):
    """
    Keep a traceability table so every FireEvent can be
    traced back to its original VIIRS detections.
    """

    columns = [
        "detection_id",
        "event_id",
        "latitude",
        "longitude",
        "acq_datetime",
        "frp",
        "confidence"
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[columns].copy()


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check(df, events_df, mapping_df):
    """
    Check the most important invariants before saving.
    """

    print()
    print("=" * 70)
    print("FINAL SANITY CHECK")
    print("=" * 70)

    total_detections = len(df)
    total_events = len(events_df)

    single_events = (
        events_df["detection_count"]
        .eq(1)
        .sum()
    )

    multi_events = (
        events_df["detection_count"]
        .gt(1)
        .sum()
    )

    persistent_events = (
        events_df["duration_hours"]
        .gt(PERSISTENT_THRESHOLD_HOURS)
        .sum()
    )

    chronic_events = (
        events_df["event_type"]
        .eq("chronic")
        .sum()
    )

    events_over_cap = (
        events_df["duration_hours"]
        > MAX_EVENT_DURATION_DAYS * 24
        + 1e-6
    ).sum()

    chained_events = (
        events_df["linked_next_event_id"].notna().sum()
        if "linked_next_event_id" in events_df.columns
        else 0
    )

    events_over_5km = (
        events_df["displacement_km"]
        > 5
    ).sum()

    duplicate_detection_ids = (
        mapping_df["detection_id"]
        .duplicated()
        .sum()
        if "detection_id" in mapping_df.columns
        else "N/A"
    )

    unique_detection_ids = (
        mapping_df["detection_id"]
        .nunique()
        if "detection_id" in mapping_df.columns
        else "N/A"
    )

    events_without_detections = (
        set(events_df["event_id"])
        - set(mapping_df["event_id"])
    )

    print(
        f"Input detections:          "
        f"{total_detections:,}"
    )

    print(
        f"FireEvents:                "
        f"{total_events:,}"
    )

    print(
        f"Single-detection events:   "
        f"{single_events:,}"
    )

    print(
        f"Multi-detection events:    "
        f"{multi_events:,}"
    )

    print(
        f"Persistent (>24h):         "
        f"{persistent_events:,}"
    )

    print(
        f"Chronic (>{CHRONIC_THRESHOLD_DAYS:g} days):        "
        f"{chronic_events:,}"
    )

    print(
        f"Events at duration cap ({MAX_EVENT_DURATION_DAYS:g}d): "
        f"{events_over_cap:,}"
    )

    print(
        f"  ...chain-linked to a continuation: "
        f"{chained_events:,}"
    )

    print(
        f"Events >5 km displacement: "
        f"{events_over_5km:,}"
    )

    print(
        f"Unique detection IDs:      "
        f"{unique_detection_ids:,}"
        if isinstance(unique_detection_ids, int)
        else
        f"Unique detection IDs:      {unique_detection_ids}"
    )

    print(
        f"Duplicate detection IDs:   "
        f"{duplicate_detection_ids:,}"
        if isinstance(duplicate_detection_ids, int)
        else
        f"Duplicate detection IDs:   {duplicate_detection_ids}"
    )

    print(
        f"Events without detections: "
        f"{len(events_without_detections):,}"
    )

    # Assertions catch unexpected problems.
    assert total_detections == len(mapping_df), (
        "Not every detection appears in the mapping."
    )

    assert events_without_detections == set(), (
        "At least one FireEvent has no mapped detection."
    )

    if "detection_id" in mapping_df.columns:
        assert duplicate_detection_ids == 0, (
            "A detection is mapped to multiple events."
        )

    assert events_over_cap == 0, (
        "An event exceeds the V3 duration safety valve — this "
        "should be impossible by construction."
    )

    print()
    print("SANITY CHECK PASSED")


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(events_df, mapping_df):
    """
    Save the two final V2 CSV files.
    """

    events_df.to_csv(
        EVENT_OUTPUT_FILE,
        index=False
    )

    mapping_df.to_csv(
        MAPPING_OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"FireEvent dataset:\n"
        f"  {Path(EVENT_OUTPUT_FILE).resolve()}"
    )

    print(
        f"Detection mapping:\n"
        f"  {Path(MAPPING_OUTPUT_FILE).resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_viirs_data(
        INPUT_FILE
    )

    event_records = create_fire_events(
        df
    )

    df = assign_event_ids(
        df,
        event_records
    )

    events_df = build_event_dataset(
        df
    )

    events_df = link_chained_events(
        events_df
    )

    mapping_df = create_detection_mapping(
        df
    )

    run_sanity_check(
        df,
        events_df,
        mapping_df
    )

    save_outputs(
        events_df,
        mapping_df
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V3 COMPLETE")
    print("=" * 70)

    print(
        f"Detections processed: "
        f"{len(df):,}"
    )

    print(
        f"FireEvents created: "
        f"{len(events_df):,}"
    )

    print()
    print("Largest events by detection count:")

    print(
        events_df.nlargest(
            10,
            "detection_count"
        )[
            [
                "event_id",
                "detection_count",
                "duration_hours",
                "displacement_km",
                "event_type",
                "review_flag"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()