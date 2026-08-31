"""
FireEvent V2 generation for West Bengal VIIRS detections.

V2 event definition
-------------------
A VIIRS detection is assigned to an existing FireEvent when:

1. It is within 1.0 km of the event's current centroid.
2. It occurs within 48 hours of the event's most recent detection.
3. The resulting event is no more than 7 days from its first detection.

Otherwise, a new FireEvent is created.

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

# FireEvent V2 thresholds
MAX_SPATIAL_KM = 1.0
MAX_GAP_HOURS = 48.0
MAX_EVENT_DURATION_DAYS = 7.0

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
        "bright_ti5"
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
# CREATE FIRE EVENTS
# ============================================================

def create_fire_events(df):
    """
    Greedy temporal-spatial FireEvent construction.

    Each incoming detection is compared with currently active
    events. The closest event centroid within 1 km is selected,
    provided that:
        - time gap <= 48 hours
        - total event age <= 7 days

    If no event qualifies, a new event is started.
    """

    print()
    print("=" * 70)
    print("CREATING FIRE EVENTS - V2")
    print("=" * 70)

    print(f"Spatial threshold:       {MAX_SPATIAL_KM} km")
    print(f"Temporal gap threshold:  {MAX_GAP_HOURS} hours")
    print(
        f"Maximum event duration: "
        f"{MAX_EVENT_DURATION_DAYS} days"
    )

    # Active events are events that can still accept new detections.
    active_events = []

    # All events ever created.
    event_records = []

    for index, row in df.iterrows():

        current_time = row["acq_datetime"]
        current_lat = float(row["latitude"])
        current_lon = float(row["longitude"])

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
        # Find spatially compatible active events.
        # ----------------------------------------------------

        candidates = []

        for event in active_events:

            distance_km = haversine_km(
                current_lat,
                current_lon,
                event["centroid_lat"],
                event["centroid_lon"]
            )

            if distance_km <= MAX_SPATIAL_KM:

                candidates.append(
                    (distance_km, event)
                )

        # ----------------------------------------------------
        # Assign to nearest compatible event.
        # ----------------------------------------------------

        if candidates:

            distance_km, event = min(
                candidates,
                key=lambda x: x[0]
            )

            event["indices"].append(index)
            event["last_time"] = current_time

            # Incrementally update event centroid.
            n = len(event["indices"])

            event["centroid_lat"] += (
                current_lat - event["centroid_lat"]
            ) / n

            event["centroid_lon"] += (
                current_lon - event["centroid_lon"]
            ) / n

        # ----------------------------------------------------
        # No compatible event -> create a new event.
        # ----------------------------------------------------

        else:

            event = {
                "start_time": current_time,
                "last_time": current_time,
                "centroid_lat": current_lat,
                "centroid_lon": current_lon,
                "indices": [index]
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

    events_df["event_type"] = np.select(
        [
            events_df["detection_count"].eq(1),
            events_df["duration_hours"].gt(24)
        ],
        [
            "single_detection",
            "persistent"
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

    events_df["review_flag"] = (
        (
            events_df["duration_hours"]
            > MAX_EVENT_DURATION_DAYS * 24
            + 1e-6
        )
        |
        (
            events_df["displacement_km"] > 5
        )
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
        .gt(24)
        .sum()
    )

    events_over_7_days = (
        events_df["duration_hours"]
        > MAX_EVENT_DURATION_DAYS * 24
        + 1e-6
    ).sum()

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
        f"Events >7 days:            "
        f"{events_over_7_days:,}"
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

    assert events_over_7_days == 0, (
        "An event exceeds the 7-day V2 limit."
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
    print("V2 COMPLETE")
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
