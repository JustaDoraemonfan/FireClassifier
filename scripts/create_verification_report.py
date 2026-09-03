"""
FireDistinguish
---------------
Create a unified verification review dataset.

Inputs:
    data/verification/verification_candidates_v2_osm.csv
    data/verification/satellite/satellite_evidence_v1.csv

Output:
    data/verification/verification_review_v1.csv

This script does NOT modify the input datasets.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OSM_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_candidates_v2_osm.csv"
)

SATELLITE_INDICES_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "satellite_indices_v1.csv"
)

SATELLITE_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "satellite_evidence_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_review_v1.csv"
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"\n[ERROR] {message}")
    sys.exit(1)


def check_file(path):
    if not path.exists():
        fail(f"File not found:\n{path}")


# ============================================================
# START
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — VERIFICATION REPORT")
print("=" * 75)


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "-" * 75)
print("1. CHECKING INPUT DATASETS")
print("-" * 75)

check_file(OSM_FILE)
check_file(SATELLITE_FILE)

print(f"[PASS] OSM dataset found")
print(f"       {OSM_FILE}")

print(f"[PASS] Satellite dataset found")
print(f"       {SATELLITE_FILE}")


# ============================================================
# LOAD
# ============================================================

print("\n" + "-" * 75)
print("2. LOADING DATA")
print("-" * 75)

try:
    osm_df = pd.read_csv(OSM_FILE)
except Exception as exc:
    fail(f"Could not read OSM dataset: {exc}")

try:
    satellite_df = pd.read_csv(SATELLITE_FILE)
except Exception as exc:
    fail(f"Could not read satellite dataset: {exc}")


# Quantitative spectral-index evidence is optional at report-generation time.
# If the index stage has run, merge its event-level statistics into the review
# dataset. Do not make the report unusable merely because indices have not run.
satellite_indices_df = None
if SATELLITE_INDICES_FILE.exists():
    try:
        satellite_indices_df = pd.read_csv(SATELLITE_INDICES_FILE)
        print(
            f"Satellite index records: {len(satellite_indices_df)}"
        )
    except Exception as exc:
        print(
            f"[WARNING] Could not read satellite index dataset: {exc}"
        )
else:
    print(
        "[WARNING] Satellite index dataset not found; "
        "quantitative NDVI/NBR evidence will be unavailable."
    )


print(f"OSM records       : {len(osm_df)}")
print(f"Satellite records : {len(satellite_df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

osm_required = {
    "event_id",
    "start_time",
    "duration_hours",
    "detection_count",
    "centroid_lat",
    "centroid_lon",
    "displacement_km",
    "max_frp",
    "event_type",
}

satellite_required = {
    "event_id",
    "satellite_search_status",
    "satellite_observations_found",
    "before_item_id",
    "before_datetime",
    "before_cloud_cover",
    "before_difference_hours",
    "before_selection_quality",
    "after_item_id",
    "after_datetime",
    "after_cloud_cover",
    "after_difference_hours",
    "after_selection_quality",
}


missing_osm = osm_required - set(osm_df.columns)

missing_satellite = (
    satellite_required - set(satellite_df.columns)
)

if missing_osm:
    fail(
        "OSM dataset missing columns:\n"
        + "\n".join(sorted(missing_osm))
    )

if missing_satellite:
    fail(
        "Satellite dataset missing columns:\n"
        + "\n".join(sorted(missing_satellite))
    )

print("[PASS] OSM schema verified")
print("[PASS] Satellite schema verified")


# ============================================================
# CHECK EVENT IDs
# ============================================================

print("\n" + "-" * 75)
print("3. CHECKING EVENT ID CONSISTENCY")
print("-" * 75)

osm_ids = set(
    osm_df["event_id"].astype(str)
)

satellite_ids = set(
    satellite_df["event_id"].astype(str)
)

osm_only = osm_ids - satellite_ids
satellite_only = satellite_ids - osm_ids
common_ids = osm_ids & satellite_ids

print(f"OSM event IDs       : {len(osm_ids)}")
print(f"Satellite event IDs : {len(satellite_ids)}")
print(f"Common event IDs    : {len(common_ids)}")

if osm_only:
    print(
        f"[WARNING] Events missing from satellite dataset: "
        f"{len(osm_only)}"
    )

if satellite_only:
    print(
        f"[WARNING] Events missing from OSM dataset: "
        f"{len(satellite_only)}"
    )

if not osm_only and not satellite_only:
    print("[PASS] All event IDs match")


# ============================================================
# MERGE
# ============================================================

print("\n" + "-" * 75)
print("4. MERGING EVIDENCE")
print("-" * 75)

# Only use satellite-specific columns during merge.
# This avoids duplicated columns from the candidate dataset.

satellite_columns = [
    "event_id",
    "satellite_search_status",
    "satellite_observations_found",
    "before_item_id",
    "before_datetime",
    "before_cloud_cover",
    "before_difference_hours",
    "before_selection_quality",
    "after_item_id",
    "after_datetime",
    "after_cloud_cover",
    "after_difference_hours",
    "after_selection_quality",
]

satellite_subset = satellite_df[
    satellite_columns
].copy()


review_df = osm_df.merge(
    satellite_subset,
    on="event_id",
    how="left",
    validate="one_to_one",
)


# Merge event-level quantitative satellite indices when available.
# These values are generated by calculate_satellite_indices.py and preserve
# the canonical definitions: ΔNDVI = before − after and dNBR = before − after.
if satellite_indices_df is not None and "event_id" in satellite_indices_df.columns:
    index_columns = [
        "event_id",
        "before_ndvi_mean",
        "before_ndvi_median",
        "after_ndvi_mean",
        "after_ndvi_median",
        "delta_ndvi_mean",
        "delta_ndvi_median",
        "delta_ndvi_p90",
        "before_nbr_mean",
        "before_nbr_median",
        "after_nbr_mean",
        "after_nbr_median",
        "dnbr_mean",
        "dnbr_median",
        "dnbr_p90",
        "before_ndbi_mean",
        "before_ndbi_median",
    ]
    available_index_columns = [
        c for c in index_columns if c in satellite_indices_df.columns
    ]
    index_subset = satellite_indices_df[available_index_columns].copy()

    if index_subset["event_id"].duplicated().any():
        fail(
            "Satellite index dataset contains duplicate event_id values; "
            "cannot safely merge quantitative evidence."
        )

    review_df = review_df.merge(
        index_subset,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    print(
        "[PASS] Quantitative satellite indices merged: "
        f"{len(available_index_columns) - 1} metrics"
    )
else:
    print(
        "[WARNING] Satellite index dataset has no event_id; "
        "quantitative metrics were not merged."
    )


print(
    f"[PASS] Merged records: "
    f"{len(review_df)}"
)


# ============================================================
# ADD REVIEW FIELDS
# ============================================================

print("\n" + "-" * 75)
print("5. ADDING REVIEW FIELDS")
print("-" * 75)


# These fields are intentionally blank.
# They are meant to be filled during manual verification.

review_df["verification_label"] = ""

review_df["verification_confidence"] = ""

review_df["verification_reason"] = ""

review_df["source_class"] = ""

review_df["source_class_confidence"] = ""

review_df["source_class_reason"] = ""

review_df["disputed"] = False

review_df["reviewer_notes"] = ""


# ============================================================
# DERIVED SATELLITE STATUS
# ============================================================

def satellite_status(row):

    before = pd.notna(
        row["before_item_id"]
    )

    after = pd.notna(
        row["after_item_id"]
    )

    if before and after:
        return "before_and_after"

    if before:
        return "before_only"

    if after:
        return "after_only"

    return "no_satellite_observation"


review_df["satellite_evidence_status"] = (
    review_df.apply(
        satellite_status,
        axis=1
    )
)


# ============================================================
# DERIVED QUALITY FLAG
# ============================================================

def evidence_quality(row):

    before_quality = (
        row["before_selection_quality"]
    )

    after_quality = (
        row["after_selection_quality"]
    )

    if (
        before_quality == "preferred_cloud"
        and after_quality == "preferred_cloud"
    ):
        return "high"

    if (
        before_quality in {
            "preferred_cloud",
            "acceptable_cloud",
        }
        and after_quality in {
            "preferred_cloud",
            "acceptable_cloud",
        }
    ):
        return "medium"

    if (
        pd.notna(row["before_item_id"])
        or pd.notna(row["after_item_id"])
    ):
        return "low"

    return "none"


review_df["satellite_evidence_quality"] = (
    review_df.apply(
        evidence_quality,
        axis=1
    )
)


# ============================================================
# REORDER COLUMNS
# ============================================================

preferred_order = [
    # Identity
    "event_id",
    "selection_category",

    # FIRMS / event
    "start_time",
    "duration_hours",
    "detection_count",
    "centroid_lat",
    "centroid_lon",
    "displacement_km",
    "max_frp",
    "event_type",

    # OSM
    "osm_checked",
    "osm_radius_m",
    "osm_status",
    "osm_element_count",
    "osm_industrial_count",
    "osm_nearest_industrial_km",
    "osm_nearest_industrial_type",
    "osm_landuse",
    "osm_manmade_types",
    "osm_power_types",
    "osm_place_types",
    "osm_natural_features",
    "osm_waterways",
    "osm_context",
    "osm_error",

    # Satellite
    "satellite_search_status",
    "satellite_observations_found",
    "satellite_evidence_status",
    "satellite_evidence_quality",

    "before_item_id",
    "before_datetime",
    "before_cloud_cover",
    "before_difference_hours",
    "before_selection_quality",

    "after_item_id",
    "after_datetime",
    "after_cloud_cover",
    "after_difference_hours",
    "after_selection_quality",

    # Quantitative satellite indices
    "before_ndvi_mean",
    "before_ndvi_median",
    "after_ndvi_mean",
    "after_ndvi_median",
    "delta_ndvi_mean",
    "delta_ndvi_median",
    "delta_ndvi_p90",
    "before_nbr_mean",
    "before_nbr_median",
    "after_nbr_mean",
    "after_nbr_median",
    "dnbr_mean",
    "dnbr_median",
    "dnbr_p90",

    "before_ndbi_mean",
    "before_ndbi_median",

    # Human verification
    "verification_label",
    "verification_confidence",
    "verification_reason",
    "source_class",
    "source_class_confidence",
    "source_class_reason",
    "disputed",
    "reviewer_notes",
]


# Only reorder columns that actually exist.
existing_preferred = [
    column
    for column in preferred_order
    if column in review_df.columns
]

remaining = [
    column
    for column in review_df.columns
    if column not in existing_preferred
]

review_df = review_df[
    existing_preferred + remaining
]


# ============================================================
# SAVE
# ============================================================

print("\n" + "-" * 75)
print("6. SAVING REPORT")
print("-" * 75)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

review_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("[PASS] Verification report saved:")
print(OUTPUT_FILE)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("VERIFICATION REPORT COMPLETE")
print("=" * 75)

print(
    f"\nTotal candidate events : "
    f"{len(review_df)}"
)

print(
    f"With BEFORE imagery    : "
    f"{review_df['before_item_id'].notna().sum()}"
)

print(
    f"With AFTER imagery     : "
    f"{review_df['after_item_id'].notna().sum()}"
)

print(
    "\nSatellite evidence quality:"
)

print(
    review_df[
        "satellite_evidence_quality"
    ]
    .value_counts()
    .to_string()
)

if "osm_industrial_count" in review_df.columns:

    print(
        "\nOSM industrial features:"
    )

    print(
        review_df[
            "osm_industrial_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


print("\n" + "-" * 75)
print("MANUAL REVIEW FIELDS")
print("-" * 75)

print(
    """
verification_label
verification_confidence
verification_reason
reviewer_notes
"""
)

print(
    "\nThese fields are intentionally blank."
)

print(
    "They will be filled during manual verification."
)

print("\n" + "=" * 75)