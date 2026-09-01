"""
FireDistinguish
---------------
Select a representative set of candidate events for manual verification.

IMPORTANT:
- This script DOES NOT modify the final event dataset.
- It uses FIRMS/event-level features only (geometry, timing, intensity).
- OSM, satellite imagery, and weather are NOT used for selection.
- The two "wildfire_season_region" / "recurring_location" categories
  below are SAMPLING HEURISTICS ONLY — they widen which candidates get
  human eyes on them. They are never written as a label, never used as
  ground truth, and a human reviewer may assign ANY class to an event
  pulled in by any category. This keeps selection separate from
  labeling per the project's evidence-hierarchy rule.
- Output is a separate verification candidate CSV.

Input:
    data/processed/fire_events_west_bengal_2024_v2.csv

Output:
    data/verification/verification_candidates_v1.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/processed/fire_events_west_bengal_2024_v2.csv"
)

OUTPUT_DIR = Path("data/verification")

OUTPUT_PATH = OUTPUT_DIR / "verification_candidates_v1.csv"

RANDOM_SEED = 42


# ============================================================
# SAMPLE SIZE
# ============================================================

# Total target = approximately 150-165 events.
#
# Rationale: the original ~30-event pilot is too small and too
# intensity/duration-biased to train anything. West Bengal's detection
# volume skews heavily toward persistent industrial-style sources
# (brick kilns, plants), so a purely random or purely "extreme value"
# sample under-represents genuine wildfires. The two new categories
# below (wildfire_season_region, recurring_location) exist to correct
# that imbalance BEFORE evidence-gathering and human review, not to
# decide the label itself.

N_HIGH_FRP = 15
N_LONG_DURATION = 15
N_CHRONIC = 10
N_HIGH_DETECTIONS = 15
N_HIGH_DISPLACEMENT = 12
N_SINGLE_DETECTION = 15
N_MULTI_DETECTION = 15
N_GEOGRAPHIC = 16
N_WILDFIRE_SEASON_REGION = 20
N_RECURRING_LOCATION = 15
N_RANDOM_BASELINE = 15


# ------------------------------------------------------------
# Dry-season / forest-region heuristic (sampling only, not a rule)
# ------------------------------------------------------------
#
# Coarse bounding boxes for West Bengal's forested/hilly tracts,
# used only to widen candidate sampling toward locations and months
# where genuine wildfires are more plausible. This is NOT a
# forest-cover dataset and NOT a labeling rule — OSM/land-cover
# enrichment and human review still decide the actual class.
#
# Box 1: south-western Chhotanagpur plateau forests
#        (Purulia / Bankura / Jhargram belt)
# Box 2: northern hill/dooars forests
#        (Darjeeling / Jalpaiguri / Alipurduar belt)

FOREST_REGION_BOXES = [
    {"lat_min": 22.0, "lat_max": 23.5, "lon_min": 86.0, "lon_max": 87.5},
    {"lat_min": 26.3, "lat_max": 27.2, "lon_min": 88.0, "lon_max": 89.9},
]

# Classic dry-season window for forest fires in this belt.
WILDFIRE_SEASON_MONTHS = [2, 3, 4, 5]

# Grid cell size (degrees) used to detect thermal sources that recur
# across many separately-clustered events at ~the same spot — a
# strong industrial/persistent-source signal, useful for building out
# the industrial side of the verified set.
RECURRING_LOCATION_GRID_DEG = 0.01


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — VERIFICATION CANDIDATE SELECTION")
print("=" * 75)

if not INPUT_PATH.exists():
    print(f"\n[ERROR] Input file not found:")
    print(INPUT_PATH)
    raise SystemExit(1)

df = pd.read_csv(INPUT_PATH)

print(f"\nInput file : {INPUT_PATH}")
print(f"Events     : {len(df):,}")
print(f"Columns    : {len(df.columns)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "event_id",
    "start_time",
    "end_time",
    "duration_hours",
    "detection_count",
    "centroid_lat",
    "centroid_lon",
    "displacement_km",
    "mean_frp",
    "max_frp",
    "frp_variance",
    "mean_brightness_ti4",
    "max_brightness_ti4",
    "mean_brightness_ti5",
    "max_brightness_ti5",
    "event_type",
    "review_flag",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("\n[ERROR] Missing required columns:")
    for col in missing:
        print(f"  - {col}")
    raise SystemExit(1)

print("[PASS] Required columns present.")


# ============================================================
# DATA TYPES
# ============================================================

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df["end_time"] = pd.to_datetime(
    df["end_time"],
    errors="coerce"
)


# ============================================================
# HELPER — ADD CANDIDATES WITHOUT DUPLICATES
# ============================================================

selected = []


def add_candidates(frame, n, category):
    """
    Add up to n events from frame while avoiding duplicate event IDs.
    """

    if len(frame) == 0:
        print(f"\n[WARN] No candidates available for: {category}")
        return

    available = frame[
        ~frame["event_id"].isin(
            [x["event_id"] for x in selected]
        )
    ]

    if len(available) == 0:
        print(f"\n[WARN] No unused candidates available for: {category}")
        return

    take = min(n, len(available))

    sampled = available.head(take)

    for _, row in sampled.iterrows():

        selected.append({
            "event_id": row["event_id"],
            "selection_category": category,
        })

    print(
        f"[SELECT] {category:25s} "
        f"{take} events"
    )


# ============================================================
# 1. HIGH FRP
# ============================================================

print("\n" + "-" * 75)
print("1. HIGH-INTENSITY EVENTS")
print("-" * 75)

high_frp = df.sort_values(
    "max_frp",
    ascending=False
)

add_candidates(
    high_frp,
    N_HIGH_FRP,
    "high_frp"
)


# ============================================================
# 1B. CHRONIC EVENTS (ALL OF THEM)
# ============================================================
#
# Only 10 chronic events exist in the whole dataset. They are the
# strongest available signal for the normal-persistent-industrial
# case, so pull in all of them rather than sampling.

print("\n" + "-" * 75)
print("1B. CHRONIC EVENTS")
print("-" * 75)

chronic = df[df["event_type"] == "chronic"].sort_values(
    "duration_hours",
    ascending=False,
)

add_candidates(
    chronic,
    N_CHRONIC,
    "chronic"
)


# ============================================================
# 2. LONG DURATION
# ============================================================

print("\n" + "-" * 75)
print("2. LONG-DURATION EVENTS")
print("-" * 75)

long_duration = df.sort_values(
    "duration_hours",
    ascending=False
)

add_candidates(
    long_duration,
    N_LONG_DURATION,
    "long_duration"
)


# ============================================================
# 3. MANY DETECTIONS
# ============================================================

print("\n" + "-" * 75)
print("3. HIGH DETECTION-COUNT EVENTS")
print("-" * 75)

high_detections = df.sort_values(
    "detection_count",
    ascending=False
)

add_candidates(
    high_detections,
    N_HIGH_DETECTIONS,
    "high_detection_count"
)


# ============================================================
# 4. HIGH DISPLACEMENT
# ============================================================

print("\n" + "-" * 75)
print("4. HIGH-DISPLACEMENT EVENTS")
print("-" * 75)

high_displacement = df.sort_values(
    "displacement_km",
    ascending=False
)

add_candidates(
    high_displacement,
    N_HIGH_DISPLACEMENT,
    "high_displacement"
)


# ============================================================
# 5. SINGLE DETECTIONS
# ============================================================

print("\n" + "-" * 75)
print("5. SINGLE-DETECTION EVENTS")
print("-" * 75)

single_detection = df[
    df["detection_count"] == 1
].copy()

# Random sample rather than simply taking first rows.
single_detection = single_detection.sample(
    frac=1,
    random_state=RANDOM_SEED
)

add_candidates(
    single_detection,
    N_SINGLE_DETECTION,
    "single_detection"
)


# ============================================================
# 6. MULTI-DETECTION EVENTS
# ============================================================

print("\n" + "-" * 75)
print("6. MULTI-DETECTION EVENTS")
print("-" * 75)

multi_detection = df[
    df["detection_count"] >= 2
].copy()

multi_detection = multi_detection.sample(
    frac=1,
    random_state=RANDOM_SEED
)

add_candidates(
    multi_detection,
    N_MULTI_DETECTION,
    "multi_detection"
)


# ============================================================
# 7. GEOGRAPHICALLY DIVERSE EVENTS
# ============================================================

print("\n" + "-" * 75)
print("7. GEOGRAPHICALLY DIVERSE EVENTS")
print("-" * 75)

# Divide West Bengal into a coarse grid.
#
# We deliberately use only event coordinates here.
# No external geographic information is being used.

geo = df.copy()

geo["lat_bin"] = pd.cut(
    geo["centroid_lat"],
    bins=4,
    labels=False
)

geo["lon_bin"] = pd.cut(
    geo["centroid_lon"],
    bins=4,
    labels=False
)

geo["geo_cell"] = (
    geo["lat_bin"].astype(str)
    + "_"
    + geo["lon_bin"].astype(str)
)

# Shuffle cells for deterministic geographic diversity.
cells = list(
    geo["geo_cell"]
    .dropna()
    .unique()
)

rng = np.random.default_rng(RANDOM_SEED)

rng.shuffle(cells)

geo_selected = []

for cell in cells:

    cell_events = geo[
        geo["geo_cell"] == cell
    ]

    available = cell_events[
        ~cell_events["event_id"].isin(
            [x["event_id"] for x in selected]
        )
    ]

    if len(available) == 0:
        continue

    # Pick the strongest FRP event from each cell.
    chosen = available.sort_values(
        "max_frp",
        ascending=False
    ).iloc[0]

    geo_selected.append(chosen)

    if len(geo_selected) >= N_GEOGRAPHIC:
        break


for row in geo_selected:

    selected.append({
        "event_id": row["event_id"],
        "selection_category": "geographic_diversity",
    })

print(
    f"[SELECT] geographic_diversity       "
    f"{len(geo_selected)} events"
)


# ============================================================
# 8. WILDFIRE-SEASON / FOREST-REGION CANDIDATES
# ============================================================
#
# Sampling heuristic only (see header note). Pulls events that fall
# in the dry-season window AND inside a coarse forest-region bounding
# box. This does NOT assign or imply a label — it only makes sure
# wildfire-plausible events aren't drowned out by industrial repeats
# during human review.

print("\n" + "-" * 75)
print("8. WILDFIRE-SEASON / FOREST-REGION CANDIDATES")
print("-" * 75)

season_mask = df["start_time"].dt.month.isin(WILDFIRE_SEASON_MONTHS)

region_mask = pd.Series(False, index=df.index)

for box in FOREST_REGION_BOXES:
    region_mask |= (
        df["centroid_lat"].between(box["lat_min"], box["lat_max"])
        & df["centroid_lon"].between(box["lon_min"], box["lon_max"])
    )

wildfire_season_region = df[season_mask & region_mask].copy()

# Favor events with some spread/persistence over single flare-ups —
# still just a sort for sampling priority, not a filter rule.
wildfire_season_region = wildfire_season_region.sort_values(
    ["displacement_km", "duration_hours"],
    ascending=False,
)

add_candidates(
    wildfire_season_region,
    N_WILDFIRE_SEASON_REGION,
    "wildfire_season_region"
)


# ============================================================
# 9. RECURRING-LOCATION CANDIDATES
# ============================================================
#
# Sampling heuristic only (see header note). Snaps event centroids to
# a coarse grid and finds cells that produced MANY separately
# clustered events across the year — i.e. a spot that keeps
# reigniting/relighting. That temporal-recurrence pattern is a
# classic persistent-industrial-source signature, useful for growing
# the industrial side of the verified set beyond just the 10 chronic
# events above.

print("\n" + "-" * 75)
print("9. RECURRING-LOCATION CANDIDATES")
print("-" * 75)

recur = df.copy()

recur["grid_lat"] = (
    (recur["centroid_lat"] / RECURRING_LOCATION_GRID_DEG)
    .round()
    * RECURRING_LOCATION_GRID_DEG
)

recur["grid_lon"] = (
    (recur["centroid_lon"] / RECURRING_LOCATION_GRID_DEG)
    .round()
    * RECURRING_LOCATION_GRID_DEG
)

recur["grid_cell"] = (
    recur["grid_lat"].astype(str)
    + "_"
    + recur["grid_lon"].astype(str)
)

cell_counts = (
    recur.groupby("grid_cell")["event_id"]
    .count()
    .sort_values(ascending=False)
)

# Only cells that recurred at least 3 separate times are meaningful.
recurring_cells = cell_counts[cell_counts >= 3].index.tolist()

recurring_selected = []

for cell in recurring_cells:

    cell_events = recur[recur["grid_cell"] == cell]

    available = cell_events[
        ~cell_events["event_id"].isin(
            [x["event_id"] for x in selected]
        )
    ]

    if len(available) == 0:
        continue

    # One representative per recurring cell, to keep this category
    # spatially diverse rather than one site dominating it.
    chosen = available.sort_values(
        "max_frp",
        ascending=False
    ).iloc[0]

    recurring_selected.append(chosen)

    if len(recurring_selected) >= N_RECURRING_LOCATION:
        break

for row in recurring_selected:

    selected.append({
        "event_id": row["event_id"],
        "selection_category": "recurring_location",
    })

print(
    f"[SELECT] recurring_location          "
    f"{len(recurring_selected)} events"
)


# ============================================================
# 10. RANDOM BASELINE
# ============================================================
#
# A small, unweighted random sample with no heuristic at all. This
# guards against every other category's sampling bias by giving
# reviewers some events that aren't "interesting" by any of the
# above measures — necessary for an honest read on how the dataset
# looks in the typical case, and for negative/unknown examples.

print("\n" + "-" * 75)
print("10. RANDOM BASELINE")
print("-" * 75)

random_baseline = df.sample(
    frac=1,
    random_state=RANDOM_SEED + 1,
)

add_candidates(
    random_baseline,
    N_RANDOM_BASELINE,
    "random_baseline"
)


# ============================================================
# BUILD OUTPUT
# ============================================================

selection_df = pd.DataFrame(selected)

# Remove any accidental duplicates.
selection_df = (
    selection_df
    .drop_duplicates(
        subset=["event_id"]
    )
)


# ============================================================
# JOIN ORIGINAL EVENT INFORMATION
# ============================================================

candidate_df = selection_df.merge(
    df,
    on="event_id",
    how="left"
)


# ============================================================
# ADD REVIEW FIELDS
# ============================================================

candidate_df["verification_status"] = "not_reviewed"

candidate_df["verified_label"] = ""

candidate_df["verification_confidence"] = ""

candidate_df["verification_reason"] = ""

candidate_df["osm_checked"] = False

candidate_df["satellite_checked"] = False

candidate_df["weather_checked"] = False


# ============================================================
# SORT
# ============================================================

candidate_df = candidate_df.sort_values(
    [
        "selection_category",
        "start_time",
        "event_id",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

candidate_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SELECTION COMPLETE")
print("=" * 75)

print(f"""
Input events       : {len(df):,}
Selected candidates: {len(candidate_df):,}

Output:
{OUTPUT_PATH}
""")


print("\nSelection categories:")

print(
    candidate_df[
        "selection_category"
    ]
    .value_counts()
    .to_string()
)


print("\nCandidate events:")

print(
    candidate_df[
        [
            "event_id",
            "selection_category",
            "start_time",
            "duration_hours",
            "detection_count",
            "centroid_lat",
            "centroid_lon",
            "displacement_km",
            "max_frp",
            "event_type",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 75)
print("IMPORTANT")
print("=" * 75)

print(f"""
The original {len(df):,}-event dataset was NOT modified.

The new file is only a verification-candidate list.

wildfire_season_region and recurring_location are sampling
heuristics only — they do not assign or imply a label.

Next step:
    OSM enrichment of these candidate events.
""")