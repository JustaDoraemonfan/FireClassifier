"""
FireDistinguish
---------------
Select a representative set of candidate events for manual verification.

IMPORTANT:
- This script DOES NOT modify the final event dataset.
- It uses FIRMS/event-level features only.
- OSM, satellite imagery, and weather are NOT used for selection.
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

# Total target = approximately 30 events

N_HIGH_FRP = 5
N_LONG_DURATION = 5
N_HIGH_DETECTIONS = 5
N_HIGH_DISPLACEMENT = 3
N_SINGLE_DETECTION = 4
N_MULTI_DETECTION = 4
N_GEOGRAPHIC = 4


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

print("""
The original 6,894-event dataset was NOT modified.

The new file is only a verification-candidate list.

Next step:
    OSM enrichment of these candidate events.
""")