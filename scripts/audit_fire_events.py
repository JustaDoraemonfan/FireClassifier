from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    "data/processed/fire_events_west_bengal_2024_v2.csv"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def status(condition, message):
    symbol = "PASS" if condition else "WARN"
    print(f"[{symbol}] {message}")


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING DATASET")

if not DATASET_PATH.exists():
    print(f"[ERROR] Dataset not found:")
    print(DATASET_PATH)
    print()
    print("Check DATASET_PATH at the top of this script.")
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print(f"File       : {DATASET_PATH}")
print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")


# ============================================================
# ACTUAL SCHEMA
# ============================================================

section("1. SCHEMA")

for i, column in enumerate(df.columns, 1):
    print(f"{i:2d}. {column}")


EXPECTED_COLUMNS = [
    "event_id",
    "start_time",
    "end_time",
    "duration_hours",
    "detection_count",
    "centroid_lat",
    "centroid_lon",
    "footprint_wkt",
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

missing_columns = [
    c for c in EXPECTED_COLUMNS
    if c not in df.columns
]

extra_columns = [
    c for c in df.columns
    if c not in EXPECTED_COLUMNS
]

status(
    len(missing_columns) == 0,
    "All expected columns are present."
)

if missing_columns:
    print("\nMissing columns:")
    for c in missing_columns:
        print(f"  - {c}")

if extra_columns:
    print("\nAdditional columns:")
    for c in extra_columns:
        print(f"  - {c}")


# ============================================================
# MISSING VALUES
# ============================================================

section("2. MISSING VALUES")

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("[PASS] No missing values.")
else:
    print("[WARN] Missing values found:")
    for column, count in missing.items():
        percentage = count / len(df) * 100
        print(
            f"  {column:30s} "
            f"{count:6,} "
            f"({percentage:.2f}%)"
        )

print(f"\nTotal missing cells: {df.isna().sum().sum():,}")


# ============================================================
# EVENT IDS
# ============================================================

section("3. EVENT ID VALIDATION")

duplicate_ids = df["event_id"].duplicated().sum()
unique_ids = df["event_id"].nunique()

status(
    duplicate_ids == 0,
    f"Duplicate event IDs: {duplicate_ids:,}"
)

print(f"Unique event IDs: {unique_ids:,}")

# Check expected naming pattern
bad_id_format = ~df["event_id"].astype(str).str.match(
    r"^WB2024_EVT_\d+$"
)

status(
    bad_id_format.sum() == 0,
    f"Event ID format problems: {bad_id_format.sum():,}"
)


# ============================================================
# COORDINATES
# ============================================================

section("4. GEOGRAPHIC VALIDATION")

invalid_lat = (
    (df["centroid_lat"] < -90) |
    (df["centroid_lat"] > 90) |
    df["centroid_lat"].isna()
).sum()

invalid_lon = (
    (df["centroid_lon"] < -180) |
    (df["centroid_lon"] > 180) |
    df["centroid_lon"].isna()
).sum()

status(
    invalid_lat == 0,
    f"Invalid centroid latitude values: {invalid_lat:,}"
)

status(
    invalid_lon == 0,
    f"Invalid centroid longitude values: {invalid_lon:,}"
)

print("\nGeographic extent:")
print(
    f"Latitude : "
    f"{df['centroid_lat'].min():.5f} → "
    f"{df['centroid_lat'].max():.5f}"
)

print(
    f"Longitude: "
    f"{df['centroid_lon'].min():.5f} → "
    f"{df['centroid_lon'].max():.5f}"
)


# ============================================================
# TIME VALIDATION
# ============================================================

section("5. TEMPORAL VALIDATION")

df["start_time"] = pd.to_datetime(
    df["start_time"],
    errors="coerce"
)

df["end_time"] = pd.to_datetime(
    df["end_time"],
    errors="coerce"
)

bad_start = df["start_time"].isna().sum()
bad_end = df["end_time"].isna().sum()

status(
    bad_start == 0,
    f"Invalid start timestamps: {bad_start:,}"
)

status(
    bad_end == 0,
    f"Invalid end timestamps: {bad_end:,}"
)

reversed_events = (
    df["end_time"] < df["start_time"]
).sum()

status(
    reversed_events == 0,
    f"Events where end < start: {reversed_events:,}"
)

print("\nTemporal extent:")
print(f"Start: {df['start_time'].min()}")
print(f"End  : {df['end_time'].max()}")


# ============================================================
# DURATION
# ============================================================

section("6. DURATION VALIDATION")

negative_duration = (
    df["duration_hours"] < 0
).sum()

status(
    negative_duration == 0,
    f"Negative durations: {negative_duration:,}"
)

zero_duration = (
    df["duration_hours"] == 0
).sum()

print(f"Zero-duration events: {zero_duration:,}")

print("\nDuration statistics:")
print(
    df["duration_hours"]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# DETECTION COUNT
# ============================================================

section("7. DETECTION COUNT")

invalid_detection_count = (
    df["detection_count"] <= 0
).sum()

status(
    invalid_detection_count == 0,
    f"Detection counts <= 0: {invalid_detection_count:,}"
)

print("\nDetection count statistics:")
print(
    df["detection_count"]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# FRP
# ============================================================

section("8. FRP VALIDATION")

for column in ["mean_frp", "max_frp", "frp_variance"]:

    negative = (
        df[column] < 0
    ).sum()

    status(
        negative == 0,
        f"{column}: negative values = {negative:,}"
    )

    print(f"\n{column}:")
    print(
        df[column]
        .describe()
        .round(3)
        .to_string()
    )


# ============================================================
# BRIGHTNESS
# ============================================================

section("9. BRIGHTNESS VALIDATION")

brightness_columns = [
    "mean_brightness_ti4",
    "max_brightness_ti4",
    "mean_brightness_ti5",
    "max_brightness_ti5",
]

for column in brightness_columns:

    negative = (
        df[column] < 0
    ).sum()

    status(
        negative == 0,
        f"{column}: negative values = {negative:,}"
    )

    print(
        f"{column}: "
        f"min={df[column].min():.2f}, "
        f"max={df[column].max():.2f}, "
        f"mean={df[column].mean():.2f}"
    )


# ============================================================
# DISPLACEMENT
# ============================================================

section("10. DISPLACEMENT")

negative_displacement = (
    df["displacement_km"] < 0
).sum()

status(
    negative_displacement == 0,
    f"Negative displacement values: {negative_displacement:,}"
)

print(
    df["displacement_km"]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# FOOTPRINT
# ============================================================

section("11. FOOTPRINT WKT")

missing_wkt = df["footprint_wkt"].isna().sum()

status(
    missing_wkt == 0,
    f"Missing footprint WKT: {missing_wkt:,}"
)

print("\nGeometry types:")

geometry_types = (
    df["footprint_wkt"]
    .astype(str)
    .str.extract(r"^([A-Z]+)", expand=False)
    .value_counts()
)

for geometry, count in geometry_types.items():
    print(f"  {geometry:15s}: {count:,}")


# ============================================================
# EVENT TYPES
# ============================================================

section("12. EVENT TYPE DISTRIBUTION")

event_types = (
    df["event_type"]
    .value_counts(dropna=False)
)

for event_type, count in event_types.items():

    percentage = count / len(df) * 100

    print(
        f"{str(event_type):30s} "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# REVIEW FLAGS
# ============================================================

section("13. REVIEW FLAG DISTRIBUTION")

review_counts = (
    df["review_flag"]
    .value_counts(dropna=False)
)

for value, count in review_counts.items():

    percentage = count / len(df) * 100

    print(
        f"{str(value):30s} "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# INTERNAL CONSISTENCY
# ============================================================

section("14. INTERNAL CONSISTENCY CHECKS")

# Duration calculated directly from timestamps
calculated_duration = (
    df["end_time"] - df["start_time"]
).dt.total_seconds() / 3600

duration_difference = (
    calculated_duration - df["duration_hours"]
).abs()

duration_mismatch = (
    duration_difference > 0.01
).sum()

status(
    duration_mismatch == 0,
    f"Duration mismatches (>0.01 h): {duration_mismatch:,}"
)


# Detection count should be integer-like
non_integer_detection = (
    df["detection_count"] % 1 != 0
).sum()

status(
    non_integer_detection == 0,
    f"Non-integer detection counts: {non_integer_detection:,}"
)


# Max FRP should not normally be below mean FRP
frp_order_problem = (
    df["max_frp"] < df["mean_frp"]
).sum()

status(
    frp_order_problem == 0,
    f"max_frp < mean_frp: {frp_order_problem:,}"
)


# Max brightness should not be below mean brightness
ti4_order_problem = (
    df["max_brightness_ti4"]
    < df["mean_brightness_ti4"]
).sum()

ti5_order_problem = (
    df["max_brightness_ti5"]
    < df["mean_brightness_ti5"]
).sum()

status(
    ti4_order_problem == 0,
    f"TI4 max < TI4 mean: {ti4_order_problem:,}"
)

status(
    ti5_order_problem == 0,
    f"TI5 max < TI5 mean: {ti5_order_problem:,}"
)


# ============================================================
# MONTHLY DISTRIBUTION
# ============================================================

section("15. MONTHLY DISTRIBUTION")

monthly = (
    df["start_time"]
    .dt.month
    .value_counts()
    .sort_index()
)

for month, count in monthly.items():

    percentage = count / len(df) * 100

    print(
        f"Month {month:02d}: "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# HOURLY DISTRIBUTION
# ============================================================

section("16. HOUR-OF-DAY DISTRIBUTION")

hourly = (
    df["start_time"]
    .dt.hour
    .value_counts()
    .sort_index()
)

for hour, count in hourly.items():
    print(f"{hour:02d}:00 → {count:,}")


# ============================================================
# BASIC OUTLIERS
# ============================================================

section("17. IQR OUTLIER SCREEN")

numeric_columns = [
    "duration_hours",
    "detection_count",
    "displacement_km",
    "mean_frp",
    "max_frp",
    "frp_variance",
    "mean_brightness_ti4",
    "max_brightness_ti4",
    "mean_brightness_ti5",
    "max_brightness_ti5",
]

for column in numeric_columns:

    series = df[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (series < lower) |
        (series > upper)
    ).sum()

    percentage = outliers / len(series) * 100

    print(
        f"{column:25s} "
        f"{outliers:6,} "
        f"({percentage:5.2f}%)"
    )


# ============================================================
# TOP EVENTS
# ============================================================

section("18. EXTREME EVENTS")

print("\nHighest FRP events:")

top_frp = (
    df.nlargest(10, "max_frp")[
        [
            "event_id",
            "start_time",
            "centroid_lat",
            "centroid_lon",
            "max_frp",
            "duration_hours",
            "detection_count",
        ]
    ]
)

print(top_frp.to_string(index=False))


print("\nLongest events:")

top_duration = (
    df.nlargest(10, "duration_hours")[
        [
            "event_id",
            "start_time",
            "centroid_lat",
            "centroid_lon",
            "duration_hours",
            "detection_count",
            "max_frp",
        ]
    ]
)

print(top_duration.to_string(index=False))


print("\nHighest displacement events:")

top_displacement = (
    df.nlargest(10, "displacement_km")[
        [
            "event_id",
            "start_time",
            "centroid_lat",
            "centroid_lon",
            "displacement_km",
            "duration_hours",
            "detection_count",
        ]
    ]
)

print(top_displacement.to_string(index=False))


# ============================================================
# FINAL SUMMARY
# ============================================================

section("FINAL DATASET SUMMARY")

print(f"""
Rows                  : {len(df):,}
Columns               : {len(df.columns)}
Unique event IDs      : {df['event_id'].nunique():,}
Duplicate event IDs   : {df['event_id'].duplicated().sum():,}

Missing cells         : {df.isna().sum().sum():,}

Time range
----------
Start                 : {df['start_time'].min()}
End                   : {df['end_time'].max()}

Geographic range
----------------
Latitude              : {df['centroid_lat'].min():.5f}
                        → {df['centroid_lat'].max():.5f}

Longitude             : {df['centroid_lon'].min():.5f}
                        → {df['centroid_lon'].max():.5f}

Event types
-----------
{df['event_type'].value_counts().to_string()}

Review flags
------------
{df['review_flag'].value_counts(dropna=False).to_string()}
""")

section("AUDIT COMPLETE")

print("""
No files were modified.
No rows were removed.
No values were changed.

This was a read-only quality audit.
""")