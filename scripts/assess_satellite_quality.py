"""
FireDistinguish
---------------
Satellite imagery quality assessment.

Purpose:
    Assess whether Sentinel-2 BEFORE and AFTER TIFF chips are
    suitable for manual burn/change verification.

IMPORTANT:
    - Does NOT modify any TIFF files.
    - Does NOT modify the original event dataset.
    - Does NOT decide whether an event is a fire.
    - Produces a separate quality-assessment CSV.

Outputs:
    data/verification/satellite/satellite_quality_v1.csv

Quality dimensions:
    - file readability
    - valid pixel percentage
    - raw reflectance brightness/darkness
    - potential reflectance clipping
    - raw reflectance contrast
    - BEFORE vs AFTER brightness difference

The resulting quality flag is only a screening aid.
It is NOT a cloud classifier.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError:
    print("[ERROR] rasterio is not installed.")
    print("Install it with:")
    print("    pip install rasterio")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHIPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "chips"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite_quality_v1.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

# Percentage of pixels that must contain usable data.
MIN_VALID_PERCENT = 70.0

# Very dark images are often unusable.
DARK_PERCENT_THRESHOLD = 65.0

# Very bright images may be saturated / cloud dominated.
BRIGHT_PERCENT_THRESHOLD = 65.0

# Excessive saturation can indicate clipped imagery.
CLIPPED_REFLECTANCE_PERCENT_THRESHOLD = 15.0

# Very low contrast can indicate haze/cloud/black imagery.
LOW_CONTRAST_THRESHOLD = 0.03

# Very large brightness difference between BEFORE and AFTER
# is suspicious, although it does not automatically mean bad imagery.
BRIGHTNESS_DIFFERENCE_THRESHOLD = 0.25


# ============================================================
# HELPERS
# ============================================================

def safe_percentile(values, percentile):
    """Return percentile while handling empty arrays."""

    if values.size == 0:
        return np.nan

    return float(
        np.percentile(values, percentile)
    )


def normalize_band(band):
    """
    Normalize a raster band robustly to 0-1 for relative contrast only.

    IMPORTANT:
        This percentile stretch must not be used to infer physical
        brightness or sensor/data clipping.
    """

    band = band.astype(np.float32)

    finite = np.isfinite(band)

    if not finite.any():
        return np.full(
            band.shape,
            np.nan,
            dtype=np.float32
        )

    values = band[finite]

    p2 = np.percentile(values, 2)
    p98 = np.percentile(values, 98)

    if p98 <= p2:
        return np.zeros_like(
            band,
            dtype=np.float32
        )

    normalized = (
        (band - p2)
        / (p98 - p2)
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    normalized[~finite] = np.nan

    return normalized


def inspect_tiff(path):
    """
    Inspect a TIFF and return quality metrics.
    """

    result = {
        "file_exists": path.exists(),
        "readable": False,
        "width": None,
        "height": None,
        "bands": None,
        "valid_percent": np.nan,
        "mean_brightness": np.nan,
        "median_brightness": np.nan,
        "dark_percent": np.nan,
        "bright_percent": np.nan,
        "clipped_reflectance_percent": np.nan,
        "contrast": np.nan,
        "min_value": np.nan,
        "max_value": np.nan,
        "error": ""
    }

    if not path.exists():

        result["error"] = "file_missing"

        return result

    try:

        with rasterio.open(path) as src:

            result["readable"] = True

            result["width"] = src.width
            result["height"] = src.height
            result["bands"] = src.count

            # ------------------------------------------------
            # Read first three bands when available.
            # For RGB chips this corresponds to RGB.
            # ------------------------------------------------

            band_count = min(src.count, 3)

            data = src.read(
                list(range(1, band_count + 1))
            ).astype(np.float32)

            # ------------------------------------------------
            # Determine valid pixels
            # ------------------------------------------------

            valid_mask = np.all(
                np.isfinite(data),
                axis=0
            )

            # Account for nodata if defined.
            if src.nodata is not None:

                valid_mask &= np.all(
                    data != src.nodata,
                    axis=0
                )

            total_pixels = valid_mask.size

            valid_pixels = int(
                valid_mask.sum()
            )

            if total_pixels > 0:

                result["valid_percent"] = (
                    valid_pixels
                    / total_pixels
                    * 100.0
                )

            if valid_pixels == 0:

                result["error"] = (
                    "no_valid_pixels"
                )

                return result

            # ------------------------------------------------
            # Raw reflectance metrics
            #
            # The downloader requests Sentinel-2 REFLECTANCE
            # as FLOAT32. Use the raw values for brightness and
            # clipping metrics. Do NOT use percentile-normalized
            # values for physical interpretation.
            # ------------------------------------------------

            valid_original = data[:, valid_mask]

            # Average the available first-three bands per pixel.
            # These are the RGB-like bands used by this QA script.
            pixel_reflectance = np.mean(
                valid_original,
                axis=0
            )

            result["mean_brightness"] = float(
                np.mean(pixel_reflectance)
            )

            result["median_brightness"] = float(
                np.median(pixel_reflectance)
            )

            # Conservative reflectance sanity thresholds.
            # These are screening heuristics, not cloud detection.
            dark_pixels = (
                pixel_reflectance < 0.02
            )

            result["dark_percent"] = (
                np.mean(dark_pixels)
                * 100.0
            )

            bright_pixels = (
                pixel_reflectance > 0.80
            )

            result["bright_percent"] = (
                np.mean(bright_pixels)
                * 100.0
            )

            # Potential clipping is measured from the raw data,
            # not from a percentile-stretched display.
            clipped_pixels = (
                pixel_reflectance >= 1.0
            )

            result["clipped_reflectance_percent"] = (
                np.mean(clipped_pixels)
                * 100.0
            )

            # Raw-reflectance dynamic range. This is a simple
            # scene-contrast indicator and is NOT a haze/cloud
            # classifier.
            p5 = safe_percentile(
                pixel_reflectance,
                5
            )

            p95 = safe_percentile(
                pixel_reflectance,
                95
            )

            result["contrast"] = (
                p95 - p5
            )

            # ------------------------------------------------
            # Original value range
            # ------------------------------------------------

            valid_original = data[
                :, valid_mask
            ]

            result["min_value"] = float(
                np.min(valid_original)
            )

            result["max_value"] = float(
                np.max(valid_original)
            )

    except Exception as exc:

        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def assess_single_image(metrics):
    """
    Determine whether one image has obvious quality problems.
    """

    issues = []

    if not metrics["file_exists"]:

        issues.append(
            "file missing"
        )

        return issues

    if not metrics["readable"]:

        issues.append(
            "TIFF unreadable"
        )

        return issues

    if metrics["valid_percent"] < MIN_VALID_PERCENT:

        issues.append(
            "low valid-pixel coverage"
        )

    if metrics["dark_percent"] > DARK_PERCENT_THRESHOLD:

        issues.append(
            "predominantly low-reflectance"
        )

    if metrics["bright_percent"] > BRIGHT_PERCENT_THRESHOLD:

        issues.append(
            "predominantly high-reflectance"
        )

    if (
        metrics["clipped_reflectance_percent"]
        > CLIPPED_REFLECTANCE_PERCENT_THRESHOLD
    ):

        issues.append(
            "high potential reflectance clipping"
        )

    if (
        metrics["contrast"]
        < LOW_CONTRAST_THRESHOLD
    ):

        issues.append(
            "very low contrast"
        )

    return issues


def classify_pair(before, after):
    """
    Classify BEFORE/AFTER pair.

    IMPORTANT:
        This is a quality screening label, NOT a fire label.
    """

    before_issues = assess_single_image(
        before
    )

    after_issues = assess_single_image(
        after
    )

    all_issues = []

    for issue in before_issues:

        all_issues.append(
            f"BEFORE: {issue}"
        )

    for issue in after_issues:

        all_issues.append(
            f"AFTER: {issue}"
        )

    # --------------------------------------------------------
    # Missing / unreadable
    # --------------------------------------------------------

    if (
        not before["readable"]
        or not after["readable"]
    ):

        return (
            "POOR",
            "; ".join(all_issues)
        )

    # --------------------------------------------------------
    # Check valid coverage
    # --------------------------------------------------------

    if (
        before["valid_percent"]
        < MIN_VALID_PERCENT
        or
        after["valid_percent"]
        < MIN_VALID_PERCENT
    ):

        return (
            "POOR",
            "; ".join(all_issues)
        )

    # --------------------------------------------------------
    # Strong image-quality problems
    # --------------------------------------------------------

    if len(all_issues) >= 2:

        return (
            "POOR",
            "; ".join(all_issues)
        )

    # --------------------------------------------------------
    # One notable issue
    # --------------------------------------------------------

    if len(all_issues) == 1:

        return (
            "QUESTIONABLE",
            "; ".join(all_issues)
        )

    # --------------------------------------------------------
    # BEFORE / AFTER brightness difference
    # --------------------------------------------------------

    brightness_difference = abs(
        before["mean_brightness"]
        - after["mean_brightness"]
    )

    if (
        brightness_difference
        > BRIGHTNESS_DIFFERENCE_THRESHOLD
    ):

        return (
            "QUESTIONABLE",
            "large BEFORE/AFTER brightness difference"
        )

    # --------------------------------------------------------
    # Good
    # --------------------------------------------------------

    return (
        "GOOD",
        "No obvious image-quality problems detected"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — SATELLITE IMAGE QUALITY ASSESSMENT")
print("=" * 75)

print(
    f"\nProject root : {PROJECT_ROOT}"
)

print(
    f"Chips folder : {CHIPS_DIR}"
)

print(
    f"Output file  : {OUTPUT_FILE}"
)


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not CHIPS_DIR.exists():

    print(
        "\n[ERROR] Satellite chips directory not found:"
    )

    print(CHIPS_DIR)

    sys.exit(1)


# ============================================================
# FIND EVENTS
# ============================================================

event_dirs = sorted(
    [
        path
        for path in CHIPS_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith("WB2024_EVT_")
    ]
)

print(
    f"\nEvents found: {len(event_dirs)}"
)


if not event_dirs:

    print(
        "\n[ERROR] No event directories found."
    )

    sys.exit(1)


# ============================================================
# PROCESS
# ============================================================

records = []

print("\n" + "-" * 75)
print("PROCESSING SATELLITE CHIPS")
print("-" * 75)


for index, event_dir in enumerate(
    event_dirs,
    start=1
):

    event_id = event_dir.name

    before_file = (
        event_dir / "before.tif"
    )

    after_file = (
        event_dir / "after.tif"
    )

    print(
        f"\n[{index}/{len(event_dirs)}] "
        f"{event_id}"
    )

    print(
        f"BEFORE : {before_file.name}"
    )

    before = inspect_tiff(
        before_file
    )

    print(
        f"AFTER  : {after_file.name}"
    )

    after = inspect_tiff(
        after_file
    )

    quality, reason = classify_pair(
        before,
        after
    )

    brightness_difference = np.nan

    if (
        before["readable"]
        and after["readable"]
    ):

        brightness_difference = abs(
            before["mean_brightness"]
            - after["mean_brightness"]
        )

    print(
        f"QUALITY: {quality}"
    )

    print(
        f"REASON : {reason}"
    )

    records.append({

        "event_id": event_id,

        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        "before_readable":
            before["readable"],

        "before_width":
            before["width"],

        "before_height":
            before["height"],

        "before_bands":
            before["bands"],

        "before_valid_percent":
            before["valid_percent"],

        "before_mean_brightness":
            before["mean_brightness"],

        "before_median_brightness":
            before["median_brightness"],

        "before_dark_percent":
            before["dark_percent"],

        "before_bright_percent":
            before["bright_percent"],

        "before_clipped_reflectance_percent":
            before["clipped_reflectance_percent"],

        "before_contrast":
            before["contrast"],

        "before_min_value":
            before["min_value"],

        "before_max_value":
            before["max_value"],

        "before_error":
            before["error"],

        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        "after_readable":
            after["readable"],

        "after_width":
            after["width"],

        "after_height":
            after["height"],

        "after_bands":
            after["bands"],

        "after_valid_percent":
            after["valid_percent"],

        "after_mean_brightness":
            after["mean_brightness"],

        "after_median_brightness":
            after["median_brightness"],

        "after_dark_percent":
            after["dark_percent"],

        "after_bright_percent":
            after["bright_percent"],

        "after_clipped_reflectance_percent":
            after["clipped_reflectance_percent"],

        "after_contrast":
            after["contrast"],

        "after_min_value":
            after["min_value"],

        "after_max_value":
            after["max_value"],

        "after_error":
            after["error"],

        # ----------------------------------------------------
        # PAIR
        # ----------------------------------------------------

        "brightness_difference":
            brightness_difference,

        "imagery_quality":
            quality,

        "quality_reason":
            reason,

        # Explicitly document the interpretation of these metrics.
        "brightness_metric_basis":
            "raw Sentinel-2 reflectance",
        "clipping_metric_basis":
            "raw reflectance >= 1.0; potential clipping only",
        "contrast_metric_basis":
            "raw reflectance p95 - p5",
    })


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    records
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SATELLITE QUALITY ASSESSMENT COMPLETE")
print("=" * 75)

print(
    f"\nEvents assessed : {len(df)}"
)

print(
    f"Output          : {OUTPUT_FILE}"
)

print("\nQuality distribution:")

print(
    df["imagery_quality"]
    .value_counts()
    .to_string()
)

print("\nReadable imagery:")

print(
    f"BEFORE : "
    f"{df['before_readable'].sum()}/{len(df)}"
)

print(
    f"AFTER  : "
    f"{df['after_readable'].sum()}/{len(df)}"
)

print("\nDetailed results:")

print(
    df[
        [
            "event_id",
            "imagery_quality",
            "before_valid_percent",
            "after_valid_percent",
            "before_mean_brightness",
            "after_mean_brightness",
            "brightness_difference",
            "quality_reason",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    """
These quality labels DO NOT determine whether an event
is a fire.

They only indicate whether the satellite imagery appears
suitable for manual interpretation.

Brightness, darkness, and clipping metrics are based on
raw reflectance values. Percentile normalization is not
used for physical brightness or clipping decisions.

This script is NOT a cloud or haze classifier. Scene-level
brightness/contrast flags are screening signals only and
must be interpreted with satellite metadata and the actual
event chip.

GOOD:
    Suitable for manual comparison.

QUESTIONABLE:
    Potential issue; inspect manually.

POOR:
    Strong quality problem; satellite evidence should
    be treated cautiously or marked inconclusive.

Next step:
    Use these quality results when manually reviewing
    the 30 verification candidates.
"""
)