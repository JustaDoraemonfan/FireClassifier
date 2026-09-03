"""
FireDistinguish
---------------
Calculate Sentinel-2 vegetation and burn indices for verification events.

Input:
    data/verification/satellite/chips/<EVENT_ID>/before.tif
    data/verification/satellite/chips/<EVENT_ID>/after.tif

Expected band order:
    1 = B02 Blue
    2 = B03 Green
    3 = B04 Red
    4 = B08 NIR
    5 = B11 SWIR1
    6 = B12 SWIR2

Calculates:

    NDVI  = (NIR - RED) / (NIR + RED)

    NBR   = (NIR - SWIR2) / (NIR + SWIR2)

    dNBR  = NBR_before - NBR_after

The original TIFF files are NOT modified.

Output:
    data/verification/satellite/satellite_indices_v1.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHIPS_ROOT = (
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
    / "satellite"
    / "satellite_indices_v1.csv"
)


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_divide(numerator, denominator):
    """
    Safely calculate numerator / denominator.

    Pixels where denominator is zero are returned as NaN.
    """

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result


def calculate_ndvi(red, nir):
    """
    NDVI = (NIR - RED) / (NIR + RED)
    """

    return safe_divide(
        nir - red,
        nir + red
    )


def calculate_nbr(nir, swir2):
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    """

    return safe_divide(
        nir - swir2,
        nir + swir2
    )


def calculate_ndbi(swir1, nir):
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

    Built-up index. Uses bands already downloaded for NDVI/NBR
    (B11 SWIR1, B08 NIR) — no extra chip download needed.

    Purpose here: corroborate OSM's industrial-facility tags with
    an independent, pixel-level signal. A candidate with a nearby
    OSM industrial tag AND high NDBI at the centroid is much
    stronger industrial evidence than either alone. Conversely, a
    wildfire_season_region candidate with low NDBI and a genuine
    NDVI/NBR drop has no competing built-up explanation nearby.

    Like NDVI/NBR, this is evidence for a human reviewer to weigh
    alongside OSM context and seasonality — never an automated
    labeling rule on its own.
    """

    return safe_divide(
        swir1 - nir,
        swir1 + nir
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(array):
    """
    Calculate robust summary statistics for an index array.
    """

    valid = array[
        np.isfinite(array)
    ]

    if valid.size == 0:

        return {
            "valid_pixels": 0,
            "valid_percent": 0.0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "valid_pixels": int(valid.size),
        "valid_percent": float(
            100.0
            * valid.size
            / array.size
        ),

        "mean": float(
            np.mean(valid)
        ),

        "median": float(
            np.median(valid)
        ),

        "std": float(
            np.std(valid)
        ),

        "p10": float(
            np.percentile(valid, 10)
        ),

        "p25": float(
            np.percentile(valid, 25)
        ),

        "p75": float(
            np.percentile(valid, 75)
        ),

        "p90": float(
            np.percentile(valid, 90)
        ),

        "min": float(
            np.min(valid)
        ),

        "max": float(
            np.max(valid)
        ),
    }


# ============================================================
# READ SENTINEL-2 CHIP
# ============================================================

def read_chip(path):

    with rasterio.open(path) as src:

        if src.count < 6:

            raise ValueError(
                f"{path.name} contains only "
                f"{src.count} bands; expected 6."
            )

        blue = src.read(1).astype(
            np.float32
        )

        green = src.read(2).astype(
            np.float32
        )

        red = src.read(3).astype(
            np.float32
        )

        nir = src.read(4).astype(
            np.float32
        )

        swir1 = src.read(5).astype(
            np.float32
        )

        swir2 = src.read(6).astype(
            np.float32
        )

        return {
            "blue": blue,
            "green": green,
            "red": red,
            "nir": nir,
            "swir1": swir1,
            "swir2": swir2,
        }


# ============================================================
# PROCESS ONE EVENT
# ============================================================

def process_event(event_dir):

    event_id = event_dir.name

    before_file = (
        event_dir
        / "before.tif"
    )

    after_file = (
        event_dir
        / "after.tif"
    )

    print(
        f"\nProcessing: {event_id}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not before_file.exists():

        print(
            "    [MISSING] before.tif"
        )

        return {
            "event_id": event_id,
            "status": "MISSING_BEFORE",
        }

    if not after_file.exists():

        print(
            "    [MISSING] after.tif"
        )

        return {
            "event_id": event_id,
            "status": "MISSING_AFTER",
        }

    # --------------------------------------------------------
    # Read chips
    # --------------------------------------------------------

    try:

        before = read_chip(
            before_file
        )

        after = read_chip(
            after_file
        )

    except Exception as exc:

        print(
            f"    [ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "event_id": event_id,
            "status": "READ_ERROR",
            "error": str(exc),
        }

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    before_shape = before["red"].shape
    after_shape = after["red"].shape

    print(
        f"    BEFORE size: "
        f"{before_shape[1]} x "
        f"{before_shape[0]}"
    )

    print(
        f"    AFTER size : "
        f"{after_shape[1]} x "
        f"{after_shape[0]}"
    )

    if before_shape != after_shape:

        print(
            "    [ERROR] "
            "BEFORE and AFTER dimensions differ."
        )

        return {
            "event_id": event_id,
            "status": "DIMENSION_MISMATCH",
        }

    # --------------------------------------------------------
    # Calculate BEFORE indices
    # --------------------------------------------------------

    print(
        "    Calculating BEFORE NDVI..."
    )

    before_ndvi = calculate_ndvi(
        before["red"],
        before["nir"]
    )

    print(
        "    Calculating BEFORE NBR..."
    )

    before_nbr = calculate_nbr(
        before["nir"],
        before["swir2"]
    )

    # --------------------------------------------------------
    # Calculate AFTER indices
    # --------------------------------------------------------

    print(
        "    Calculating AFTER NDVI..."
    )

    after_ndvi = calculate_ndvi(
        after["red"],
        after["nir"]
    )

    print(
        "    Calculating AFTER NBR..."
    )

    after_nbr = calculate_nbr(
        after["nir"],
        after["swir2"]
    )

    # --------------------------------------------------------
    # Calculate NDBI (built-up index)
    # --------------------------------------------------------
    #
    # Computed for BEFORE only — NDBI here is used to check whether
    # there's a built-up structure at this location at all, not to
    # track change over time the way NDVI/NBR do. Reuses swir1
    # (B11) already loaded for the chip; no extra download.

    print(
        "    Calculating BEFORE NDBI..."
    )

    before_ndbi = calculate_ndbi(
        before["swir1"],
        before["nir"]
    )

    # --------------------------------------------------------
    # Calculate changes
    # --------------------------------------------------------

    delta_ndvi = (
        before_ndvi
        - after_ndvi
    )

    dnbr = (
        before_nbr
        - after_nbr
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    before_ndvi_stats = (
        calculate_statistics(
            before_ndvi
        )
    )

    after_ndvi_stats = (
        calculate_statistics(
            after_ndvi
        )
    )

    delta_ndvi_stats = (
        calculate_statistics(
            delta_ndvi
        )
    )

    before_nbr_stats = (
        calculate_statistics(
            before_nbr
        )
    )

    after_nbr_stats = (
        calculate_statistics(
            after_nbr
        )
    )

    before_ndbi_stats = (
        calculate_statistics(
            before_ndbi
        )
    )

    dnbr_stats = (
        calculate_statistics(
            dnbr
        )
    )

    # --------------------------------------------------------
    # Quality assessment
    # --------------------------------------------------------

    before_valid = (
        before_nbr_stats[
            "valid_percent"
        ]
    )

    after_valid = (
        after_nbr_stats[
            "valid_percent"
        ]
    )

    paired_valid = np.isfinite(
        before_nbr
    ) & np.isfinite(
        after_nbr
    )

    paired_percent = (
        100.0
        * np.count_nonzero(
            paired_valid
        )
        / paired_valid.size
    )

    if (
        before_valid < 10
        or after_valid < 10
        or paired_percent < 10
    ):

        status = "LOW_VALID_DATA"

    else:

        status = "PASS"

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"    BEFORE NDVI mean : "
        f"{before_ndvi_stats['mean']:.4f}"
    )

    print(
        f"    AFTER NDVI mean  : "
        f"{after_ndvi_stats['mean']:.4f}"
    )

    print(
        f"    Delta NDVI mean  : "
        f"{delta_ndvi_stats['mean']:.4f}"
    )

    print(
        f"    BEFORE NBR mean  : "
        f"{before_nbr_stats['mean']:.4f}"
    )

    print(
        f"    AFTER NBR mean   : "
        f"{after_nbr_stats['mean']:.4f}"
    )

    print(
        f"    dNBR mean        : "
        f"{dnbr_stats['mean']:.4f}"
    )

    print(
        f"    dNBR P90         : "
        f"{dnbr_stats['p90']:.4f}"
    )

    print(
        f"    Paired pixels    : "
        f"{paired_percent:.1f}%"
    )

    print(
        f"    Status           : "
        f"{status}"
    )

    # --------------------------------------------------------
    # Build record
    # --------------------------------------------------------

    return {

        "event_id":
            event_id,

        "status":
            status,

        # ---------------------------------------------
        # NDVI
        # ---------------------------------------------

        "before_ndvi_mean":
            before_ndvi_stats["mean"],

        "before_ndvi_median":
            before_ndvi_stats["median"],

        "before_ndvi_std":
            before_ndvi_stats["std"],

        "before_ndvi_p10":
            before_ndvi_stats["p10"],

        "before_ndvi_p90":
            before_ndvi_stats["p90"],

        "after_ndvi_mean":
            after_ndvi_stats["mean"],

        "after_ndvi_median":
            after_ndvi_stats["median"],

        "after_ndvi_std":
            after_ndvi_stats["std"],

        "after_ndvi_p10":
            after_ndvi_stats["p10"],

        "after_ndvi_p90":
            after_ndvi_stats["p90"],

        "delta_ndvi_mean":
            delta_ndvi_stats["mean"],

        "delta_ndvi_median":
            delta_ndvi_stats["median"],

        "delta_ndvi_p90":
            delta_ndvi_stats["p90"],

        # ---------------------------------------------
        # NBR
        # ---------------------------------------------

        "before_nbr_mean":
            before_nbr_stats["mean"],

        "before_nbr_median":
            before_nbr_stats["median"],

        "before_nbr_std":
            before_nbr_stats["std"],

        "before_nbr_p10":
            before_nbr_stats["p10"],

        "before_nbr_p90":
            before_nbr_stats["p90"],

        "after_nbr_mean":
            after_nbr_stats["mean"],

        "after_nbr_median":
            after_nbr_stats["median"],

        "after_nbr_std":
            after_nbr_stats["std"],

        "after_nbr_p10":
            after_nbr_stats["p10"],

        "after_nbr_p90":
            after_nbr_stats["p90"],

        # ---------------------------------------------
        # dNBR
        # ---------------------------------------------

        "dnbr_mean":
            dnbr_stats["mean"],

        "dnbr_median":
            dnbr_stats["median"],

        "dnbr_std":
            dnbr_stats["std"],

        "dnbr_p10":
            dnbr_stats["p10"],

        "dnbr_p25":
            dnbr_stats["p25"],

        "dnbr_p75":
            dnbr_stats["p75"],

        "dnbr_p90":
            dnbr_stats["p90"],

        "dnbr_min":
            dnbr_stats["min"],

        "dnbr_max":
            dnbr_stats["max"],

        # ---------------------------------------------
        # NDBI (built-up index, BEFORE only)
        # ---------------------------------------------

        "before_ndbi_mean":
            before_ndbi_stats["mean"],

        "before_ndbi_median":
            before_ndbi_stats["median"],

        "before_ndbi_std":
            before_ndbi_stats["std"],

        "before_ndbi_p90":
            before_ndbi_stats["p90"],

        "before_ndbi_valid_percent":
            before_ndbi_stats["valid_percent"],

        # ---------------------------------------------
        # Quality
        # ---------------------------------------------

        "before_ndvi_valid_percent":
            before_ndvi_stats[
                "valid_percent"
            ],

        "after_ndvi_valid_percent":
            after_ndvi_stats[
                "valid_percent"
            ],

        "before_nbr_valid_percent":
            before_nbr_stats[
                "valid_percent"
            ],

        "after_nbr_valid_percent":
            after_nbr_stats[
                "valid_percent"
            ],

        "paired_nbr_valid_percent":
            paired_percent,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "FIREDISTINGUISH — "
        "SATELLITE INDEX CALCULATION"
    )
    print("=" * 75)

    print(
        f"\nInput:"
    )

    print(
        CHIPS_ROOT
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    if not CHIPS_ROOT.exists():

        print(
            "\n[ERROR] Chips directory does not exist."
        )

        sys.exit(1)

    event_directories = sorted(
        [
            p
            for p in CHIPS_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"\nEvents found: "
        f"{len(event_directories)}"
    )

    if not event_directories:

        print(
            "[ERROR] No event directories found."
        )

        sys.exit(1)

    records = []

    for index, event_dir in enumerate(
        event_directories,
        start=1
    ):

        print(
            "\n" + "-" * 75
        )

        print(
            f"[{index}/{len(event_directories)}]"
        )

        record = process_event(
            event_dir
        )

        records.append(
            record
        )

    # ========================================================
    # SAVE
    # ========================================================

    df = pd.DataFrame(
        records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "SATELLITE INDEX CALCULATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nEvents processed : "
        f"{len(df)}"
    )

    if "status" in df.columns:

        print(
            "\nStatus:"
        )

        print(
            df["status"]
            .value_counts()
            .to_string()
        )

    if "dnbr_mean" in df.columns:

        valid_dnbr = df[
            pd.to_numeric(
                df["dnbr_mean"],
                errors="coerce"
            ).notna()
        ]

        print(
            f"\nEvents with valid dNBR:"
            f" {len(valid_dnbr)}"
        )

        if len(valid_dnbr) > 0:

            print(
                "\ndNBR mean range:"
            )

            print(
                f"    Minimum : "
                f"{valid_dnbr['dnbr_mean'].min():.4f}"
            )

            print(
                f"    Maximum : "
                f"{valid_dnbr['dnbr_mean'].max():.4f}"
            )

            print(
                f"    Median  : "
                f"{valid_dnbr['dnbr_mean'].median():.4f}"
            )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "dNBR is evidence of spectral change, "
        "not by itself proof of fire."
    )

    print(
        "We will combine it with FIRMS, OSM, "
        "imagery quality, and manual review."
    )

    print(
        "\n" + "=" * 75
    )


if __name__ == "__main__":
    main()