"""
FireDistinguish
---------------
Audit downloaded Sentinel-2 GeoTIFF chips.

Checks:
- band count
- image dimensions
- NaN percentage
- zero percentage
- reflectance statistics
- whether the image is effectively black
- whether the image has usable signal

Does NOT modify any imagery.
"""

from pathlib import Path
import sys

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHIPS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "chips"
)


BAND_NAMES = [
    "B02_BLUE",
    "B03_GREEN",
    "B04_RED",
    "B08_NIR",
    "B11_SWIR1",
    "B12_SWIR2",
]


def percentage(value, total):
    if total == 0:
        return 0.0

    return 100.0 * value / total


def audit_file(path):

    print(f"\n    FILE: {path.name}")

    try:

        with rasterio.open(path) as src:

            print(
                f"    Size       : "
                f"{src.width} x {src.height}"
            )

            print(
                f"    Bands      : "
                f"{src.count}"
            )

            print(
                f"    Data type  : "
                f"{src.dtypes[0]}"
            )

            if src.count < 6:

                print(
                    "    [WARNING] "
                    "Expected 6 spectral bands."
                )

            overall_values = []

            for band_index in range(
                1,
                min(src.count, 6) + 1
            ):

                data = src.read(
                    band_index
                ).astype(
                    np.float32
                )

                total = data.size

                finite = np.isfinite(data)

                finite_count = np.count_nonzero(
                    finite
                )

                nan_count = (
                    total
                    - finite_count
                )

                zero_count = np.count_nonzero(
                    data == 0
                )

                valid = data[finite]

                if valid.size == 0:

                    print(
                        f"    {BAND_NAMES[band_index - 1]}"
                        f" : NO VALID PIXELS"
                    )

                    continue

                minimum = float(
                    np.min(valid)
                )

                maximum = float(
                    np.max(valid)
                )

                mean = float(
                    np.mean(valid)
                )

                median = float(
                    np.median(valid)
                )

                p02 = float(
                    np.percentile(valid, 2)
                )

                p98 = float(
                    np.percentile(valid, 98)
                )

                print(
                    f"    "
                    f"{BAND_NAMES[band_index - 1]:10s}"
                    f" | "
                    f"min={minimum:.6f}"
                    f" max={maximum:.6f}"
                    f" mean={mean:.6f}"
                    f" median={median:.6f}"
                    f" p02={p02:.6f}"
                    f" p98={p98:.6f}"
                    f" zero={percentage(zero_count, total):5.1f}%"
                    f" nan={percentage(nan_count, total):5.1f}%"
                )

                overall_values.extend(
                    valid.tolist()
                )

            # ------------------------------------------------
            # OVERALL ASSESSMENT
            # ------------------------------------------------

            if not overall_values:

                print(
                    "\n    [BAD] "
                    "NO VALID DATA"
                )

                return "BAD"

            overall = np.array(
                overall_values,
                dtype=np.float32
            )

            finite_overall = np.isfinite(
                overall
            )

            overall = overall[
                finite_overall
            ]

            near_zero = np.count_nonzero(
                np.abs(overall) < 1e-6
            )

            zero_percentage = percentage(
                near_zero,
                len(overall)
            )

            value_range = (
                float(np.max(overall))
                - float(np.min(overall))
            )

            print(
                f"\n    Overall near-zero pixels:"
                f" {zero_percentage:.2f}%"
            )

            print(
                f"    Overall value range:"
                f" {value_range:.6f}"
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            if zero_percentage > 95:

                print(
                    "\n    [BAD] "
                    "IMAGE IS EFFECTIVELY BLACK"
                )

                print(
                    "    This should NOT be used "
                    "for verification."
                )

                return "BAD"

            elif zero_percentage > 50:

                print(
                    "\n    [WARNING] "
                    "Large amount of zero data"
                )

                return "WARNING"

            elif value_range < 0.01:

                print(
                    "\n    [WARNING] "
                    "Very low dynamic range"
                )

                return "WARNING"

            else:

                print(
                    "\n    [PASS] "
                    "Usable spectral signal detected"
                )

                return "PASS"

    except Exception as exc:

        print(
            f"\n    [ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        return "ERROR"


def main():

    print("=" * 75)
    print("FIREDISTINGUISH — SATELLITE CHIP AUDIT")
    print("=" * 75)

    print(
        f"\nChips directory:"
    )

    print(CHIPS_ROOT)

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

    results = []

    for index, event_dir in enumerate(
        event_directories,
        start=1
    ):

        print(
            "\n" + "-" * 75
        )

        print(
            f"[{index}/{len(event_directories)}] "
            f"{event_dir.name}"
        )

        print(
            "-" * 75
        )

        for label in [
            "before",
            "after"
        ]:

            tif = (
                event_dir
                / f"{label}.tif"
            )

            if not tif.exists():

                print(
                    f"\n    [MISSING] "
                    f"{label}.tif"
                )

                results.append({
                    "event_id": event_dir.name,
                    "image": label,
                    "status": "MISSING",
                })

                continue

            status = audit_file(
                tif
            )

            results.append({
                "event_id": event_dir.name,
                "image": label,
                "status": status,
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)

    for status in [
        "PASS",
        "WARNING",
        "BAD",
        "ERROR",
        "MISSING",
    ]:

        count = sum(
            1
            for result in results
            if result["status"] == status
        )

        print(
            f"{status:10s}: {count}"
        )

    print("\nBy event:")

    for event_dir in event_directories:

        event_results = [
            r
            for r in results
            if r["event_id"]
            == event_dir.name
        ]

        statuses = ", ".join(
            f"{r['image']}={r['status']}"
            for r in event_results
        )

        print(
            f"  {event_dir.name}: "
            f"{statuses}"
        )

    print("\n" + "=" * 75)

    print(
        """
IMPORTANT:

PASS     = usable spectral signal
WARNING  = questionable; inspect carefully
BAD      = should not be used for verification
ERROR    = processing/read problem
MISSING  = imagery was not downloaded

A BAD image does NOT mean there was no fire.
It means the satellite evidence is unusable.
"""
    )

    print("=" * 75)


if __name__ == "__main__":
    main()