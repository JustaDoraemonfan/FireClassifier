"""
FireDistinguish
---------------
Create human-viewable RGB PNG previews from Sentinel-2 6-band GeoTIFF chips.

Expected band order:
    B02 Blue
    B03 Green
    B04 Red
    B08 NIR
    B11 SWIR
    B12 SWIR

Reads:
    data/verification/satellite/chips/*/before.tif
    data/verification/satellite/chips/*/after.tif

Writes:
    before_rgb.png
    after_rgb.png

The original TIFF files are NOT modified.
"""

from pathlib import Path
import sys

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHIPS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "chips"
)


def stretch_band(band):
    """Stretch a spectral band to 0–255 for visualisation."""

    band = band.astype(np.float32)

    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros(band.shape, dtype=np.uint8)

    values = band[valid]

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    stretched = (band - low) / (high - low)
    stretched = np.clip(stretched, 0, 1)

    return (stretched * 255).astype(np.uint8)


def create_rgb(input_file, output_file):

    print(f"    Reading: {input_file.name}")

    with rasterio.open(input_file) as src:

        if src.count < 4:
            raise ValueError(
                f"Only {src.count} bands found. "
                "Need at least B02, B03 and B04."
            )

        # Our TIFF band order is:
        #
        # 1 = B02 Blue
        # 2 = B03 Green
        # 3 = B04 Red
        # 4 = B08 NIR
        # 5 = B11 SWIR
        # 6 = B12 SWIR

        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

    # Convert each band into a viewable 0–255 range.
    red = stretch_band(red)
    green = stretch_band(green)
    blue = stretch_band(blue)

    # RGB image requires:
    # Red, Green, Blue
    rgb = np.dstack(
        [red, green, blue]
    )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    image.save(output_file)

    print(
        f"    [PASS] Saved: {output_file.name}"
    )


def main():

    print("=" * 75)
    print("FIREDISTINGUISH — SATELLITE RGB PREVIEW GENERATOR")
    print("=" * 75)

    print(
        f"\nChips directory:"
    )
    print(CHIPS_ROOT)

    if not CHIPS_ROOT.exists():

        print(
            "\n[ERROR] Chips directory does not exist."
        )

        print(
            "Run download_satellite_chips.py first."
        )

        sys.exit(1)

    event_directories = sorted(
        [
            path
            for path in CHIPS_ROOT.iterdir()
            if path.is_dir()
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

    successful = 0
    failed = 0

    for index, event_dir in enumerate(
        event_directories,
        start=1
    ):

        print(
            f"\n[{index}/{len(event_directories)}] "
            f"{event_dir.name}"
        )

        before_tif = (
            event_dir
            / "before.tif"
        )

        after_tif = (
            event_dir
            / "after.tif"
        )

        before_png = (
            event_dir
            / "before_rgb.png"
        )

        after_png = (
            event_dir
            / "after_rgb.png"
        )

        event_success = True

        # ====================================================
        # BEFORE
        # ====================================================

        if before_tif.exists():

            if before_png.exists():

                print(
                    "    [SKIP] before_rgb.png "
                    "already exists"
                )

            else:

                try:

                    create_rgb(
                        before_tif,
                        before_png
                    )

                except Exception as exc:

                    print(
                        f"    [ERROR] BEFORE: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    event_success = False

        else:

            print(
                "    [MISSING] before.tif"
            )

            event_success = False

        # ====================================================
        # AFTER
        # ====================================================

        if after_tif.exists():

            if after_png.exists():

                print(
                    "    [SKIP] after_rgb.png "
                    "already exists"
                )

            else:

                try:

                    create_rgb(
                        after_tif,
                        after_png
                    )

                except Exception as exc:

                    print(
                        f"    [ERROR] AFTER: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    event_success = False

        else:

            print(
                "    [MISSING] after.tif"
            )

            event_success = False

        if event_success:

            successful += 1

        else:

            failed += 1

    print("\n" + "=" * 75)
    print("RGB PREVIEW GENERATION COMPLETE")
    print("=" * 75)

    print(
        f"\nEvents successfully processed: "
        f"{successful}"
    )

    print(
        f"Events with errors/missing data: "
        f"{failed}"
    )

    print(
        "\nGenerated:"
    )

    print(
        "    before_rgb.png"
    )

    print(
        "    after_rgb.png"
    )

    print(
        "\nOriginal .tif files were NOT modified."
    )

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()