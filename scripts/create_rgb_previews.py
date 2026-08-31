"""
FireDistinguish
---------------
Create human-viewable RGB PNG previews from Sentinel-2 GeoTIFF chips.

Expected band order:
    B02 Blue
    B03 Green
    B04 Red
    B08 NIR
    B11 SWIR
    B12 SWIR
    SCL Scene Classification (optional — only present in chips
        downloaded after the SCL band was added to
        download_satellite_chips.py)

Reads:
    data/verification/satellite/chips/*/before.tif
    data/verification/satellite/chips/*/after.tif

Writes:
    before_rgb.png
    after_rgb.png

The original TIFF files are NOT modified.

WHY THIS VERSION LOOKS DIFFERENT FROM BEFORE:
    The previous version stretched each of R/G/B independently between
    its own 2nd and 98th percentile. That maximizes per-channel
    contrast but (a) throws away the true brightness relationship
    between channels, producing washed-out/greyish "hazy" color casts,
    and (b) has no gamma correction, so reflectance values — which are
    physically dim (a bright field might reflect ~20-30% of light) —
    render as dull mid-grey instead of the punchy "true color" look
    you're used to from tools like EO Browser / Sentinel Hub.

    This version instead uses a shared gain + gamma curve across all
    three channels (the same approach Sentinel Hub's own default true
    color script uses), and — when the SCL band is available — masks
    out cloud / cloud-shadow / cirrus / no-data pixels instead of
    silently rendering them as ordinary color, so a reviewer can see
    at a glance how much of the chip is actually usable.
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


# ============================================================
# TRUE-COLOR RENDERING PARAMETERS
# ============================================================

# Gain applied to reflectance (0-1) before gamma correction.
# Sentinel-2 true-color reflectance rarely exceeds ~0.3 for
# non-snow/cloud land surfaces, so a gain of ~3.5 maps that up
# toward the top of the display range instead of leaving it dull.
TRUE_COLOR_GAIN = 3.5

# Gamma correction exponent. Values < 1 applied as a power
# brighten midtones (matches Sentinel Hub's default true-color
# rendering) rather than leaving reflectance visually linear/flat.
TRUE_COLOR_GAMMA = 1 / 1.8

# SCL classes to treat as unusable / not real ground reflectance.
#   0  No data
#   1  Saturated or defective
#   3  Cloud shadow
#   8  Cloud, medium probability
#   9  Cloud, high probability
#  10  Thin cirrus
SCL_MASK_CLASSES = {0, 1, 3, 8, 9, 10}

# Color used to flag masked pixels in the preview (magenta —
# deliberately not a color that occurs naturally in a true-color
# composite, so it can't be mistaken for burned/unburned ground).
MASK_OVERLAY_COLOR = (255, 0, 255)


def enhance_true_color(band):
    """
    Render one reflectance band (0-1 float) to 0-255 uint8 using a
    shared gain + gamma curve, instead of an independent percentile
    stretch. Keeping gain/gamma identical across R/G/B (call this
    the same way for each channel) preserves natural color balance.
    """

    band = band.astype(np.float32)
    band = np.nan_to_num(band, nan=0.0)

    band = np.clip(band, 0.0, None) * TRUE_COLOR_GAIN
    band = np.clip(band, 0.0, 1.0)
    band = band ** TRUE_COLOR_GAMMA

    return (band * 255).astype(np.uint8)


def build_cloud_mask(scl_band):
    """
    Return a boolean array (True = masked/unusable) from an SCL band.
    """

    scl = np.rint(scl_band).astype(np.int16)

    return np.isin(scl, list(SCL_MASK_CLASSES))


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
        # 7 = SCL (optional, only in chips downloaded after this
        #     script's SCL support was added)

        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

        scl = src.read(7) if src.count >= 7 else None

    # Convert each band into a viewable 0–255 range using the same
    # gain/gamma curve so color balance is preserved.
    red = enhance_true_color(red)
    green = enhance_true_color(green)
    blue = enhance_true_color(blue)

    # RGB image requires:
    # Red, Green, Blue
    rgb = np.dstack(
        [red, green, blue]
    )

    clear_percent = None

    if scl is not None:

        mask = build_cloud_mask(scl)

        total_pixels = mask.size
        masked_pixels = int(mask.sum())
        clear_percent = 100.0 * (
            (total_pixels - masked_pixels) / total_pixels
        )

        rgb[mask] = MASK_OVERLAY_COLOR

        print(
            f"    Clear pixels: {clear_percent:.1f}% "
            f"(cloud/shadow/cirrus/no-data flagged in magenta)"
        )

    else:

        print(
            "    [WARN] No SCL band found in this TIFF — chip was "
            "downloaded before SCL support was added. Cloud/haze "
            "pixels cannot be flagged. Re-download this chip to "
            "get masking."
        )

    image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    image.save(output_file)

    print(
        f"    [PASS] Saved: {output_file.name}"
    )

    return clear_percent


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