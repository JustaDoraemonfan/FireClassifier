"""
FireDistinguish
---------------
Generate visual satellite evidence from existing Sentinel-2 TIFF chips.

This script DOES NOT download satellite imagery.

For every event folder containing:
    before.tif
    after.tif

it generates:
    ndvi_change.png
    dnbr.png
    evidence_panel.png

The evidence panel contains:
    BEFORE RGB
    AFTER RGB
    NDVI CHANGE
    dNBR

Expected TIFF band order by default:
    1 = B02 (Blue)
    2 = B03 (Green)
    3 = B04 (Red)
    4 = B08 (NIR)
    5 = B11 (SWIR1)
    6 = B12 (SWIR2)

If your TIFFs use a different band order, change BAND_MAP below.
"""

from pathlib import Path
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt


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


# ------------------------------------------------------------
# Expected Sentinel-2 band order
# ------------------------------------------------------------
#
# Change these numbers ONLY if your TIFFs have a different
# band ordering.
#
# Default:
#
#   Band 1 = B02
#   Band 2 = B03
#   Band 3 = B04
#   Band 4 = B08
#   Band 5 = B11
#   Band 6 = B12
#
# ------------------------------------------------------------

BAND_MAP = {
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B08": 4,
    "B11": 5,
    "B12": 6,
}


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — SATELLITE VISUAL EVIDENCE")
print("=" * 75)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Chips folder : {CHIPS_DIR}")


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not CHIPS_DIR.exists():

    print("\n[ERROR] Chips directory does not exist:")
    print(CHIPS_DIR)

    sys.exit(1)


event_dirs = sorted(
    [
        p
        for p in CHIPS_DIR.iterdir()
        if p.is_dir()
    ]
)


print(
    f"\nEvent folders found: {len(event_dirs)}"
)


if not event_dirs:

    print("\n[ERROR] No event folders found.")

    sys.exit(1)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_tiff(path):
    """
    Read a Sentinel-2 TIFF.

    Returns:
        data   : numpy array, shape (bands, height, width)
        profile: rasterio profile
    """

    with rasterio.open(path) as src:

        data = src.read()

        profile = src.profile.copy()

    return data, profile


def get_band(data, band_name):
    """
    Extract a band using BAND_MAP.
    """

    band_number = BAND_MAP[band_name]

    if band_number > data.shape[0]:

        raise ValueError(
            f"Required {band_name} "
            f"(band {band_number}) not present. "
            f"TIFF contains only {data.shape[0]} bands."
        )

    return data[band_number - 1].astype(np.float32)


def clean_band(band):
    """
    Convert invalid numerical values to NaN.
    """

    band = band.astype(np.float32)

    band[~np.isfinite(band)] = np.nan

    return band


def normalize_rgb(red, green, blue):
    """
    Create a displayable RGB image.

    Sentinel-2 reflectance values may be:
        0–1
    or:
        0–10000

    This function handles both approximately.
    """

    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    valid = np.isfinite(rgb)

    if not np.any(valid):

        return np.zeros_like(rgb)

    max_value = np.nanpercentile(
        rgb[valid],
        98
    )

    min_value = np.nanpercentile(
        rgb[valid],
        2
    )

    if max_value <= min_value:

        return np.zeros_like(rgb)

    rgb = (
        rgb - min_value
    ) / (
        max_value - min_value
    )

    rgb = np.clip(
        rgb,
        0,
        1
    )

    return rgb


def calculate_ndvi(nir, red):

    denominator = nir + red

    ndvi = np.full_like(
        nir,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(nir)
        &
        np.isfinite(red)
        &
        (np.abs(denominator) > 1e-6)
    )

    ndvi[valid] = (
        (nir[valid] - red[valid])
        /
        denominator[valid]
    )

    return ndvi


def calculate_nbr(nir, swir2):

    denominator = nir + swir2

    nbr = np.full_like(
        nir,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(nir)
        &
        np.isfinite(swir2)
        &
        (np.abs(denominator) > 1e-6)
    )

    nbr[valid] = (
        (nir[valid] - swir2[valid])
        /
        denominator[valid]
    )

    return nbr


def save_single_map(
    array,
    output_path,
    title,
    colorbar_label,
    cmap="RdYlGn",
    symmetric=False
):

    plt.figure(
        figsize=(7, 7)
    )

    masked = np.ma.masked_invalid(
        array
    )

    if symmetric:

        finite_values = array[
            np.isfinite(array)
        ]

        if len(finite_values) > 0:

            limit = np.percentile(
                np.abs(finite_values),
                98
            )

            limit = max(
                limit,
                0.01
            )

        else:

            limit = 1.0

        image = plt.imshow(
            masked,
            cmap=cmap,
            vmin=-limit,
            vmax=limit
        )

    else:

        image = plt.imshow(
            masked,
            cmap=cmap
        )

    plt.title(title)

    plt.axis("off")

    cbar = plt.colorbar(
        image,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        colorbar_label
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def percentile_stats(array):

    valid = array[
        np.isfinite(array)
    ]

    if len(valid) == 0:

        return {
            "valid_pixels": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p10": np.nan,
        }

    return {
        "valid_pixels": len(valid),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
        "p10": float(np.percentile(valid, 10)),
    }


# ============================================================
# PROCESS EVENTS
# ============================================================

successful = 0
failed = 0
skipped = 0


for index, event_dir in enumerate(
    event_dirs,
    start=1
):

    event_id = event_dir.name

    print("\n" + "-" * 75)

    print(
        f"[{index}/{len(event_dirs)}] {event_id}"
    )

    before_path = (
        event_dir
        / "before.tif"
    )

    after_path = (
        event_dir
        / "after.tif"
    )


    # --------------------------------------------------------
    # Check TIFFs
    # --------------------------------------------------------

    if not before_path.exists():

        print(
            "[SKIP] before.tif not found"
        )

        skipped += 1

        continue


    if not after_path.exists():

        print(
            "[SKIP] after.tif not found"
        )

        skipped += 1

        continue


    try:

        print(
            "Loading BEFORE TIFF..."
        )

        before, before_profile = read_tiff(
            before_path
        )

        print(
            f"BEFORE bands : {before.shape[0]}"
        )

        print(
            f"BEFORE size  : "
            f"{before.shape[2]} x "
            f"{before.shape[1]}"
        )


        print(
            "Loading AFTER TIFF..."
        )

        after, after_profile = read_tiff(
            after_path
        )

        print(
            f"AFTER bands  : {after.shape[0]}"
        )

        print(
            f"AFTER size   : "
            f"{after.shape[2]} x "
            f"{after.shape[1]}"
        )


        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if before.shape != after.shape:

            raise ValueError(
                "BEFORE and AFTER TIFF dimensions "
                "do not match."
            )


        # ----------------------------------------------------
        # Extract bands
        # ----------------------------------------------------

        before_blue = clean_band(
            get_band(before, "B02")
        )

        before_green = clean_band(
            get_band(before, "B03")
        )

        before_red = clean_band(
            get_band(before, "B04")
        )

        before_nir = clean_band(
            get_band(before, "B08")
        )

        before_swir2 = clean_band(
            get_band(before, "B12")
        )


        after_blue = clean_band(
            get_band(after, "B02")
        )

        after_green = clean_band(
            get_band(after, "B03")
        )

        after_red = clean_band(
            get_band(after, "B04")
        )

        after_nir = clean_band(
            get_band(after, "B08")
        )

        after_swir2 = clean_band(
            get_band(after, "B12")
        )


        # ----------------------------------------------------
        # Create RGB
        # ----------------------------------------------------

        before_rgb = normalize_rgb(
            before_red,
            before_green,
            before_blue
        )

        after_rgb = normalize_rgb(
            after_red,
            after_green,
            after_blue
        )


        # ----------------------------------------------------
        # Calculate NDVI
        # ----------------------------------------------------

        ndvi_before = calculate_ndvi(
            before_nir,
            before_red
        )

        ndvi_after = calculate_ndvi(
            after_nir,
            after_red
        )


        # Convention: positive ΔNDVI means NDVI decreased after the event.
        delta_ndvi = (
            ndvi_before
            - ndvi_after
        )


        # ----------------------------------------------------
        # Calculate NBR
        # ----------------------------------------------------

        nbr_before = calculate_nbr(
            before_nir,
            before_swir2
        )

        nbr_after = calculate_nbr(
            after_nir,
            after_swir2
        )


        # ----------------------------------------------------
        # Calculate dNBR
        # ----------------------------------------------------

        dnbr = (
            nbr_before
            - nbr_after
        )


        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        ndvi_change_path = (
            event_dir
            / "ndvi_change.png"
        )

        dnbr_path = (
            event_dir
            / "dnbr.png"
        )

        evidence_panel_path = (
            event_dir
            / "evidence_panel.png"
        )


        # ----------------------------------------------------
        # Save NDVI change
        # ----------------------------------------------------

        save_single_map(
            delta_ndvi,
            ndvi_change_path,
            "NDVI Change (Before − After)",
            "Δ NDVI",
            cmap="RdYlGn",
            symmetric=True
        )


        # ----------------------------------------------------
        # Save dNBR
        # ----------------------------------------------------

        save_single_map(
            dnbr,
            dnbr_path,
            "dNBR (NBR Before − NBR After)",
            "dNBR",
            cmap="RdYlGn_r",
            symmetric=False
        )


        # ----------------------------------------------------
        # Create 4-panel evidence image
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 10)
        )


        # ----------------------------------------------------
        # BEFORE RGB
        # ----------------------------------------------------

        axes[0, 0].imshow(
            before_rgb
        )

        axes[0, 0].set_title(
            "BEFORE — Sentinel-2 RGB"
        )

        axes[0, 0].axis("off")


        # ----------------------------------------------------
        # AFTER RGB
        # ----------------------------------------------------

        axes[0, 1].imshow(
            after_rgb
        )

        axes[0, 1].set_title(
            "AFTER — Sentinel-2 RGB"
        )

        axes[0, 1].axis("off")


        # ----------------------------------------------------
        # NDVI CHANGE
        # ----------------------------------------------------

        ndvi_masked = np.ma.masked_invalid(
            delta_ndvi
        )

        ndvi_values = delta_ndvi[
            np.isfinite(delta_ndvi)
        ]

        if len(ndvi_values) > 0:

            ndvi_limit = np.percentile(
                np.abs(ndvi_values),
                98
            )

            ndvi_limit = max(
                ndvi_limit,
                0.01
            )

        else:

            ndvi_limit = 1.0


        im3 = axes[1, 0].imshow(
            ndvi_masked,
            cmap="RdYlGn",
            vmin=-ndvi_limit,
            vmax=ndvi_limit
        )

        axes[1, 0].set_title(
            "NDVI Change — Before − After"
        )

        axes[1, 0].axis("off")


        # ----------------------------------------------------
        # dNBR
        # ----------------------------------------------------

        dnbr_masked = np.ma.masked_invalid(
            dnbr
        )

        dnbr_values = dnbr[
            np.isfinite(dnbr)
        ]

        if len(dnbr_values) > 0:

            dnbr_limit = np.percentile(
                np.abs(dnbr_values),
                98
            )

            dnbr_limit = max(
                dnbr_limit,
                0.01
            )

        else:

            dnbr_limit = 1.0


        im4 = axes[1, 1].imshow(
            dnbr_masked,
            cmap="RdYlGn_r",
            vmin=-dnbr_limit,
            vmax=dnbr_limit
        )

        axes[1, 1].set_title(
            "dNBR — NBR Before − NBR After"
        )

        axes[1, 1].axis("off")


        # ----------------------------------------------------
        # Colorbars
        # ----------------------------------------------------

        fig.colorbar(
            im3,
            ax=axes[1, 0],
            fraction=0.046,
            pad=0.04
        )

        fig.colorbar(
            im4,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04
        )


        fig.suptitle(
            f"FireDistinguish — {event_id}",
            fontsize=16
        )


        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96
            ]
        )


        plt.savefig(
            evidence_panel_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        ndvi_stats = percentile_stats(
            delta_ndvi
        )

        dnbr_stats = percentile_stats(
            dnbr
        )


        print(
            "\nSpectral statistics:"
        )

        print(
            f"NDVI change (Before − After) mean : "
            f"{ndvi_stats['mean']:.4f}"
        )

        print(
            f"NDVI change (Before − After) p90  : "
            f"{ndvi_stats['p90']:.4f}"
        )

        print(
            f"dNBR mean        : "
            f"{dnbr_stats['mean']:.4f}"
        )

        print(
            f"dNBR p90         : "
            f"{dnbr_stats['p90']:.4f}"
        )

        print(
            f"Valid pixels     : "
            f"{dnbr_stats['valid_pixels']}"
        )


        print(
            "\n[PASS] Evidence generated"
        )

        print(
            f"       {ndvi_change_path.name}"
        )

        print(
            f"       {dnbr_path.name}"
        )

        print(
            f"       {evidence_panel_path.name}"
        )


        successful += 1


    except Exception as e:

        print(
            f"\n[ERROR] {type(e).__name__}: {e}"
        )

        failed += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SATELLITE VISUAL EVIDENCE COMPLETE")
print("=" * 75)

print(
    f"\nEvent folders       : {len(event_dirs)}"
)

print(
    f"Successfully processed : {successful}"
)

print(
    f"Skipped             : {skipped}"
)

print(
    f"Failed              : {failed}"
)

print(
    "\nGenerated per event:"
)

print(
    "    ndvi_change.png"
)

print(
    "    dnbr.png"
)

print(
    "    evidence_panel.png"
)

print(
    "\nOriginal TIFF files were NOT modified."
)

print("=" * 75)