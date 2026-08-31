"""
FireDistinguish
---------------
Download small Sentinel-2 evidence chips for verification candidates.

This script:
1. Loads CDSE credentials from .env
2. Reads the already-selected Sentinel-2 observations
3. Requests small event-centered images
4. Saves BEFORE and AFTER imagery
5. Saves reproducible metadata
6. Is safe to rerun

IMPORTANT:
- Does NOT modify the original FIRMS dataset.
- Does NOT download complete Sentinel-2 scenes.
- Uses small event-centered requests.
"""

import os
import sys
import time
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from pyproj import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_review_v1.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "chips"
)


# CDSE authentication
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)


# Sentinel Hub Process API
PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "api/v1/process"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

# Event-centered area.
#
# 0.01 degrees on each side gives approximately
# 1 km x 1 km around the event.
BBOX_SIZE = 0.005


# Target ground sampling distance for the source evidence grid.
#
# The 10 m grid matches the native resolution of B02/B03/B04/B08.
# B11/B12 are native 20 m and are resampled to this 10 m grid by
# Sentinel Hub. The output dimensions are calculated per event from
# the geographic bounding box, rather than forcing 512 x 512 pixels.
TARGET_RESOLUTION_M = 10.0


# Maximum cloud cover we accept for actual downloads.
#
# The catalog selection already chose good observations,
# but this prevents obviously bad imagery from being downloaded.
MAX_CLOUD_COVER = 50.0


# Seconds between requests.
REQUEST_DELAY = 2.0


# Number of retries for temporary HTTP failures.
MAX_RETRIES = 4


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"\n[ERROR] {message}")
    sys.exit(1)


def safe_filename(value):
    """
    Make a string safe for use as a Windows filename.
    """
    value = str(value)

    invalid = '<>:"/\\|?*'

    for char in invalid:
        value = value.replace(char, "_")

    return value


def get_float(value):
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# VALIDATE
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — SENTINEL-2 CHIP DOWNLOADER")
print("=" * 75)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Input file   : {INPUT_FILE}")
print(f"Output root  : {OUTPUT_ROOT}")

if not ENV_FILE.exists():
    fail(".env file not found.")

if not CLIENT_ID:
    fail("CDSE_CLIENT_ID missing from .env")

if not CLIENT_SECRET:
    fail("CDSE_CLIENT_SECRET missing from .env")

if not INPUT_FILE.exists():
    fail(
        "Verification report not found:\n"
        f"{INPUT_FILE}"
    )

print("\n[PASS] Credentials found")
print("[PASS] Verification report found")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "-" * 75)
print("1. LOADING VERIFICATION DATA")
print("-" * 75)

try:
    df = pd.read_csv(INPUT_FILE)
except Exception as exc:
    fail(f"Could not read verification report: {exc}")


print(f"Events loaded: {len(df)}")


required_columns = [
    "event_id",
    "centroid_lat",
    "centroid_lon",

    "before_item_id",
    "before_datetime",
    "before_cloud_cover",

    "after_item_id",
    "after_datetime",
    "after_cloud_cover",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    fail(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("[PASS] Required columns present")


# ============================================================
# AUTHENTICATION
# ============================================================

print("\n" + "-" * 75)
print("2. COPERNICUS AUTHENTICATION")
print("-" * 75)

token_payload = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}


try:

    response = requests.post(
        TOKEN_URL,
        data=token_payload,
        timeout=30,
    )

    response.raise_for_status()

    token_json = response.json()

    ACCESS_TOKEN = token_json.get(
        "access_token"
    )

    if not ACCESS_TOKEN:
        fail("No access token returned.")

    print("[PASS] Authentication successful")

except requests.HTTPError:

    print(
        f"[ERROR] Authentication failed: "
        f"HTTP {response.status_code}"
    )

    print(response.text[:1000])

    sys.exit(1)

except Exception as exc:

    fail(
        f"{type(exc).__name__}: {exc}"
    )


# ============================================================
# REQUEST HEADERS
# ============================================================

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


# ============================================================
# EVALSCRIPT
# ============================================================

# We request:
#
# B04 = Red
# B03 = Green
# B02 = Blue
#
# B08 = NIR
#
# B11 = SWIR
# B12 = SWIR
#
# SCL = Scene Classification Layer (per-pixel cloud/shadow/cirrus/snow
#       classification). This is requested at native "DN" units
#       (it is a class code, not a reflectance value) alongside the
#       reflectance bands. It lets downstream scripts mask out pixels
#       that are locally cloudy/hazy/shadowed even when the scene-wide
#       cloud cover percentage looks fine — the scene-wide percentage
#       says almost nothing about whether OUR small 1 km chip, centered
#       exactly on a fire, happens to sit under thin cirrus or smoke.
#
# SCL class codes (Sentinel-2 L2A):
#   0  No data            6  Water
#   1  Saturated/defective 7  Unclassified
#   2  Dark area pixels    8  Cloud medium probability
#   3  Cloud shadows       9  Cloud high probability
#   4  Vegetation         10  Thin cirrus
#   5  Not vegetated      11  Snow/ice
#
# The output is FLOAT32 rather than UINT8 so that
# spectral values remain useful for calculating indices later.
#
# Sentinel-2 L2A reflectance values are returned scaled
# appropriately by Sentinel Hub when units are specified.

EVALSCRIPT = """
//VERSION=3

function setup() {

    return {

        input: [
            {
                bands: [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B11",
                    "B12"
                ],
                units: "REFLECTANCE"
            },
            {
                bands: ["SCL"],
                units: "DN"
            }
        ],

        output: {
            bands: 7,
            sampleType: "FLOAT32"
        }
    };
}


function evaluatePixel(sample) {

    return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B08,
        sample.B11,
        sample.B12,
        sample.SCL
    ];
}
"""


# ============================================================
# BUILD PROCESS REQUEST
# ============================================================

def utm_epsg_for_longitude(longitude):
    """Return the northern-hemisphere UTM EPSG code for a longitude."""
    zone = int((longitude + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    return 32600 + zone


def build_projected_bbox(latitude, longitude):
    """
    Transform the event-centered geographic bbox to its local UTM CRS.

    The source event window is still defined in WGS84/CRS84, but the
    Sentinel Hub request is made in a projected metric CRS so that the
    requested output width/height correspond to a physically meaningful
    ground grid instead of degree-based pixels.
    """
    epsg = utm_epsg_for_longitude(longitude)
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    min_lon = longitude - BBOX_SIZE
    min_lat = latitude - BBOX_SIZE
    max_lon = longitude + BBOX_SIZE
    max_lat = latitude + BBOX_SIZE

    corners = [
        transformer.transform(min_lon, min_lat),
        transformer.transform(min_lon, max_lat),
        transformer.transform(max_lon, min_lat),
        transformer.transform(max_lon, max_lat),
    ]

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]

    return [min(xs), min(ys), max(xs), max(ys)], epsg


def estimate_output_dimensions(projected_bbox):
    """
    Estimate output dimensions for approximately TARGET_RESOLUTION_M
    pixels in the projected metric CRS.
    """
    min_x, min_y, max_x, max_y = projected_bbox

    width = max(
        1,
        int(round((max_x - min_x) / TARGET_RESOLUTION_M)),
    )
    height = max(
        1,
        int(round((max_y - min_y) / TARGET_RESOLUTION_M)),
    )

    return width, height


def build_payload(
    latitude,
    longitude,
    start_datetime,
    end_datetime,
    image_width,
    image_height,
):
    """
    Build a Sentinel Hub Process API request.
    """

    bbox, output_epsg = build_projected_bbox(
        latitude,
        longitude,
    )

    payload = {

        "input": {

            "bounds": {

                "bbox": bbox,

                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/EPSG/0/"
                        f"{output_epsg}"
                    )
                },
            },

            "data": [

                {
                    "type": "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {

                            "from": start_datetime,
                            "to": end_datetime,
                        },

                        "maxCloudCoverage": MAX_CLOUD_COVER,

                        "mosaickingOrder": "leastCC",
                    },

                    "processing": {

                        "upsampling": "BILINEAR",

                        "downsampling": "BILINEAR",
                    },
                }
            ],
        },

        "output": {

            "width": image_width,
            "height": image_height,

            "responses": [

                {
                    "identifier": "default",

                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },

        "evalscript": EVALSCRIPT,
    }

    return payload


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_chip(
    event_id,
    label,
    latitude,
    longitude,
    observation_datetime,
    output_file,
):
    """
    Download one event-centered Sentinel-2 chip.

    Returns:
        True  = success
        False = failure
    """

    if output_file.exists():

        print(
            f"    [SKIP] {label} already exists"
        )

        return True


    # Sentinel Hub requires an interval.
    #
    # We use a narrow interval around the selected
    # observation so that the API can select the
    # corresponding Sentinel-2 acquisition.

    observation_datetime = str(
        observation_datetime
    )

    if observation_datetime.endswith("Z"):

        start_datetime = observation_datetime
        end_datetime = observation_datetime

    else:

        start_datetime = observation_datetime
        end_datetime = observation_datetime


    projected_bbox, output_epsg = build_projected_bbox(
        latitude,
        longitude,
    )
    image_width, image_height = estimate_output_dimensions(
        projected_bbox
    )

    payload = build_payload(
        latitude,
        longitude,
        start_datetime,
        end_datetime,
        image_width,
        image_height,
    )


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"    Downloading {label} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            response = requests.post(

                PROCESS_URL,

                headers=headers,

                json=payload,

                timeout=180,
            )

            response.raise_for_status()


            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                output_file,
                "wb"
            ) as file:

                file.write(
                    response.content
                )


            print(
                f"    [PASS] Saved: "
                f"{output_file.name}"
            )

            return True


        except requests.HTTPError as exc:

            status = (
                response.status_code
                if response is not None
                else "unknown"
            )

            print(
                f"    [HTTP ERROR] "
                f"{status}: "
                f"{str(exc)[:300]}"
            )

            if response is not None:

                print(
                    f"    Response: "
                    f"{response.text[:500]}"
                )


        except Exception as exc:

            print(
                f"    [ERROR] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


        if attempt < MAX_RETRIES:

            wait_time = (
                REQUEST_DELAY
                * attempt
                * 2
            )

            print(
                f"    Waiting {wait_time:.1f}s..."
            )

            time.sleep(wait_time)


    print(
        f"    [FAILED] "
        f"{event_id} / {label}"
    )

    return False


# ============================================================
# PROCESS EVENTS
# ============================================================

print("\n" + "-" * 75)
print("3. DOWNLOADING EVENT CHIPS")
print("-" * 75)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


successful = 0
failed = 0
skipped = 0


for index, row in df.iterrows():

    event_id = str(
        row["event_id"]
    )

    latitude = get_float(
        row["centroid_lat"]
    )

    longitude = get_float(
        row["centroid_lon"]
    )

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{event_id}"
    )

    print(
        f"Location: "
        f"{latitude}, {longitude}"
    )


    if latitude is None or longitude is None:

        print(
            "[FAILED] Invalid coordinates"
        )

        failed += 1

        continue


    event_directory = (
        OUTPUT_ROOT
        / safe_filename(event_id)
    )

    event_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    projected_bbox, output_epsg = build_projected_bbox(
        latitude,
        longitude,
    )
    image_width, image_height = estimate_output_dimensions(
        projected_bbox
    )

    # --------------------------------------------------------
    # SAVE EVENT METADATA
    # --------------------------------------------------------

    metadata = {

        "event_id": event_id,

        "latitude": latitude,

        "longitude": longitude,

        "before": {

            "item_id": str(
                row["before_item_id"]
            ),

            "datetime": str(
                row["before_datetime"]
            ),

            "cloud_cover": get_float(
                row["before_cloud_cover"]
            ),
        },

        "after": {

            "item_id": str(
                row["after_item_id"]
            ),

            "datetime": str(
                row["after_datetime"]
            ),

            "cloud_cover": get_float(
                row["after_cloud_cover"]
            ),
        },

        "processing": {

            "bbox_size_degrees": BBOX_SIZE,

            "request_crs": f"EPSG:{output_epsg}",

            "projected_bbox": projected_bbox,

            "target_resolution_m": TARGET_RESOLUTION_M,

            "image_width": image_width,

            "image_height": image_height,

            "grid_note": (
                "Output grid is approximately 10 m in a local UTM "
                "metric CRS. B11/B12 are native 20 m bands resampled "
                "to the 10 m output grid."
            ),

            "bands": [
                "B02",
                "B03",
                "B04",
                "B08",
                "B11",
                "B12",
                "SCL",
            ],

            "native_band_resolution_m": {
                "B02": 10,
                "B03": 10,
                "B04": 10,
                "B08": 10,
                "B11": 20,
                "B12": 20,
                "SCL": 20,
            },

            "scl_note": (
                "Band 7 (SCL) is a per-pixel classification code, not "
                "reflectance. It is used by downstream scripts to mask "
                "cloud / cloud-shadow / cirrus / no-data pixels. See "
                "https://sentiwiki.copernicus.eu (Scene Classification) "
                "for the full class list."
            ),

            "max_cloud_cover": MAX_CLOUD_COVER,

            "resampling": {
                "continuous_reflectance": "BILINEAR",
                "note": (
                    "Sentinel Hub resamples the requested bands onto "
                    "the approximately 10 m UTM output grid; B11/B12 "
                    "remain 20 m native information even when "
                    "represented on the 10 m grid."
                ),
            },
        },
    }


    metadata_file = (
        event_directory
        / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    before_file = (
        event_directory
        / "before.tif"
    )

    before_success = download_chip(

        event_id=event_id,

        label="BEFORE",

        latitude=latitude,

        longitude=longitude,

        observation_datetime=row[
            "before_datetime"
        ],

        output_file=before_file,
    )


    time.sleep(
        REQUEST_DELAY
    )


    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    after_file = (
        event_directory
        / "after.tif"
    )

    after_success = download_chip(

        event_id=event_id,

        label="AFTER",

        latitude=latitude,

        longitude=longitude,

        observation_datetime=row[
            "after_datetime"
        ],

        output_file=after_file,
    )


    if before_success and after_success:

        successful += 1

    else:

        failed += 1


    print(
        f"    Waiting {REQUEST_DELAY}s..."
    )

    time.sleep(
        REQUEST_DELAY
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SATELLITE CHIP DOWNLOAD COMPLETE")
print("=" * 75)

print(
    f"\nEvents processed : {len(df)}"
)

print(
    f"Complete events  : {successful}"
)

print(
    f"Failed events    : {failed}"
)

print(
    f"\nOutput directory:"
)

print(
    OUTPUT_ROOT
)

print("\nEach event contains:")

print(
    """
    metadata.json
    before.tif
    after.tif
    """
)

print(
    f"Target source grid: approximately {TARGET_RESOLUTION_M:g} m"
)

print(
    "\nBands stored in each TIFF:"
)

print(
    """
    B02 — Blue
    B03 — Green
    B04 — Red
    B08 — NIR
    B11 — SWIR
    B12 — SWIR
    SCL — Scene Classification (cloud/shadow/cirrus mask)
    """
)

print(
    "NOTE: existing chips downloaded before this change only have "
    "6 bands. download_chip() skips files that already exist, so "
    "delete or move old event folders under data/verification/satellite/chips/ "
    "if you want them re-downloaded with the SCL band included."
)

print("\n" + "=" * 75)