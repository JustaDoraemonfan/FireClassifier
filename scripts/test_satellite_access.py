"""
FireDistinguish
---------------
First Sentinel-2 / Copernicus Data Space access test.

This script:
1. Loads CDSE OAuth credentials from .env
2. Requests an OAuth token
3. Searches Sentinel-2 L2A imagery
4. Uses a real FireDistinguish event
5. Prints available satellite observations

It DOES NOT download imagery yet.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "catalog/v1/search"
)


# ============================================================
# TEST EVENT
# ============================================================

EVENT_ID = "WB2024_EVT_003065"

LATITUDE = 22.476870
LONGITUDE = 87.892820

# Search ±30 days around the FIRMS event.
SEARCH_DAYS = 30

# Small bounding box around event.
# Approximately 1 km x 1 km.
BBOX_SIZE = 0.01


# ============================================================
# VALIDATE CREDENTIALS
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — SENTINEL-2 ACCESS TEST")
print("=" * 75)

print(f"\nProject root : {PROJECT_ROOT}")
print(f".env exists  : {ENV_FILE.exists()}")

if not CLIENT_ID:
    print("\n[ERROR] CDSE_CLIENT_ID is missing from .env")
    sys.exit(1)

if not CLIENT_SECRET:
    print("\n[ERROR] CDSE_CLIENT_SECRET is missing from .env")
    sys.exit(1)

print("[PASS] Client ID loaded")
print("[PASS] Client secret loaded")


# ============================================================
# AUTHENTICATION
# ============================================================

print("\n" + "-" * 75)
print("1. COPERNICUS AUTHENTICATION")
print("-" * 75)

token_data = {
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}

try:

    response = requests.post(
        TOKEN_URL,
        data=token_data,
        timeout=30
    )

    response.raise_for_status()

    token_json = response.json()

    access_token = token_json.get(
        "access_token"
    )

    if not access_token:
        print("[ERROR] No access token returned.")
        print(response.text)
        sys.exit(1)

    print("[PASS] Authentication successful")

    expires_in = token_json.get(
        "expires_in",
        "unknown"
    )

    print(
        f"Token lifetime: {expires_in} seconds"
    )

except requests.HTTPError:

    print("[ERROR] Authentication failed.")

    print(
        f"HTTP status: {response.status_code}"
    )

    # Don't print credentials.
    print(
        "Response:",
        response.text[:1000]
    )

    sys.exit(1)

except Exception as e:

    print(
        f"[ERROR] {type(e).__name__}: {e}"
    )

    sys.exit(1)


# ============================================================
# BUILD SEARCH WINDOW
# ============================================================

event_date = datetime(
    2024,
    4,
    22,
    tzinfo=timezone.utc
)

search_start = (
    event_date - timedelta(
        days=SEARCH_DAYS
    )
)

search_end = (
    event_date + timedelta(
        days=SEARCH_DAYS
    )
)


bbox = [
    LONGITUDE - BBOX_SIZE,
    LATITUDE - BBOX_SIZE,
    LONGITUDE + BBOX_SIZE,
    LATITUDE + BBOX_SIZE,
]


print("\n" + "-" * 75)
print("2. SENTINEL-2 CATALOG SEARCH")
print("-" * 75)

print(
    f"Event ID       : {EVENT_ID}"
)

print(
    f"Event location : "
    f"{LATITUDE:.6f}, {LONGITUDE:.6f}"
)

print(
    f"Search period  : "
    f"{search_start.date()} → "
    f"{search_end.date()}"
)

print(
    f"BBOX           : {bbox}"
)


# ============================================================
# CATALOG REQUEST
# ============================================================

search_payload = {

    "collections": [
        "sentinel-2-l2a"
    ],

    "datetime": (
        f"{search_start.isoformat()}/"
        f"{search_end.isoformat()}"
    ),

    "bbox": bbox,

    "limit": 20
}


headers = {

    "Authorization":
        f"Bearer {access_token}",

    "Content-Type":
        "application/json"
}


try:

    response = requests.post(
        CATALOG_URL,
        headers=headers,
        json=search_payload,
        timeout=60
    )

    response.raise_for_status()

    catalog = response.json()

except requests.HTTPError:

    print("\n[ERROR] Catalog search failed.")

    print(
        f"HTTP status: {response.status_code}"
    )

    print(
        response.text[:2000]
    )

    sys.exit(1)

except Exception as e:

    print(
        f"\n[ERROR] {type(e).__name__}: {e}"
    )

    sys.exit(1)


# ============================================================
# PROCESS RESULTS
# ============================================================

features = catalog.get(
    "features",
    []
)


print("\n[PASS] Catalog request successful")

print(
    f"Observations found: {len(features)}"
)


if not features:

    print(
        "\nNo Sentinel-2 L2A observations "
        "were found in this search window."
    )

    print(
        "This does NOT mean imagery is unavailable "
        "in general; we'll broaden the search."
    )

    sys.exit(0)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "-" * 75)
print("3. AVAILABLE SENTINEL-2 OBSERVATIONS")
print("-" * 75)


for i, feature in enumerate(
    features,
    start=1
):

    properties = feature.get(
        "properties",
        {}
    )

    item_id = feature.get(
        "id",
        "unknown"
    )

    datetime_value = properties.get(
        "datetime",
        "unknown"
    )

    cloud_cover = properties.get(
        "eo:cloud_cover",
        properties.get(
            "cloudCover",
            "unknown"
        )
    )

    print(
        f"\n[{i}]"
    )

    print(
        f"ID           : {item_id}"
    )

    print(
        f"Datetime     : {datetime_value}"
    )

    print(
        f"Cloud cover  : {cloud_cover}"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("SATELLITE ACCESS TEST COMPLETE")
print("=" * 75)

print(
    """
Authentication       : SUCCESS
Catalog API          : SUCCESS
Sentinel-2 search    : SUCCESS

Next step:
Select the best before/after observations and
build the automated 30-event satellite evidence pipeline.
"""
)