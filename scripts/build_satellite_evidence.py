"""
FireDistinguish
---------------
Build Sentinel-2 satellite evidence metadata for verification candidates.

This stage DOES NOT download imagery.

It:
1. Loads the verification candidates (currently 159; not hardcoded).
2. Authenticates against Copernicus Data Space.
3. Searches Sentinel-2 L2A observations around each event.
4. Separates observations into BEFORE and AFTER.
5. Selects the best usable observation on each side.
6. Saves a reproducible evidence index.

Output:
    data/verification/satellite/satellite_evidence_v1.csv
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_candidates_v2_osm.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "satellite_evidence_v1.csv"
)

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# COPERNICUS API
# ============================================================

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "catalog/v1/search"
)


# ============================================================
# SATELLITE SEARCH CONFIGURATION
# ============================================================

SEARCH_DAYS_BEFORE = 30
SEARCH_DAYS_AFTER = 30

# Approximately 1 km x 1 km around event.
BBOX_SIZE = 0.01

# Selection thresholds.
PREFERRED_CLOUD = 30.0
ACCEPTABLE_CLOUD = 50.0

# Maximum observations requested per event.
CATALOG_LIMIT = 100

# Delay between requests.
REQUEST_DELAY_SECONDS = 1.0


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")


# ============================================================
# HELPERS
# ============================================================

def fail(message: str) -> None:
    print(f"\n[ERROR] {message}")
    sys.exit(1)


def authenticate() -> tuple[str, int]:

    print("\n" + "-" * 75)
    print("1. COPERNICUS AUTHENTICATION")
    print("-" * 75)

    if not CLIENT_ID:
        fail("CDSE_CLIENT_ID is missing from .env")

    if not CLIENT_SECRET:
        fail("CDSE_CLIENT_SECRET is missing from .env")

    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    try:

        response = requests.post(
            TOKEN_URL,
            data=payload,
            timeout=30,
        )

        response.raise_for_status()

        token_json = response.json()

        token = token_json.get("access_token")

        if not token:
            fail("Copernicus returned no access token.")

        expires_in = token_json.get("expires_in", 600)

        print("[PASS] Authentication successful")

        return token, expires_in

    except requests.HTTPError:

        print(
            f"[ERROR] HTTP {response.status_code}"
        )

        print(
            response.text[:1000]
        )

        sys.exit(1)

    except Exception as exc:

        fail(
            f"{type(exc).__name__}: {exc}"
        )


def search_sentinel2(
    latitude: float,
    longitude: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> list[dict]:

    bbox = [
        longitude - BBOX_SIZE,
        latitude - BBOX_SIZE,
        longitude + BBOX_SIZE,
        latitude + BBOX_SIZE,
    ]

    payload = {

        "collections": [
            "sentinel-2-l2a"
        ],

        "datetime": (
            f"{start_time.isoformat()}/"
            f"{end_time.isoformat()}"
        ),

        "bbox": bbox,

        "limit": CATALOG_LIMIT,
    }

    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        CATALOG_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code == 401:

        # Token expired mid-run — refresh once and retry this
        # same search before giving up on it.

        print(
            "    [AUTH] 401 received — token expired, "
            "refreshing and retrying..."
        )

        headers["Authorization"] = (
            f"Bearer {get_token(force_refresh=True)}"
        )

        response = requests.post(
            CATALOG_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

    response.raise_for_status()

    data = response.json()

    return data.get("features", [])


def extract_observation(
    feature: dict,
) -> dict:

    properties = feature.get(
        "properties",
        {}
    )

    item_id = feature.get(
        "id",
        ""
    )

    datetime_value = properties.get(
        "datetime"
    )

    if not datetime_value:
        return {}

    observation_time = pd.to_datetime(
        datetime_value,
        utc=True
    )

    cloud_cover = properties.get(
        "eo:cloud_cover"
    )

    try:

        cloud_cover = float(
            cloud_cover
        )

    except (
        TypeError,
        ValueError,
    ):

        cloud_cover = None

    return {

        "satellite_item_id":
            item_id,

        "satellite_datetime":
            observation_time,

        "cloud_cover":
            cloud_cover,
    }


def choose_best(
    observations: list[dict],
    side: str,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> dict | None:
    """Select a clean BEFORE or AFTER observation around the event interval.

    BEFORE observations must occur strictly before event_start.
    AFTER observations must occur strictly after event_end.
    This prevents imagery acquired during a multi-hour/day event from being
    incorrectly treated as post-event evidence.
    """

    if not observations:
        return None

    if side == "before":
        boundary = event_start
        candidates = [
            x for x in observations
            if x["satellite_datetime"] < event_start
        ]
    elif side == "after":
        boundary = event_end
        candidates = [
            x for x in observations
            if x["satellite_datetime"] > event_end
        ]
    else:
        raise ValueError(
            "side must be before or after"
        )

    if not candidates:
        return None

    def with_difference(x: dict) -> dict:
        selected = x.copy()
        selected["difference_hours"] = (
            selected["satellite_datetime"] - boundary
        ).total_seconds() / 3600.0
        return selected

    candidates = [with_difference(x) for x in candidates]

    def distance(x):
        return abs(x["difference_hours"])

    preferred = [
        x for x in candidates
        if (
            x["cloud_cover"] is not None
            and x["cloud_cover"] <= PREFERRED_CLOUD
        )
    ]

    if preferred:
        selected = min(preferred, key=distance)
        selected["selection_quality"] = "preferred_cloud"
        return selected

    acceptable = [
        x for x in candidates
        if (
            x["cloud_cover"] is not None
            and x["cloud_cover"] <= ACCEPTABLE_CLOUD
        )
    ]

    if acceptable:
        selected = min(acceptable, key=distance)
        selected["selection_quality"] = "acceptable_cloud"
        return selected

    selected = min(candidates, key=distance)
    selected["selection_quality"] = "cloudy_fallback"
    return selected


def format_observation(
    observation: dict | None,
) -> dict:

    if observation is None:

        return {

            "item_id": None,

            "datetime": None,

            "cloud_cover": None,

            "difference_hours": None,

            "selection_quality": (
                "no_observation"
            ),
        }

    return {

        "item_id":
            observation["satellite_item_id"],

        "datetime":
            observation["satellite_datetime"].isoformat(),

        "cloud_cover":
            observation["cloud_cover"],

        "difference_hours":
            round(
                observation["difference_hours"],
                2
            ),

        "selection_quality":
            observation["selection_quality"],
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — SATELLITE EVIDENCE BUILDER")
print("=" * 75)

print(
    f"\nInput file : {INPUT_FILE}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)


# ============================================================
# LOAD CANDIDATES
# ============================================================

if not INPUT_FILE.exists():

    fail(
        f"Input file not found:\n{INPUT_FILE}"
    )

try:

    df = pd.read_csv(
        INPUT_FILE
    )

except Exception as exc:

    fail(
        f"Could not read input CSV: {exc}"
    )


required_columns = {

    "event_id",
    "start_time",
    "end_time",
    "centroid_lat",
    "centroid_lon",
}


missing = (
    required_columns
    - set(df.columns)
)

if missing:

    fail(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )


print(
    f"\nCandidates loaded: {len(df)}"
)

print(
    "[PASS] Required columns present."
)


# ============================================================
# PARSE EVENT TIMES
# ============================================================

df["event_start"] = pd.to_datetime(
    df["start_time"],
    utc=True,
    errors="coerce",
)

df["event_end"] = pd.to_datetime(
    df["end_time"],
    utc=True,
    errors="coerce",
)

bad_times = df[
    df["event_start"].isna()
    | df["event_end"].isna()
    | (df["event_end"] < df["event_start"])
]

if len(bad_times) > 0:

    print(
        f"\n[WARNING] "
        f"{len(bad_times)} events have invalid start/end times."
    )


# ============================================================
# AUTHENTICATE
# ============================================================

# CDSE/Keycloak access tokens are short-lived (typically ~600s).
# A catalog search over ~159 events can run long enough to cross
# that boundary, so get_token() refreshes proactively before expiry
# and search_sentinel2() also force-refreshes on an explicit 401.

TOKEN_ISSUED_AT = 0.0
TOKEN_TTL = 600
CURRENT_TOKEN = None


def get_token(force_refresh: bool = False) -> str:

    global CURRENT_TOKEN, TOKEN_ISSUED_AT, TOKEN_TTL

    expired_soon = (
        time.time() - TOKEN_ISSUED_AT
    ) > (TOKEN_TTL - 60)

    if force_refresh or CURRENT_TOKEN is None or expired_soon:

        CURRENT_TOKEN, TOKEN_TTL = authenticate()
        TOKEN_ISSUED_AT = time.time()

    return CURRENT_TOKEN


get_token()


# ============================================================
# PROCESS EVENTS
# ============================================================

results = []

successful_searches = 0
failed_searches = 0


print("\n" + "-" * 75)
print("2. SEARCHING SENTINEL-2")
print("-" * 75)


for index, row in df.iterrows():

    event_id = row["event_id"]

    latitude = float(
        row["centroid_lat"]
    )

    longitude = float(
        row["centroid_lon"]
    )

    event_start = row["event_start"]
    event_end = row["event_end"]

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{event_id}"
    )

    print(
        f"Location: "
        f"{latitude:.6f}, "
        f"{longitude:.6f}"
    )

    print(
        f"Event: "
        f"{event_start.isoformat()} to {event_end.isoformat()}"
    )

    base_record = row.to_dict()

    # --------------------------------------------------------
    # Invalid event time
    # --------------------------------------------------------

    if (
        pd.isna(event_start)
        or pd.isna(event_end)
        or event_end < event_start
    ):

        base_record.update({

            "satellite_search_status":
                "invalid_event_time",

            "satellite_observations_found":
                0,

            "before_item_id":
                None,

            "before_datetime":
                None,

            "before_cloud_cover":
                None,

            "before_difference_hours":
                None,

            "before_selection_quality":
                "no_observation",

            "after_item_id":
                None,

            "after_datetime":
                None,

            "after_cloud_cover":
                None,

            "after_difference_hours":
                None,

            "after_selection_quality":
                "no_observation",
        })

        results.append(
            base_record
        )

        continue

    # --------------------------------------------------------
    # Search ±30 days
    # --------------------------------------------------------

    search_start = (
        event_start
        - timedelta(
            days=SEARCH_DAYS_BEFORE
        )
    )

    search_end = (
        event_end
        + timedelta(
            days=SEARCH_DAYS_AFTER
        )
    )

    try:

        features = search_sentinel2(
            latitude=latitude,
            longitude=longitude,
            start_time=search_start,
            end_time=search_end,
        )

        successful_searches += 1

        print(
            f"Observations found: "
            f"{len(features)}"
        )

    except requests.HTTPError as exc:

        failed_searches += 1

        print(
            "[ERROR] Catalog request failed:"
        )

        print(
            str(exc)
        )

        base_record.update({

            "satellite_search_status":
                "failed",

            "satellite_observations_found":
                0,

            "before_item_id":
                None,

            "before_datetime":
                None,

            "before_cloud_cover":
                None,

            "before_difference_hours":
                None,

            "before_selection_quality":
                "search_failed",

            "after_item_id":
                None,

            "after_datetime":
                None,

            "after_cloud_cover":
                None,

            "after_difference_hours":
                None,

            "after_selection_quality":
                "search_failed",
        })

        results.append(
            base_record
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        continue

    except Exception as exc:

        failed_searches += 1

        print(
            "[ERROR] Unexpected error:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        base_record.update({

            "satellite_search_status":
                "failed",

            "satellite_observations_found":
                0,

            "before_item_id":
                None,

            "before_datetime":
                None,

            "before_cloud_cover":
                None,

            "before_difference_hours":
                None,

            "before_selection_quality":
                "search_failed",

            "after_item_id":
                None,

            "after_datetime":
                None,

            "after_cloud_cover":
                None,

            "after_difference_hours":
                None,

            "after_selection_quality":
                "search_failed",
        })

        results.append(
            base_record
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        continue

    # --------------------------------------------------------
    # Convert catalog features
    # --------------------------------------------------------

    observations = []

    for feature in features:

        observation = extract_observation(
            feature,
        )

        if observation:

            observations.append(
                observation
            )

    # --------------------------------------------------------
    # Select BEFORE / AFTER
    # --------------------------------------------------------

    before = choose_best(
        observations,
        "before",
        event_start,
        event_end,
    )

    after = choose_best(
        observations,
        "after",
        event_start,
        event_end,
    )

    before_data = format_observation(
        before
    )

    after_data = format_observation(
        after
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    base_record.update({

        "satellite_search_status":
            "success",

        "satellite_observations_found":
            len(observations),

        "before_item_id":
            before_data["item_id"],

        "before_datetime":
            before_data["datetime"],

        "before_cloud_cover":
            before_data["cloud_cover"],

        "before_difference_hours":
            before_data["difference_hours"],

        "before_selection_quality":
            before_data[
                "selection_quality"
            ],

        "after_item_id":
            after_data["item_id"],

        "after_datetime":
            after_data["datetime"],

        "after_cloud_cover":
            after_data["cloud_cover"],

        "after_difference_hours":
            after_data["difference_hours"],

        "after_selection_quality":
            after_data[
                "selection_quality"
            ],
    })

    results.append(
        base_record
    )

    # --------------------------------------------------------
    # Print selection
    # --------------------------------------------------------

    if before:

        print(
            "  BEFORE:"
        )

        print(
            f"    {before['satellite_datetime']}"
        )

        print(
            f"    Cloud: "
            f"{before['cloud_cover']}"
        )

        print(
            f"    Δ hours: "
            f"{before['difference_hours']:.2f}"
        )

        print(
            f"    Quality: "
            f"{before['selection_quality']}"
        )

    else:

        print(
            "  BEFORE: NONE"
        )

    if after:

        print(
            "  AFTER:"
        )

        print(
            f"    {after['satellite_datetime']}"
        )

        print(
            f"    Cloud: "
            f"{after['cloud_cover']}"
        )

        print(
            f"    Δ hours: "
            f"{after['difference_hours']:.2f}"
        )

        print(
            f"    Quality: "
            f"{after['selection_quality']}"
        )

    else:

        print(
            "  AFTER: NONE"
        )

    time.sleep(
        REQUEST_DELAY_SECONDS
    )


# ============================================================
# SAVE
# ============================================================

print("\n" + "-" * 75)
print("3. SAVING SATELLITE EVIDENCE")
print("-" * 75)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\n[PASS] Saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# SUMMARY
# ============================================================

before_available = (
    result_df[
        "before_item_id"
    ].notna().sum()
)

after_available = (
    result_df[
        "after_item_id"
    ].notna().sum()
)

preferred_before = (
    result_df[
        "before_selection_quality"
    ]
    .eq("preferred_cloud")
    .sum()
)

preferred_after = (
    result_df[
        "after_selection_quality"
    ]
    .eq("preferred_cloud")
    .sum()
)


print("\n" + "=" * 75)
print("SATELLITE EVIDENCE BUILD COMPLETE")
print("=" * 75)

print(
    f"\nInput events          : {len(df)}"
)

print(
    f"Successful searches   : "
    f"{successful_searches}"
)

print(
    f"Failed searches       : "
    f"{failed_searches}"
)

print(
    f"BEFORE observations   : "
    f"{before_available}/{len(df)}"
)

print(
    f"AFTER observations    : "
    f"{after_available}/{len(df)}"
)

print(
    f"Preferred BEFORE      : "
    f"{preferred_before}"
)

print(
    f"Preferred AFTER       : "
    f"{preferred_after}"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNext step:"
)

print(
    "Review the selected observations before "
    "downloading imagery."
)