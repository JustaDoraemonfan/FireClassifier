"""
FireDistinguish
---------------
OSM contextual enrichment for verification candidates.

INPUT:
    data/verification/verification_candidates_v1.csv

OUTPUT:
    data/verification/verification_candidates_v2_osm.csv

IMPORTANT:
- Does NOT modify the original event dataset.
- Does NOT assign fire labels.
- OSM is used only as contextual evidence.

ROBUSTNESS NOTES (why this looks different from a minimal version):
- Rotates across a pool of public Overpass mirrors with exponential
  backoff on 429/504, instead of failing an event on the first
  transient hiccup from a single public instance.
- Saves a checkpoint every CHECKPOINT_EVERY events, and resumes from
  an existing partial output file if the script is interrupted and
  rerun, so a network drop doesn't cost you the whole batch.
- The query itself is scoped to tags that actually help distinguish
  fire types (industrial, landuse, power, man_made, natural,
  waterway, place) instead of also pulling every building and every
  road within the radius, which is both slow/timeout-prone in dense
  areas and adds no distinguishing signal for this task.
"""

from pathlib import Path
import os
import time
import math
import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/verification/verification_candidates_v1.csv"
)

OUTPUT_PATH = Path(
    "data/verification/verification_candidates_v2_osm.csv"
)

# Search radius around each FIRMS event
RADIUS_METERS = 2000

# Rotating pool of public Overpass mirrors. If one is overloaded
# (429/504) or times out, the next attempt tries the next mirror.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Per-event retry settings. MAX_ATTEMPTS can exceed the number of
# mirrors to give one a second chance after the others are tried.
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 6

# Be polite to the public OSM service between *events* (separate
# from the backoff-on-failure above, which only applies within a
# single event's retries).
REQUEST_DELAY_SECONDS = 2.0

# Save partial progress this often, and skip already-completed
# event_ids on rerun instead of starting over.
CHECKPOINT_EVERY = 20

USER_AGENT = (
    "FireDistinguish/1.0 "
    "(research project; OSM contextual enrichment)"
)


# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two coordinates.
    """

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def run_overpass(lat, lon, radius):
    """
    Query OSM/Overpass around an event, rotating across mirrors
    with exponential backoff on 429/504/timeouts.

    Tags requested are scoped to what's actually useful for
    distinguishing fire types:
      - industrial / man_made / power  -> industrial & mining facilities
      - landuse / natural              -> forest, cropland, quarry, etc.
      - waterway / place               -> context, low request cost

    Deliberately NOT requested: blanket "building" and "highway" —
    in any populated part of West Bengal these return thousands of
    elements, which is the single biggest cause of Overpass timeouts
    here, and neither tag actually helps tell a wildfire apart from
    an industrial or agricultural source.

    Raises the last error encountered if every attempt fails.
    """

    query = f"""
    [out:json][timeout:60];

    (
      nwr["industrial"](around:{radius},{lat},{lon});
      nwr["man_made"](around:{radius},{lat},{lon});
      nwr["power"](around:{radius},{lat},{lon});
      nwr["landuse"](around:{radius},{lat},{lon});
      nwr["natural"](around:{radius},{lat},{lon});
      nwr["waterway"](around:{radius},{lat},{lon});
      nwr["place"](around:{radius},{lat},{lon});
    );

    out center tags;
    """

    headers = {
        "User-Agent": USER_AGENT
    }

    last_error = None

    for attempt in range(MAX_ATTEMPTS):

        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]

        try:

            response = requests.post(
                url,
                data=query,
                headers=headers,
                timeout=90
            )

            if response.status_code == 200:
                return response.json().get("elements", [])

            if response.status_code in (429, 504):

                last_error = requests.HTTPError(
                    f"{response.status_code} from {url}"
                )

                print(
                    f"    [{response.status_code}] {url} "
                    f"-> retrying with next mirror"
                )

            else:

                response.raise_for_status()

        except requests.RequestException as e:

            last_error = e

            print(
                f"    [{type(e).__name__}] {url} "
                f"-> retrying with next mirror"
            )

        if attempt < MAX_ATTEMPTS - 1:

            backoff = BACKOFF_BASE_SECONDS * (attempt + 1)

            print(f"    Backing off {backoff}s...")

            time.sleep(backoff)

    raise last_error


def classify_element(element):
    """
    Extract useful OSM contextual tags.
    """

    tags = element.get("tags", {})

    categories = []

    if "industrial" in tags:
        categories.append(
            f"industrial:{tags['industrial']}"
        )

    if "man_made" in tags:
        categories.append(
            f"man_made:{tags['man_made']}"
        )

    if "power" in tags:
        categories.append(
            f"power:{tags['power']}"
        )

    if "landuse" in tags:
        categories.append(
            f"landuse:{tags['landuse']}"
        )

    if "natural" in tags:
        categories.append(
            f"natural:{tags['natural']}"
        )

    if "waterway" in tags:
        categories.append(
            f"waterway:{tags['waterway']}"
        )

    if "place" in tags:
        categories.append(
            f"place:{tags['place']}"
        )

    return categories


def get_element_coordinates(element):
    """
    Extract coordinates from OSM node or center of way/relation.
    """

    if "lat" in element and "lon" in element:
        return (
            element["lat"],
            element["lon"]
        )

    center = element.get("center")

    if center:
        return (
            center.get("lat"),
            center.get("lon")
        )

    return None, None


def process_event(event_id, lat, lon):
    """
    Run the Overpass query for one event and return the enrichment
    fields as a dict (without the original row's columns — those
    are merged in by the caller).
    """

    elements = run_overpass(lat, lon, RADIUS_METERS)

    print(f"OSM elements found: {len(elements)}")

    industrial_distances = []
    industrial_types = []

    manmade_types = []
    power_types = []
    landuses = []
    natural_features = []
    waterways = []
    places = []

    all_context = []

    for element in elements:

        elat, elon = get_element_coordinates(element)

        if elat is None or elon is None:
            continue

        distance = haversine_km(lat, lon, elat, elon)

        tags = element.get("tags", {})

        if "industrial" in tags:
            industrial_distances.append(distance)
            industrial_types.append(tags.get("industrial"))

        if "man_made" in tags:
            manmade_types.append(tags["man_made"])

        if "power" in tags:
            power_types.append(tags["power"])

        if "landuse" in tags:
            landuses.append(tags["landuse"])

        if "natural" in tags:
            natural_features.append(tags["natural"])

        if "waterway" in tags:
            waterways.append(tags["waterway"])

        if "place" in tags:
            places.append(tags["place"])

        all_context.extend(classify_element(element))

    if industrial_distances:

        nearest_industrial_distance = min(industrial_distances)

        nearest_industrial_type = industrial_types[
            industrial_distances.index(nearest_industrial_distance)
        ]

        industrial_count = len(industrial_distances)

    else:

        nearest_industrial_distance = None
        nearest_industrial_type = None
        industrial_count = 0

    return {

        "osm_radius_m": RADIUS_METERS,

        "osm_element_count": len(elements),

        "osm_industrial_count": industrial_count,

        "osm_nearest_industrial_km": nearest_industrial_distance,

        "osm_nearest_industrial_type": nearest_industrial_type,

        "osm_manmade_types":
            "|".join(sorted(set(manmade_types))),

        "osm_power_types":
            "|".join(sorted(set(power_types))),

        "osm_landuse":
            "|".join(sorted(set(landuses))),

        "osm_natural_features":
            "|".join(sorted(set(natural_features))),

        "osm_waterways":
            "|".join(sorted(set(waterways))),

        "osm_place_types":
            "|".join(sorted(set(places))),

        "osm_context":
            "|".join(sorted(set(all_context))),

        "osm_status": "success",

        "osm_error": "",
    }


ERROR_FIELDS = {

    "osm_radius_m": RADIUS_METERS,
    "osm_element_count": None,
    "osm_industrial_count": None,
    "osm_nearest_industrial_km": None,
    "osm_nearest_industrial_type": "",
    "osm_manmade_types": "",
    "osm_power_types": "",
    "osm_landuse": "",
    "osm_natural_features": "",
    "osm_waterways": "",
    "osm_place_types": "",
    "osm_context": "",
    "osm_status": "error",
}


def save_checkpoint(results, path):
    """
    Atomic write: save to a temp file first, then replace the
    target, so a crash mid-write can't corrupt already-completed
    progress.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    pd.DataFrame(results).to_csv(tmp_path, index=False)

    os.replace(tmp_path, path)


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("FIREDISTINGUISH — OSM ENRICHMENT")
print("=" * 75)

if not INPUT_PATH.exists():
    print("\n[ERROR] Candidate file not found:")
    print(INPUT_PATH)
    raise SystemExit(1)

df = pd.read_csv(INPUT_PATH)

print(f"\nCandidates loaded: {len(df)}")

# Resume support: if a partial output already exists, keep whatever
# succeeded last time and only (re)process the rest.
already_done = {}

if OUTPUT_PATH.exists():

    prior = pd.read_csv(OUTPUT_PATH)

    for _, prior_row in prior.iterrows():

        if prior_row.get("osm_status") == "success":
            already_done[prior_row["event_id"]] = prior_row.to_dict()

    print(
        f"Resuming: {len(already_done)} events already "
        f"enriched successfully in a previous run."
    )


# ============================================================
# PROCESS
# ============================================================

results = []

for index, row in df.iterrows():

    event_id = row["event_id"]

    if event_id in already_done:

        results.append(already_done[event_id])

        continue

    lat = float(row["centroid_lat"])
    lon = float(row["centroid_lon"])

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"{event_id}"
    )

    print(
        f"Location: "
        f"{lat:.6f}, {lon:.6f}"
    )

    try:

        enrichment = process_event(event_id, lat, lon)

        result = row.to_dict()

        result.update(enrichment)

        results.append(result)

    except Exception as e:

        print(
            f"[FAILED] {type(e).__name__}: {e}"
        )

        result = row.to_dict()

        result.update(ERROR_FIELDS)

        result["osm_error"] = str(e)

        results.append(result)

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------

    if (
        len(results) % CHECKPOINT_EVERY == 0
        or index == len(df) - 1
    ):

        save_checkpoint(results, OUTPUT_PATH)

        print(f"[CHECKPOINT] Saved {len(results)}/{len(df)}")

    # ----------------------------------------------------------
    # Delay between events (politeness, not backoff-on-failure)
    # ----------------------------------------------------------

    if index < len(df) - 1:

        time.sleep(REQUEST_DELAY_SECONDS)


# ============================================================
# FINAL SAVE + SUMMARY
# ============================================================

save_checkpoint(results, OUTPUT_PATH)

output_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("OSM ENRICHMENT COMPLETE")
print("=" * 75)

print(
    f"""
Input candidates : {len(df)}
Output candidates: {len(output_df)}

Successful OSM queries:
{(output_df["osm_status"] == "success").sum()}

Failed OSM queries:
{(output_df["osm_status"] == "error").sum()}

Events with industrial features:
{(pd.to_numeric(output_df["osm_industrial_count"], errors="coerce").fillna(0) > 0).sum()}

Output:
{OUTPUT_PATH}
"""
)

print("=" * 75)

if (output_df["osm_status"] == "error").any():

    print(
        "\nSome events still failed after mirror rotation + "
        "backoff. Just rerun this same script — it will skip "
        "everything that already succeeded and only retry the "
        "failures."
    )