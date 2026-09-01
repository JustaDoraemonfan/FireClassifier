from pathlib import Path
import os
import time
import math
import requests
import pandas as pd


INPUT_PATH = Path(
    "data/verification/verification_candidates_v2_osm.csv"
)

OUTPUT_PATH = INPUT_PATH

RADIUS_METERS = 2000

# Rotating pool of public Overpass mirrors. If one is overloaded
# (429/504), the next request tries the next mirror in the list.
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

WAIT_SECONDS = 8

# Per-event retry settings. Each attempt uses the next mirror in
# OVERPASS_URLS (wrapping around), so MAX_ATTEMPTS can exceed the
# number of mirrors to give a mirror a second chance.
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 6

USER_AGENT = (
    "FireDistinguish/1.0 "
    "(research project; OSM contextual enrichment)"
)


def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def get_coords(element):

    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    center = element.get("center")

    if center:
        return center.get("lat"), center.get("lon")

    return None, None


def classify(element):

    tags = element.get("tags", {})
    values = []

    if "industrial" in tags:
        values.append("industrial")

    if "landuse" in tags:
        values.append(f"landuse:{tags['landuse']}")

    if "building" in tags:
        values.append(f"building:{tags['building']}")

    if "highway" in tags:
        values.append(f"highway:{tags['highway']}")

    if "place" in tags:
        values.append(f"place:{tags['place']}")

    if "natural" in tags:
        values.append(f"natural:{tags['natural']}")

    if "waterway" in tags:
        values.append(f"waterway:{tags['waterway']}")

    if "amenity" in tags:
        values.append(f"amenity:{tags['amenity']}")

    return values


def query_osm(lat, lon):
    """
    Query OSM/Overpass around an event, retrying across a rotating
    pool of public mirrors with exponential backoff on 429/504.

    Raises the last error encountered if every attempt fails.
    """

    query = f"""
    [out:json][timeout:90];

    (
      nwr["industrial"](around:{RADIUS_METERS},{lat},{lon});
      nwr["landuse"](around:{RADIUS_METERS},{lat},{lon});
      nwr["building"](around:{RADIUS_METERS},{lat},{lon});
      nwr["highway"](around:{RADIUS_METERS},{lat},{lon});
      nwr["place"](around:{RADIUS_METERS},{lat},{lon});
      nwr["natural"](around:{RADIUS_METERS},{lat},{lon});
      nwr["waterway"](around:{RADIUS_METERS},{lat},{lon});
      nwr["amenity"](around:{RADIUS_METERS},{lat},{lon});
    );

    out center tags;
    """

    last_error = None

    for attempt in range(MAX_ATTEMPTS):

        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]

        try:

            response = requests.post(
                url,
                data=query,
                headers={"User-Agent": USER_AGENT},
                timeout=120
            )

            if response.status_code == 200:
                return response.json()["elements"]

            if response.status_code in (429, 504):

                last_error = requests.HTTPError(
                    f"{response.status_code} from {url}"
                )

                print(
                    f"  [{response.status_code}] {url} "
                    f"-> retrying with next mirror"
                )

            else:

                response.raise_for_status()

        except requests.RequestException as e:

            last_error = e

            print(
                f"  [{type(e).__name__}] {url} "
                f"-> retrying with next mirror"
            )

        if attempt < MAX_ATTEMPTS - 1:

            backoff = BACKOFF_BASE_SECONDS * (attempt + 1)

            print(f"  Backing off {backoff}s...")

            time.sleep(backoff)

    raise last_error


print("=" * 75)
print("FIREDISTINGUISH — RETRY FAILED OSM QUERIES")
print("=" * 75)

if not INPUT_PATH.exists():
    print(f"[ERROR] File not found: {INPUT_PATH}")
    raise SystemExit(1)

df = pd.read_csv(INPUT_PATH)

failed_mask = df["osm_status"] == "error"

failed_indices = list(
    df.index[failed_mask]
)

print(f"\nFailed OSM records: {len(failed_indices)}")

if not failed_indices:
    print("Nothing to retry.")
    raise SystemExit(0)


for counter, idx in enumerate(failed_indices, start=1):

    row = df.loc[idx]

    event_id = row["event_id"]

    lat = float(row["centroid_lat"])
    lon = float(row["centroid_lon"])

    print(
        f"\n[{counter}/{len(failed_indices)}] "
        f"{event_id}"
    )

    print(
        f"Location: {lat:.6f}, {lon:.6f}"
    )

    try:

        elements = query_osm(lat, lon)

        industrial_distances = []
        industrial_types = []

        landuses = []
        buildings = []
        highways = []
        places = []
        natural_features = []
        waterways = []
        amenities = []
        context = []

        for element in elements:

            elat, elon = get_coords(element)

            if elat is None or elon is None:
                continue

            tags = element.get("tags", {})

            distance = haversine_km(
                lat,
                lon,
                elat,
                elon
            )

            if "industrial" in tags:

                industrial_distances.append(
                    distance
                )

                industrial_types.append(
                    tags.get("industrial", "")
                )

            if "landuse" in tags:
                landuses.append(tags["landuse"])

            if "building" in tags:
                buildings.append(tags["building"])

            if "highway" in tags:
                highways.append(tags["highway"])

            if "place" in tags:
                places.append(tags["place"])

            if "natural" in tags:
                natural_features.append(tags["natural"])

            if "waterway" in tags:
                waterways.append(tags["waterway"])

            if "amenity" in tags:
                amenities.append(tags["amenity"])

            context.extend(classify(element))

        if industrial_distances:

            nearest = min(industrial_distances)

            nearest_type = industrial_types[
                industrial_distances.index(nearest)
            ]

        else:

            nearest = None
            nearest_type = ""

        df.at[
            idx,
            "osm_element_count"
        ] = len(elements)

        df.at[
            idx,
            "osm_industrial_count"
        ] = len(industrial_distances)

        df.at[
            idx,
            "osm_nearest_industrial_km"
        ] = nearest

        df.at[
            idx,
            "osm_nearest_industrial_type"
        ] = nearest_type

        df.at[
            idx,
            "osm_landuse"
        ] = "|".join(
            sorted(set(landuses))
        )

        df.at[
            idx,
            "osm_building_types"
        ] = "|".join(
            sorted(set(buildings))
        )

        df.at[
            idx,
            "osm_highway_types"
        ] = "|".join(
            sorted(set(highways))
        )

        df.at[
            idx,
            "osm_place_types"
        ] = "|".join(
            sorted(set(places))
        )

        df.at[
            idx,
            "osm_natural_features"
        ] = "|".join(
            sorted(set(natural_features))
        )

        df.at[
            idx,
            "osm_waterways"
        ] = "|".join(
            sorted(set(waterways))
        )

        df.at[
            idx,
            "osm_amenities"
        ] = "|".join(
            sorted(set(amenities))
        )

        df.at[
            idx,
            "osm_context"
        ] = "|".join(
            sorted(set(context))
        )

        df.at[
            idx,
            "osm_status"
        ] = "success"

        df.at[
            idx,
            "osm_error"
        ] = ""

        print(
            f"[SUCCESS] "
            f"{len(elements)} elements"
        )

    except Exception as e:

        print(
            f"[FAILED AGAIN] "
            f"{type(e).__name__}: {e}"
        )

    if counter < len(failed_indices):

        print(
            f"Waiting {WAIT_SECONDS}s..."
        )

        time.sleep(WAIT_SECONDS)



# ------------------------------------------------------------------
# Atomic write: save to a temp file first, then replace the target.
# Protects the already-recovered rows if the process is interrupted
# mid-write (e.g. killed, disk full) partway through to_csv().
# ------------------------------------------------------------------

tmp_path = OUTPUT_PATH.with_suffix(
    OUTPUT_PATH.suffix + ".tmp"
)

df.to_csv(
    tmp_path,
    index=False
)

os.replace(
    tmp_path,
    OUTPUT_PATH
)


print("\n" + "=" * 75)
print("RETRY COMPLETE")
print("=" * 75)

print(
    f"""
Successful:
{(df["osm_status"] == "success").sum()}

Still failed:
{(df["osm_status"] == "error").sum()}

Events with industrial features:
{(pd.to_numeric(
    df["osm_industrial_count"],
    errors="coerce"
).fillna(0) > 0).sum()}

Saved:
{OUTPUT_PATH}
"""
)