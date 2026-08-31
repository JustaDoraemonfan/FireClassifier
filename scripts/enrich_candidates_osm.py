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
"""

from pathlib import Path
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

# Overpass endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Be polite to the public OSM service.
REQUEST_DELAY_SECONDS = 2.0

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
    Query OSM/Overpass around an event.
    """

    query = f"""
    [out:json][timeout:60];

    (
      nwr["industrial"](around:{radius},{lat},{lon});
      nwr["landuse"](around:{radius},{lat},{lon});
      nwr["building"](around:{radius},{lat},{lon});
      nwr["highway"](around:{radius},{lat},{lon});
      nwr["place"](around:{radius},{lat},{lon});
      nwr["natural"](around:{radius},{lat},{lon});
      nwr["waterway"](around:{radius},{lat},{lon});
      nwr["amenity"](around:{radius},{lat},{lon});
    );

    out center tags;
    """

    headers = {
        "User-Agent": USER_AGENT
    }

    response = requests.post(
        OVERPASS_URL,
        data=query,
        headers=headers,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


def classify_element(element):
    """
    Extract useful OSM contextual tags.
    """

    tags = element.get("tags", {})

    categories = []

    if "industrial" in tags:
        categories.append("industrial")

    if "landuse" in tags:
        categories.append(
            f"landuse:{tags['landuse']}"
        )

    if "building" in tags:
        categories.append(
            f"building:{tags['building']}"
        )

    if "highway" in tags:
        categories.append(
            f"highway:{tags['highway']}"
        )

    if "place" in tags:
        categories.append(
            f"place:{tags['place']}"
        )

    if "natural" in tags:
        categories.append(
            f"natural:{tags['natural']}"
        )

    if "waterway" in tags:
        categories.append(
            f"waterway:{tags['waterway']}"
        )

    if "amenity" in tags:
        categories.append(
            f"amenity:{tags['amenity']}"
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


# ============================================================
# PROCESS
# ============================================================

results = []

for index, row in df.iterrows():

    event_id = row["event_id"]

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

        data = run_overpass(
            lat,
            lon,
            RADIUS_METERS
        )

        elements = data.get(
            "elements",
            []
        )

        print(
            f"OSM elements found: "
            f"{len(elements)}"
        )

        # ----------------------------------------------------
        # Storage
        # ----------------------------------------------------

        industrial_distances = []
        industrial_types = []

        landuses = []
        buildings = []
        highways = []
        places = []
        natural_features = []
        waterways = []
        amenities = []

        all_context = []

        # ----------------------------------------------------
        # Process OSM elements
        # ----------------------------------------------------

        for element in elements:

            elat, elon = get_element_coordinates(
                element
            )

            if elat is None or elon is None:
                continue

            distance = haversine_km(
                lat,
                lon,
                elat,
                elon
            )

            tags = element.get(
                "tags",
                {}
            )

            # ------------------------------
            # Industrial
            # ------------------------------

            if "industrial" in tags:

                industrial_distances.append(
                    distance
                )

                industrial_types.append(
                    tags.get("industrial")
                )

            # ------------------------------
            # Land use
            # ------------------------------

            if "landuse" in tags:
                landuses.append(
                    tags["landuse"]
                )

            # ------------------------------
            # Buildings
            # ------------------------------

            if "building" in tags:
                buildings.append(
                    tags["building"]
                )

            # ------------------------------
            # Roads
            # ------------------------------

            if "highway" in tags:
                highways.append(
                    tags["highway"]
                )

            # ------------------------------
            # Places
            # ------------------------------

            if "place" in tags:
                places.append(
                    tags["place"]
                )

            # ------------------------------
            # Natural
            # ------------------------------

            if "natural" in tags:
                natural_features.append(
                    tags["natural"]
                )

            # ------------------------------
            # Water
            # ------------------------------

            if "waterway" in tags:
                waterways.append(
                    tags["waterway"]
                )

            # ------------------------------
            # Amenities
            # ------------------------------

            if "amenity" in tags:
                amenities.append(
                    tags["amenity"]
                )

            # ------------------------------
            # General context
            # ------------------------------

            context = classify_element(
                element
            )

            all_context.extend(
                context
            )

        # ----------------------------------------------------
        # Derived values
        # ----------------------------------------------------

        if industrial_distances:

            nearest_industrial_distance = min(
                industrial_distances
            )

            nearest_industrial_type = (
                industrial_types[
                    industrial_distances.index(
                        nearest_industrial_distance
                    )
                ]
            )

            industrial_count = len(
                industrial_distances
            )

        else:

            nearest_industrial_distance = None
            nearest_industrial_type = None
            industrial_count = 0

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = row.to_dict()

        result.update({

            "osm_radius_m": RADIUS_METERS,

            "osm_element_count":
                len(elements),

            "osm_industrial_count":
                industrial_count,

            "osm_nearest_industrial_km":
                nearest_industrial_distance,

            "osm_nearest_industrial_type":
                nearest_industrial_type,

            "osm_landuse":
                "|".join(
                    sorted(
                        set(landuses)
                    )
                ),

            "osm_building_types":
                "|".join(
                    sorted(
                        set(buildings)
                    )
                ),

            "osm_highway_types":
                "|".join(
                    sorted(
                        set(highways)
                    )
                ),

            "osm_place_types":
                "|".join(
                    sorted(
                        set(places)
                    )
                ),

            "osm_natural_features":
                "|".join(
                    sorted(
                        set(natural_features)
                    )
                ),

            "osm_waterways":
                "|".join(
                    sorted(
                        set(waterways)
                    )
                ),

            "osm_amenities":
                "|".join(
                    sorted(
                        set(amenities)
                    )
                ),

            "osm_context":
                "|".join(
                    sorted(
                        set(all_context)
                    )
                ),

            "osm_status":
                "success",

            "osm_error":
                ""

        })

        results.append(result)

    except Exception as e:

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        result = row.to_dict()

        result.update({

            "osm_radius_m": RADIUS_METERS,

            "osm_element_count": None,

            "osm_industrial_count": None,

            "osm_nearest_industrial_km": None,

            "osm_nearest_industrial_type": "",

            "osm_landuse": "",

            "osm_building_types": "",

            "osm_highway_types": "",

            "osm_place_types": "",

            "osm_natural_features": "",

            "osm_waterways": "",

            "osm_amenities": "",

            "osm_context": "",

            "osm_status": "error",

            "osm_error": str(e)

        })

        results.append(result)

    # --------------------------------------------------------
    # Delay
    # --------------------------------------------------------

    if index < len(df) - 1:

        print(
            f"Waiting "
            f"{REQUEST_DELAY_SECONDS}s..."
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )


# ============================================================
# SAVE
# ============================================================

output_df = pd.DataFrame(
    results
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("OSM ENRICHMENT COMPLETE")
print("=" * 75)

print(
    f"""
Input candidates : {len(df)}
Output candidates: {len(output_df)}

Successful OSM queries:
{
    (output_df["osm_status"] == "success").sum()
}

Failed OSM queries:
{
    (output_df["osm_status"] == "error").sum()
}

Events with industrial features:
{
    (output_df["osm_industrial_count"].fillna(0) > 0).sum()
}

Output:
{OUTPUT_PATH}
"""
)

print("=" * 75)