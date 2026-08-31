from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "data/verification/verification_candidates_v2_osm.csv"
)

OUTPUT_PATH = Path(
    "data/verification/osm_review_report.txt"
)


df = pd.read_csv(INPUT_PATH)


def clean(value):
    if pd.isna(value):
        return "N/A"

    value = str(value).strip()

    if value == "":
        return "None recorded"

    return value


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("FIREDISTINGUISH — OSM VERIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "This report provides contextual OSM evidence only.\n"
    )

    f.write(
        "It does NOT assign fire labels or determine fire cause.\n\n"
    )

    for _, row in df.iterrows():

        f.write("\n")
        f.write("#" * 80 + "\n")

        f.write(
            f"EVENT: {clean(row['event_id'])}\n"
        )

        f.write("#" * 80 + "\n\n")

        # ----------------------------------------------------
        # Event information
        # ----------------------------------------------------

        f.write("EVENT INFORMATION\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Start time          : "
            f"{clean(row['start_time'])}\n"
        )

        f.write(
            f"End time            : "
            f"{clean(row['end_time'])}\n"
        )

        f.write(
            f"Duration (hours)    : "
            f"{clean(row['duration_hours'])}\n"
        )

        f.write(
            f"Detection count     : "
            f"{clean(row['detection_count'])}\n"
        )

        f.write(
            f"Centroid latitude   : "
            f"{clean(row['centroid_lat'])}\n"
        )

        f.write(
            f"Centroid longitude  : "
            f"{clean(row['centroid_lon'])}\n"
        )

        f.write(
            f"Displacement (km)   : "
            f"{clean(row['displacement_km'])}\n"
        )

        f.write(
            f"Mean FRP            : "
            f"{clean(row['mean_frp'])}\n"
        )

        f.write(
            f"Maximum FRP         : "
            f"{clean(row['max_frp'])}\n"
        )

        f.write(
            f"Event type          : "
            f"{clean(row['event_type'])}\n"
        )

        # ----------------------------------------------------
        # OSM
        # ----------------------------------------------------

        f.write("\n")
        f.write("OSM CONTEXT\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Query status        : "
            f"{clean(row['osm_status'])}\n"
        )

        f.write(
            f"Search radius (m)   : "
            f"{clean(row['osm_radius_m'])}\n"
        )

        f.write(
            f"OSM elements        : "
            f"{clean(row['osm_element_count'])}\n"
        )

        f.write(
            f"Industrial objects  : "
            f"{clean(row['osm_industrial_count'])}\n"
        )

        f.write(
            f"Nearest industrial  : "
            f"{clean(row['osm_nearest_industrial_km'])} km\n"
        )

        f.write(
            f"Industrial type     : "
            f"{clean(row['osm_nearest_industrial_type'])}\n"
        )

        f.write(
            f"Land use            : "
            f"{clean(row['osm_landuse'])}\n"
        )

        f.write(
            f"Building types      : "
            f"{clean(row['osm_building_types'])}\n"
        )

        f.write(
            f"Highway types       : "
            f"{clean(row['osm_highway_types'])}\n"
        )

        f.write(
            f"Place types         : "
            f"{clean(row['osm_place_types'])}\n"
        )

        f.write(
            f"Natural features    : "
            f"{clean(row['osm_natural_features'])}\n"
        )

        f.write(
            f"Waterways           : "
            f"{clean(row['osm_waterways'])}\n"
        )

        f.write(
            f"Amenities           : "
            f"{clean(row['osm_amenities'])}\n"
        )

        f.write(
            f"Context             : "
            f"{clean(row['osm_context'])}\n"
        )

        if clean(row["osm_status"]) == "error":

            f.write("\n")
            f.write(
                "NOTE: OSM query failed. "
                "Absence of features cannot be inferred.\n"
            )

        # ----------------------------------------------------
        # Verification fields
        # ----------------------------------------------------

        f.write("\n")
        f.write("VERIFICATION\n")
        f.write("-" * 80 + "\n")

        f.write(
            "Verification status : not_reviewed\n"
        )

        f.write(
            "Verified label      : not assigned\n"
        )

        f.write(
            "Satellite checked   : no\n"
        )

        f.write(
            "Weather checked     : no\n"
        )

        f.write(
            "Human conclusion    : pending\n"
        )

        f.write("\n")


print("=" * 80)
print("OSM REVIEW REPORT CREATED")
print("=" * 80)

print(f"\nEvents: {len(df)}")
print(f"Output: {OUTPUT_PATH}")