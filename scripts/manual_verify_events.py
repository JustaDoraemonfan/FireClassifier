"""
FireDistinguish
---------------
Manual verification interface for candidate fire events.

Reads:
    data/verification/verification_review_v1.csv

Satellite evidence:
    data/verification/satellite/chips/<EVENT_ID>/

Writes:
    data/verification/verification_review_v2.csv

Original datasets are never modified.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_review_v1.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "verification_review_v2.csv"
)

SATELLITE_DIR = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "satellite"
    / "chips"
)


LABELS = [
    "confirmed_fire",
    "probable_fire",
    "uncertain",
    "false_positive",
    "not_verifiable",
]

SOURCE_CLASSES = [
    "WILDFIRE",
    "INDUSTRIAL",
    "AGRICULTURAL",
    "MINING_OTHER",
    "UNKNOWN",
]

CONFIDENCES = [
    "high",
    "medium",
    "low",
]

REASONS = [
    "Clear localized burn/fire-related change",
    "Strong vegetation/burn index change",
    "Visible structural or land-surface change",
    "Evidence consistent with fire but not conclusive",
    "Weak or ambiguous satellite evidence",
    "Heavy cloud/haze limits interpretation",
    "Satellite imagery unusable",
    "No meaningful before/after change",
    "Likely non-fire thermal detection",
    "Other",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FireDistinguish Verification",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_CSV.exists():

    st.error(
        f"Verification dataset not found:\n\n{INPUT_CSV}"
    )

    st.stop()


if "df" not in st.session_state:

    # Resume from previously-saved progress (v2) if it exists,
    # so a rerun/restart never rewinds to the untouched v1 data.
    # Only fall back to the original v1 dataset on a truly fresh start.

    if OUTPUT_CSV.exists():
        st.session_state.df = pd.read_csv(OUTPUT_CSV)
    else:
        st.session_state.df = pd.read_csv(INPUT_CSV)

    if "event_id" not in st.session_state.df.columns:

        st.error(
            "The verification CSV does not contain an event_id column."
        )

        st.stop()

    # Ensure review fields exist.

    review_columns = [
        "verification_label",
        "verification_confidence",
        "verification_reason",
        "source_class",
        "source_class_confidence",
        "source_class_reason",
        "disputed",
        "reviewer_notes",
    ]

    for column in review_columns:

        if column not in st.session_state.df.columns:
            st.session_state.df[column] = ""

        st.session_state.df[column] = (
            st.session_state.df[column]
            .fillna("")
            .astype(str)
        )


df = st.session_state.df


# ============================================================
# SESSION STATE
# ============================================================

if "event_index" not in st.session_state:

    st.session_state.event_index = 0


if "initialized" not in st.session_state:

    st.session_state.initialized = True


if "message" not in st.session_state:

    st.session_state.message = ""


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_dataset():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


# ============================================================
# HELPERS
# ============================================================

def value(row, column, default="N/A"):

    if column not in row.index:
        return default

    result = row[column]

    if pd.isna(result):
        return default

    if str(result).strip() == "":
        return default

    return result


def find_image(event_dir, names):

    for name in names:

        path = event_dir / name

        if path.exists():
            return path

    return None


def format_value(row, column):

    result = value(row, column)

    if result == "N/A":
        return "N/A"

    try:

        return f"{float(result):.3f}"

    except Exception:

        return str(result)


# ============================================================
# HEADER
# ============================================================

st.title("🔥 FireDistinguish — Manual Verification")

st.caption(
    "Review the satellite evidence for each candidate fire event."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Event Navigation")

    total_events = len(df)

    st.write(
        f"**Events:** {total_events}"
    )

    # Event selector

    selected_event = st.number_input(
        "Event number",
        min_value=1,
        max_value=total_events,
        value=st.session_state.event_index + 1,
        step=1,
    )

    st.session_state.event_index = (
        selected_event - 1
    )


    # Progress

    reviewed = (
        df["verification_label"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    st.progress(
        reviewed / total_events
        if total_events
        else 0
    )

    st.write(
        f"**Reviewed:** {reviewed} / {total_events}"
    )


    st.divider()

    st.subheader("Labels")

    st.write("🟢 confirmed_fire")
    st.write("🟡 probable_fire")
    st.write("⚪ uncertain")
    st.write("🔴 false_positive")
    st.write("⚫ not_verifiable")

    st.divider()

    st.subheader("Source classes")

    st.write("🌲 WILDFIRE")
    st.write("🏭 INDUSTRIAL")
    st.write("🌾 AGRICULTURAL")
    st.write("⛏️ MINING_OTHER")
    st.write("❓ UNKNOWN")


# ============================================================
# CURRENT EVENT
# ============================================================

index = st.session_state.event_index

row = df.iloc[index]

event_id = str(
    row["event_id"]
)


event_dir = (
    SATELLITE_DIR
    / event_id
)


# ============================================================
# EVENT HEADER
# ============================================================

st.header(
    f"Event {index + 1} / {total_events}"
)

st.subheader(
    event_id
)


# ============================================================
# FIRMS / EVENT INFORMATION
# ============================================================

st.markdown("### FIRMS Event Information")

info_columns = [
    ("Start time", "start_time"),
    ("End time", "end_time"),
    ("Duration (hours)", "duration_hours"),
    ("Detection count", "detection_count"),
    ("Centroid latitude", "centroid_lat"),
    ("Centroid longitude", "centroid_lon"),
    ("Displacement (km)", "displacement_km"),
    ("Maximum FRP (MW)", "max_frp"),
    ("Event type", "event_type"),
]


cols = st.columns(3)


for i, (label, column) in enumerate(info_columns):

    with cols[i % 3]:

        raw = value(row, column)

        if column in [
            "centroid_lat",
            "centroid_lon",
            "duration_hours",
            "displacement_km",
            "max_frp",
        ]:

            raw = format_value(
                row,
                column
            )

        st.metric(
            label,
            str(raw)
        )


# ============================================================
# OSM INFORMATION
# ============================================================

osm_columns = [
    "osm_industrial_count",
    "osm_element_count",
    "osm_nearest_industrial_km",
]


available_osm = [
    column
    for column in osm_columns
    if column in df.columns
]


if available_osm:

    st.markdown("### OSM Context")

    osm_cols = st.columns(
        len(available_osm)
    )

    for i, column in enumerate(available_osm):

        with osm_cols[i]:

            st.metric(
                column.replace("_", " ").title(),
                str(value(row, column))
            )


# ============================================================
# SATELLITE DIRECTORY
# ============================================================

st.markdown("### Satellite Evidence")

if not event_dir.exists():

    st.warning(
        f"Satellite evidence directory not found:\n\n{event_dir}"
    )

else:

    st.caption(
        f"Evidence directory: `{event_dir}`"
    )


# ============================================================
# FIND IMAGES
# ============================================================

before_image = find_image(
    event_dir,
    [
        "before_rgb.png",
        "before.png",
    ]
)


after_image = find_image(
    event_dir,
    [
        "after_rgb.png",
        "after.png",
    ]
)


ndvi_image = find_image(
    event_dir,
    [
        "ndvi_change.png",
        "ndvi_change_rgb.png",
        "ndvi.png",
    ]
)


dnbr_image = find_image(
    event_dir,
    [
        "dnbr.png",
        "dnbr_rgb.png",
        "dNBR.png",
    ]
)


# ============================================================
# BEFORE / AFTER
# ============================================================

image_cols = st.columns(2)


with image_cols[0]:

    st.markdown("#### BEFORE")

    if before_image:

        st.image(
            str(before_image),
            use_container_width=True
        )

    else:

        st.warning(
            "Before image not found."
        )


with image_cols[1]:

    st.markdown("#### AFTER")

    if after_image:

        st.image(
            str(after_image),
            use_container_width=True
        )

    else:

        st.warning(
            "After image not found."
        )


# ============================================================
# QUANTITATIVE SATELLITE METRICS
# ============================================================

index_metrics = [
    ("BEFORE NDVI mean", "before_ndvi_mean"),
    ("AFTER NDVI mean", "after_ndvi_mean"),
    ("ΔNDVI mean (before − after)", "delta_ndvi_mean"),
    ("ΔNDVI median (before − after)", "delta_ndvi_median"),
    ("ΔNDVI P90", "delta_ndvi_p90"),
    ("BEFORE NBR mean", "before_nbr_mean"),
    ("AFTER NBR mean", "after_nbr_mean"),
    ("dNBR mean (before − after)", "dnbr_mean"),
    ("dNBR median (before − after)", "dnbr_median"),
    ("dNBR P90", "dnbr_p90"),
]

available_index_metrics = [
    (label, column)
    for label, column in index_metrics
    if column in df.columns
]

if available_index_metrics:
    st.markdown("### Quantitative Satellite Metrics")
    metric_cols = st.columns(3)
    for i, (label, column) in enumerate(available_index_metrics):
        with metric_cols[i % 3]:
            st.metric(label, format_value(row, column))
    st.caption(
        "Event-level Sentinel-2 spectral statistics. "
        "ΔNDVI = NDVI before − after; "
        "dNBR = NBR before − after."
    )
else:
    st.info(
        "Quantitative NDVI/NBR metrics are not present in the review dataset. "
        "Run the satellite index stage before relying on spectral-change evidence."
    )


# ============================================================
# NDVI / dNBR
# ============================================================

index_cols = st.columns(2)


with index_cols[0]:

    st.markdown("#### NDVI CHANGE")

    if ndvi_image:

        st.image(
            str(ndvi_image),
            use_container_width=True
        )

    else:

        st.info(
            "No NDVI change visualization available."
        )


with index_cols[1]:

    st.markdown("#### dNBR")

    if dnbr_image:

        st.image(
            str(dnbr_image),
            use_container_width=True
        )

    else:

        st.info(
            "No dNBR visualization available."
        )


# ============================================================
# VERIFICATION
# ============================================================

st.divider()

st.header("Manual Verification")

st.caption(
    "Event validity and source class are recorded as separate decisions. "
    "A confirmed fire does not automatically imply any particular source class."
)


current_label = value(
    row,
    "verification_label",
    ""
)

current_source_class = value(
    row,
    "source_class",
    ""
)

current_source_confidence = value(
    row,
    "source_class_confidence",
    ""
)

current_source_reason = value(
    row,
    "source_class_reason",
    ""
)

current_disputed = value(
    row,
    "disputed",
    ""
)

current_confidence = value(
    row,
    "verification_confidence",
    ""
)

current_reason = value(
    row,
    "verification_reason",
    ""
)

current_notes = value(
    row,
    "reviewer_notes",
    ""
)


# Convert N/A back to empty values.

if current_label == "N/A":
    current_label = ""

if current_source_class == "N/A":
    current_source_class = ""

if current_source_confidence == "N/A":
    current_source_confidence = ""

if current_source_reason == "N/A":
    current_source_reason = ""

if current_disputed == "N/A":
    current_disputed = ""

if current_confidence == "N/A":
    current_confidence = ""

if current_reason == "N/A":
    current_reason = ""

if current_notes == "N/A":
    current_notes = ""


# ------------------------------------------------------------
# Label
# ------------------------------------------------------------

label_index = (
    LABELS.index(current_label)
    if current_label in LABELS
    else None
)


selected_label = st.selectbox(
    "Verification label",
    options=[""] + LABELS,
    index=(
        label_index + 1
        if label_index is not None
        else 0
    ),
    key=f"label_{event_id}",
)


# ------------------------------------------------------------
# Source class
# ------------------------------------------------------------

source_class_index = (
    SOURCE_CLASSES.index(current_source_class)
    if current_source_class in SOURCE_CLASSES
    else None
)

selected_source_class = st.selectbox(
    "Source class",
    options=[""] + SOURCE_CLASSES,
    index=(
        source_class_index + 1
        if source_class_index is not None
        else 0
    ),
    help=(
        "Classify the likely source/type of the event independently "
        "from whether a genuine fire occurred."
    ),
    key=f"source_class_{event_id}",
)


# ------------------------------------------------------------
# Source class confidence
# ------------------------------------------------------------

source_confidence_index = (
    CONFIDENCES.index(current_source_confidence)
    if current_source_confidence in CONFIDENCES
    else None
)

selected_source_confidence = st.selectbox(
    "Source class confidence",
    options=[""] + CONFIDENCES,
    index=(
        source_confidence_index + 1
        if source_confidence_index is not None
        else 0
    ),
    key=f"source_confidence_{event_id}",
)


# ------------------------------------------------------------
# Source class reason
# ------------------------------------------------------------

selected_source_reason = st.text_area(
    "Source class reasoning",
    value=current_source_reason,
    height=100,
    placeholder=(
        "Explain why the evidence supports this source class..."
    ),
    key=f"source_reason_{event_id}",
)


# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

confidence_index = (
    CONFIDENCES.index(current_confidence)
    if current_confidence in CONFIDENCES
    else None
)


selected_confidence = st.selectbox(
    "Verification confidence",
    options=[""] + CONFIDENCES,
    index=(
        confidence_index + 1
        if confidence_index is not None
        else 0
    ),
    key=f"confidence_{event_id}",
)


# ------------------------------------------------------------
# Reason
# ------------------------------------------------------------

reason_index = (
    REASONS.index(current_reason)
    if current_reason in REASONS
    else None
)


selected_reason = st.selectbox(
    "Verification reason",
    options=[""] + REASONS,
    index=(
        reason_index + 1
        if reason_index is not None
        else 0
    ),
    key=f"reason_{event_id}",
)


# ------------------------------------------------------------
# Dispute flag
# ------------------------------------------------------------

selected_disputed = st.checkbox(
    "Mark this review as disputed",
    value=str(current_disputed).strip().lower() in {
        "true", "1", "yes", "y"
    },
    help=(
        "Use this when the current review should be revisited or "
        "requires another reviewer."
    ),
    key=f"disputed_{event_id}",
)


# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------

selected_notes = st.text_area(
    "Reviewer notes",
    value=current_notes,
    height=120,
    placeholder=(
        "Describe what you see in the before/after imagery..."
    ),
    key=f"notes_{event_id}",
)


# ============================================================
# SAVE REVIEW
# ============================================================

st.divider()

button_cols = st.columns(4)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

with button_cols[0]:

    if st.button(
        "💾 Save Review",
        use_container_width=True
    ):

        df.at[
            index,
            "verification_label"
        ] = selected_label

        df.at[
            index,
            "source_class"
        ] = selected_source_class

        df.at[
            index,
            "source_class_confidence"
        ] = selected_source_confidence

        df.at[
            index,
            "source_class_reason"
        ] = selected_source_reason

        df.at[
            index,
            "disputed"
        ] = selected_disputed

        df.at[
            index,
            "verification_confidence"
        ] = selected_confidence

        df.at[
            index,
            "verification_reason"
        ] = selected_reason

        df.at[
            index,
            "reviewer_notes"
        ] = selected_notes

        save_dataset()

        st.success(
            "Review saved."
        )


# ------------------------------------------------------------
# PREVIOUS
# ------------------------------------------------------------

with button_cols[1]:

    if st.button(
        "← Previous",
        use_container_width=True,
        disabled=(index == 0),
    ):

        # Save current state first.

        df.at[
            index,
            "verification_label"
        ] = selected_label

        df.at[
            index,
            "source_class"
        ] = selected_source_class

        df.at[
            index,
            "source_class_confidence"
        ] = selected_source_confidence

        df.at[
            index,
            "source_class_reason"
        ] = selected_source_reason

        df.at[
            index,
            "disputed"
        ] = selected_disputed

        df.at[
            index,
            "verification_confidence"
        ] = selected_confidence

        df.at[
            index,
            "verification_reason"
        ] = selected_reason

        df.at[
            index,
            "reviewer_notes"
        ] = selected_notes

        save_dataset()

        st.session_state.event_index -= 1

        st.rerun()


# ------------------------------------------------------------
# NEXT
# ------------------------------------------------------------

with button_cols[2]:

    if st.button(
        "Next →",
        use_container_width=True,
        disabled=(index == total_events - 1),
    ):

        df.at[
            index,
            "verification_label"
        ] = selected_label

        df.at[
            index,
            "source_class"
        ] = selected_source_class

        df.at[
            index,
            "source_class_confidence"
        ] = selected_source_confidence

        df.at[
            index,
            "source_class_reason"
        ] = selected_source_reason

        df.at[
            index,
            "disputed"
        ] = selected_disputed

        df.at[
            index,
            "verification_confidence"
        ] = selected_confidence

        df.at[
            index,
            "verification_reason"
        ] = selected_reason

        df.at[
            index,
            "reviewer_notes"
        ] = selected_notes

        save_dataset()

        st.session_state.event_index += 1

        st.rerun()


# ------------------------------------------------------------
# SAVE ALL
# ------------------------------------------------------------

with button_cols[3]:

    if st.button(
        "Save Dataset",
        use_container_width=True
    ):

        df.at[
            index,
            "verification_label"
        ] = selected_label

        df.at[
            index,
            "source_class"
        ] = selected_source_class

        df.at[
            index,
            "source_class_confidence"
        ] = selected_source_confidence

        df.at[
            index,
            "source_class_reason"
        ] = selected_source_reason

        df.at[
            index,
            "disputed"
        ] = selected_disputed

        df.at[
            index,
            "verification_confidence"
        ] = selected_confidence

        df.at[
            index,
            "verification_reason"
        ] = selected_reason

        df.at[
            index,
            "reviewer_notes"
        ] = selected_notes

        save_dataset()

        st.success(
            f"Saved:\n{OUTPUT_CSV}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Original verification_review_v1.csv is never modified. "
    "Manual reviews are saved to verification_review_v2.csv."
)