import os
from typing import List

import pandas as pd
import plotly.express as px
import spacy
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Multi-Modal Disaster Response Assistant",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Multi-Modal Disaster Response Assistant")
st.caption(
    "Text intelligence is implemented. Voice (audio) and Image/Video are supported as upload/preview; you can plug in Whisper/CNN/video models later."
)


# -----------------------------
# NLP model (safe fallback)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")


nlp = load_nlp()


# -----------------------------
# Core logic
# -----------------------------
DISASTER_KEYWORDS = {
    "Flood": ["flood", "waterlogging", "heavy rain", "overflow", "inundation"],
    "Fire": ["fire", "burning", "smoke", "flames", "wildfire"],
    "Earthquake": ["earthquake", "tremor", "aftershock", "building collapse"],
    "Cyclone": ["cyclone", "storm", "hurricane", "typhoon"],
    "Accident": ["accident", "crash", "collision", "road accident"],
}

CRITICAL_WORDS = [
    "trapped",
    "dead",
    "urgent",
    "help",
    "collapsed",
    "injured",
    "rescue",
    "critical",
]

RESOURCE_MAP = {
    "Flood": ["Boats", "Food Kits", "Drinking Water", "Medical Team"],
    "Fire": ["Fire Trucks", "Ambulance", "Rescue Team"],
    "Earthquake": ["Search Team", "Medical Unit", "Emergency Shelter"],
    "Cyclone": ["Evacuation Team", "Food Supplies", "Shelter"],
    "Accident": ["Ambulance", "Police", "Medical Staff"],
}


def classify_disaster(text: str) -> str:
    text = (text or "").lower()
    for disaster, keywords in DISASTER_KEYWORDS.items():
        for word in keywords:
            if word in text:
                return disaster
    return "Unknown"


def predict_severity(text: str) -> str:
    text_lower = (text or "").lower()
    score = sum(1 for word in CRITICAL_WORDS if word in text_lower)
    if score >= 4:
        return "Critical"
    if score >= 2:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def extract_location(text: str) -> List[str]:
    doc = nlp(text or "")

    # If blank model (no NER), doc.ents may be empty.
    if not getattr(doc, "ents", None):
        return []

    locations: List[str] = []
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            locations.append(ent.text)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for x in locations:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def get_resources(disaster: str) -> List[str]:
    return RESOURCE_MAP.get(disaster, ["Assessment Team"])


# -----------------------------
# UI: Multi-modal inputs
# -----------------------------
col_text, col_voice, col_media = st.columns([1.1, 1, 1])

with col_text:
    st.subheader("📝 Text Report")
    report_text = st.text_area(
        "Enter Disaster Report",
        height=220,
        placeholder=(
            "Heavy flooding in Visakhapatnam.\n"
            "Several elderly people are trapped.\n"
            "Urgent rescue and medicines required."
        ),
    )

with col_voice:
    st.subheader("🎙️ Audio (Voice Report)")
    st.info("Upload audio. Playback works immediately. Voice-to-text needs Whisper integration.")
    audio_file = st.file_uploader(
        "Upload audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "aac", "ogg"],
        key="audio",
    )
    if audio_file is not None:
        try:
            st.audio(audio_file, format="audio")
        except Exception as e:
            st.warning(f"Could not play audio: {e}")

with col_media:
    st.subheader("🖼️ Image / 🎥 Video Report")
    st.info("Upload an image or a video. Playback/preview works immediately. Video/Image classification needs a model integration.")

    image_file = st.file_uploader(
        "Upload image (jpg/png)",
        type=["jpg", "jpeg", "png"],
        key="image",
    )

    video_file = st.file_uploader(
        "Upload video (mp4/webm)",
        type=["mp4", "webm"],
        key="video",
    )

    if image_file is not None:
        try:
            from PIL import Image

            img = Image.open(image_file)
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render image: {e}")

    if video_file is not None:
        try:
            st.video(video_file)
        except Exception as e:
            st.warning(f"Could not play video: {e}")


# -----------------------------
# Analyze
# -----------------------------
input_mode = st.selectbox(
    "Choose input mode for analysis",
    ["Text only", "Voice (stub)", "Image/Video (stub)", "Text + (stubs)"],
)

if st.button("Analyze Report", type="primary"):
    used_text: str = report_text or ""

    # Voice (stub): if user uploaded audio and text is empty, we keep a placeholder.
    if input_mode in {"Voice (stub)", "Text + (stubs)"} and audio_file is not None:
        used_text = used_text or "(Voice transcription pending: connect Whisper)"

    # Image/Video (stub): if user uploaded media and text is empty, we keep a placeholder.
    if input_mode in {"Image/Video (stub)", "Text + (stubs)"} and (image_file is not None or video_file is not None):
        used_text = used_text or "(Media analysis pending: connect CNN/video model)"

    disaster = classify_disaster(used_text)
    severity = predict_severity(used_text)
    locations = extract_location(used_text)
    resources = get_resources(disaster)

    priority_map = {"Critical": 100, "High": 70, "Medium": 40, "Low": 10}
    priority_score = priority_map.get(severity, 0)

    st.success("Analysis Complete")

    a, b, c, d = st.columns(4)
    a.metric("Disaster Type", disaster)
    b.metric("Severity", severity)
    c.metric("Locations Found", len(locations))
    d.metric("Priority Score", priority_score)

    st.divider()

    st.subheader("📍 Extracted Locations")
    if locations:
        for loc in locations:
            st.write("•", loc)
    else:
        st.write("No location detected (NER may be limited without a spaCy NER model).")

    st.subheader("🚑 Recommended Resources")
    for item in resources:
        st.write("✅", item)


# -----------------------------
# Analytics Dashboard
# -----------------------------
st.divider()

st.subheader("📊 Disaster Analytics Dashboard")

data = pd.DataFrame(
    {
        "Disaster": ["Flood", "Fire", "Earthquake", "Cyclone", "Accident"],
        "Count": [40, 20, 15, 12, 25],
    }
)

fig = px.bar(data, x="Disaster", y="Count", title="Disaster Frequency")
st.plotly_chart(fig, use_container_width=True)

