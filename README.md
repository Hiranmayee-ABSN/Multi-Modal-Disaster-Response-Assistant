# Multi-Modal Disaster Response Assistant

An AI-powered disaster management platform that processes **text**, **voice**, and **images** to support emergency response operations.

## What it does
- **Disaster type classification** (Flood, Fire, Earthquake, Cyclone, Accident)
- **Severity assessment / priority scoring**
- **Location extraction** (NER-based with safe fallback)
- **Emergency resource recommendation**
- **Analytics dashboard**

> This repository includes a working Streamlit app. The current version uses lightweight heuristics + spaCy NER. It is structured so you can later plug in Whisper (voice) and image models.

## Tech Stack
- Python
- Streamlit
- spaCy
- Plotly
- Pandas

Optional extensions (future):
- OpenAI Whisper (voice)
- TensorFlow MobileNetV2 (image)
- Folium (map)

## Run
```bash
streamlit run app.py
```

## Notes
If `en_core_web_sm` is not installed, the app will still run using a blank spaCy pipeline (location extraction will return an empty list).

