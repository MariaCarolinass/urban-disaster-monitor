---
title: urban_disaster_monitor
app_file: app.py
sdk: gradio
sdk_version: 6.30.0
---

## Code layout

- `app.py`: Gradio interface and Space entry point.
- `disaster_vision/config.py`: paths, model repositories, colors, and media limits.
- `disaster_vision/models.py`: model downloads and bounded instance caches.
- `disaster_vision/drawing.py`: shared RGB bounding boxes and labels.
- `disaster_vision/validation.py`: uploaded image and file validation.
- `disaster_vision/inference.py`: image and video inference callbacks.
- `disaster_vision/examples.py`: on-demand example video downloads.

Run locally from this directory with `python app.py` after installing `requirements.txt`.
