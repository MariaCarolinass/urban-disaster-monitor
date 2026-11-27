"""Paths shared by experiments; optionally point to a separate data checkout."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("UDM_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
FIGURES_DIR = ROOT / "notebooks/generated/figures"
EXTERNAL_IMAGE = ROOT / "static/images/qualitative/external-disaster.jpg"
