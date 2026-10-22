"""Paths, model repositories, colors, and upload limits."""

import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = APP_DIR / "models"
EXAMPLE_VIDEO_CACHE_DIR = Path(tempfile.gettempdir()) / "urban-disaster-monitor-examples"
VIDEO_OUTPUT_PREFIX = "urban-disaster-output-"
VIDEO_OUTPUT_MAX_AGE_SECONDS = 24 * 60 * 60

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_VIDEO_SECONDS = 60
MAX_VIDEO_LONG_SIDE = 1280
MAX_VIDEO_SHORT_SIDE = 720
MAX_VIDEO_PIXELS = MAX_VIDEO_LONG_SIDE * MAX_VIDEO_SHORT_SIDE
MAX_VIDEO_FRAMES = 1800
MAX_VIDEO_FPS = 60
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

YOLO_MODEL_MAP = {
    "yolov26n": "carolinasoares/yolov26n",
    "yolov26s": "carolinasoares/yolov26s",
    "yolov26m": "carolinasoares/yolov26m",
    "yolov26l": "carolinasoares/yolov26l",
}

RFDETR_REPO_MAP = {
    "nano": "carolinasoares/urban-disaster-rfdetrnano",
    "medium": "carolinasoares/urban-disaster-rfdetrmedium",
}
RFDETR_CHECKPOINT_FILE = "checkpoint_best_total.pth"

CLASS_COLORS = {
    "cat": (255, 0, 0),
    "civilian": (255, 255, 0),
    "cow": (0, 255, 0),
    "dog": (0, 0, 255),
    "horse": (255, 0, 255),
    "rescuer": (0, 255, 255),
}

FONT_COLORS = {
    "cat": (255, 255, 255),
    "civilian": (0, 0, 0),
    "cow": (0, 0, 0),
    "dog": (255, 255, 255),
    "horse": (255, 255, 255),
    "rescuer": (0, 0, 0),
}
