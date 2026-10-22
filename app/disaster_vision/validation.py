"""Validate uploaded media before loading a model."""

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ALLOWED_IMAGE_FORMATS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def check_file_size(path, max_bytes, kind):
    try:
        size = Path(path).stat().st_size
    except (OSError, TypeError, ValueError) as exc:
        raise gr.Error(f"Could not read the {kind.lower()} file.") from exc
    if size == 0:
        raise gr.Error(f"The {kind.lower()} file is empty.")
    if size > max_bytes:
        raise gr.Error(f"{kind} exceeds the {max_bytes // (1024 * 1024)} MiB limit.")


def load_image(image_path):
    check_file_size(image_path, MAX_IMAGE_BYTES, "Image")
    try:
        with Image.open(image_path) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS:
                raise gr.Error("Use a JPEG, PNG, or WebP image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise gr.Error("Image exceeds the 12 megapixel limit.")
            source.verify()
        with Image.open(image_path) as source:
            return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    except gr.Error:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise gr.Error("The image is invalid or could not be decoded.") from exc
