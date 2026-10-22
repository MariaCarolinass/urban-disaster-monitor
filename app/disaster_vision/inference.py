"""Image and video inference callbacks used by the Gradio interface."""

import math
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr

from .config import (
    MAX_UPLOAD_BYTES,
    MAX_VIDEO_FPS,
    MAX_VIDEO_FRAMES,
    MAX_VIDEO_LONG_SIDE,
    MAX_VIDEO_PIXELS,
    MAX_VIDEO_SECONDS,
    MAX_VIDEO_SHORT_SIDE,
    VIDEO_OUTPUT_MAX_AGE_SECONDS,
    VIDEO_OUTPUT_PREFIX,
)
from .drawing import get_rfdetr_class_name, rfdetr_bounding_box, yolo_bounding_box
from .models import get_rfdetr_model, get_yolo_model
from .validation import check_file_size, load_image


def image_detection(image, conf_threshold, backend, yolo_variant, rfdetr_variant):
    if image is None:
        raise gr.Error("Upload an image before processing.")
    image_rgb = load_image(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if backend == "YOLOv26":
        model_image = get_yolo_model(yolo_variant)
        results = model_image(image_bgr, conf=conf_threshold)
        annotated_image = yolo_bounding_box(image_rgb, results)

        detected_classes = set(int(box.cls[0]) for box in results[0].boxes)
        class_names = results[0].names
        predictions = ", ".join(sorted(class_names[i] for i in detected_classes))
        return annotated_image, predictions

    if backend != "RF-DETR":
        raise gr.Error(f"Unknown model family: {backend}")
    model_image = get_rfdetr_model(variant=rfdetr_variant)
    detections = model_image.predict(image_rgb, threshold=conf_threshold)
    annotated_image = rfdetr_bounding_box(image_rgb, detections)

    class_names = []
    seen = set()
    data = getattr(detections, "data", None)
    if isinstance(data, dict) and data.get("class_name") is not None:
        for class_name in data["class_name"]:
            if class_name not in seen:
                seen.add(class_name)
                class_names.append(str(class_name))
    else:
        class_ids = getattr(detections, "class_id", None)
        if class_ids is not None:
            for class_id in class_ids:
                class_name = str(int(class_id))
                if class_name not in seen:
                    seen.add(class_name)
                    class_names.append(class_name)

    predictions = ", ".join(sorted(class_names)) if class_names else "Nenhuma classe detectada"
    return annotated_image, predictions


def _remove_old_video_outputs():
    cutoff = time.time() - VIDEO_OUTPUT_MAX_AGE_SECONDS
    for path in Path(tempfile.gettempdir()).glob(f"{VIDEO_OUTPUT_PREFIX}*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def video_detection(video_path, conf_threshold, backend, yolo_variant, rfdetr_variant, frame_skip=3):
    if not video_path:
        raise gr.Error("Upload a video before processing.")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1.")
    check_file_size(video_path, MAX_UPLOAD_BYTES, "Video")

    _remove_old_video_outputs()
    cap = None
    out = None
    output_path = None
    completed = False
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error("Could not open the video.")
        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if any(not math.isfinite(value) or value <= 0 for value in (raw_width, raw_height, fps)):
            raise gr.Error("Video dimensions or frame rate are invalid.")
        width, height = int(raw_width), int(raw_height)
        if width == 0 or height == 0:
            raise gr.Error("Video dimensions are invalid.")
        if (
            width * height > MAX_VIDEO_PIXELS
            or max(width, height) > MAX_VIDEO_LONG_SIDE
            or min(width, height) > MAX_VIDEO_SHORT_SIDE
        ):
            raise gr.Error("Video exceeds the 1280×720 resolution limit.")
        if fps > MAX_VIDEO_FPS:
            raise gr.Error("Video exceeds the 60 FPS limit.")
        reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(reported_frames) and reported_frames > 0:
            if reported_frames > MAX_VIDEO_FRAMES or reported_frames / fps > MAX_VIDEO_SECONDS:
                raise gr.Error("Video exceeds the 60 second or 1,800 frame limit.")

        with tempfile.NamedTemporaryFile(prefix=VIDEO_OUTPUT_PREFIX, suffix=".mp4", delete=False) as file:
            output_path = Path(file.name)
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            raise gr.Error("Could not create the output video.")

        if backend == "YOLOv26":
            model_video = get_yolo_model(yolo_variant)
        elif backend == "RF-DETR":
            model_video = get_rfdetr_model(rfdetr_variant)
        else:
            raise gr.Error(f"Unknown model family: {backend}")

        all_classes = set()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise gr.Error("The video contains an invalid frame.")
            if frame_count >= MAX_VIDEO_FRAMES or frame_count / fps >= MAX_VIDEO_SECONDS:
                raise gr.Error("Video exceeds the 60 second or 1,800 frame limit.")
            if frame.shape[:2] != (height, width):
                raise gr.Error("Video frame dimensions changed during processing.")

            annotated_frame = frame
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if backend == "YOLOv26":
                    results = model_video(frame, conf=conf_threshold)
                    annotated_frame = yolo_bounding_box(frame_rgb, results)
                    all_classes.update(results[0].names[int(c)] for c in results[0].boxes.cls)
                else:
                    detections = model_video.predict(frame_rgb, threshold=conf_threshold)
                    annotated_frame = rfdetr_bounding_box(frame_rgb, detections)
                    boxes = getattr(detections, "xyxy", None)
                    if boxes is not None:
                        all_classes.update(get_rfdetr_class_name(detections, i) for i in range(len(boxes)))
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            out.write(annotated_frame)
            frame_count += 1

        if frame_count == 0:
            raise gr.Error("The video contains no readable frames.")
        predictions = ", ".join(sorted(all_classes)) if all_classes else "Nenhuma classe detectada"
        completed = True
        return str(output_path), predictions
    except cv2.error as exc:
        raise gr.Error("The video is invalid or could not be decoded.") from exc
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if not completed and output_path is not None:
            output_path.unlink(missing_ok=True)
