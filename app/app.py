import math
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = APP_DIR / "models"
EXAMPLE_VIDEO_CACHE_DIR = Path(tempfile.gettempdir()) / "urban-disaster-monitor-examples"
VIDEO_OUTPUT_PREFIX = "urban-disaster-output-"
VIDEO_OUTPUT_MAX_AGE_SECONDS = 24 * 60 * 60

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
    "cat": (255, 0, 0),         # vermelho
    "civilian": (255, 255, 0),  # amarelo
    "cow": (0, 255, 0),         # verde
    "dog": (0, 0, 255),         # azul
    "horse": (255, 0, 255),     # magenta
    "rescuer": (0, 255, 255),   # ciano
}

FONT_COLORS = {
    "cat": (255, 255, 255),
    "civilian": (0, 0, 0),
    "cow": (0, 0, 0),
    "dog": (255, 255, 255),
    "horse": (255, 255, 255),
    "rescuer": (0, 0, 0),
}

def _get_class_colors(class_name):
    name = str(class_name).strip().casefold()
    return CLASS_COLORS.get(name, (255, 255, 255)), FONT_COLORS.get(name, (0, 0, 0))

def get_model_path(model_variant):
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    repo_id = YOLO_MODEL_MAP[model_variant]
    return hf_hub_download(repo_id=repo_id, filename=f"{model_variant}.pt", cache_dir=MODEL_CACHE_DIR)

@lru_cache(maxsize=1)
def get_yolo_model(variant):
    return YOLO(get_model_path(variant))

@lru_cache(maxsize=1)
def get_rfdetr_model(variant="nano"):
    try:
        if variant == "nano":
            from rfdetr import RFDETRNano as _ModelClass
        else:
            from rfdetr import RFDETRMedium as _ModelClass
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. Run `pip install -r app/requirements.txt` first."
        ) from exc

    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    repo_id = RFDETR_REPO_MAP[variant]
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=RFDETR_CHECKPOINT_FILE,
        cache_dir=MODEL_CACHE_DIR,
    )
    return _ModelClass(pretrain_weights=weights_path)

def _draw_bounding_boxes(image, annotations):
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    height, width = image.shape[:2]

    for box, class_name, confidence in annotations:
        x1, y1, x2, y2 = map(int, box)
        box_color, text_color = _get_class_colors(class_name)
        label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, min(x1, width - text_w - 2))
        text_y = y1 if y1 >= text_h + 3 else min(height - 1, y1 + text_h + 3)

        bg_tl = (text_x, text_y - text_h)
        bg_br = (text_x + text_w, text_y)

        cv2.rectangle(annotated_image, bg_tl, bg_br, box_color, -1)

        cv2.putText(annotated_image, label, (text_x, text_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)

    return annotated_image

def yolo_bounding_box(image, results):
    class_names = results[0].names
    annotations = (
        (box.xyxy[0], class_names[int(box.cls[0])], float(box.conf[0]))
        for box in results[0].boxes
    )
    return _draw_bounding_boxes(image, annotations)

def _get_rfdetr_class_name(detections, index):
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        class_names = data.get("class_name")
        if class_names is not None:
            return str(class_names[index])

    class_ids = getattr(detections, "class_id", None)
    if class_ids is not None:
        return str(int(class_ids[index]))

    return "object"

def rfdetr_bounding_box(image, detections):
    boxes = getattr(detections, "xyxy", None)
    confidences = getattr(detections, "confidence", None)
    if boxes is None or len(boxes) == 0:
        return image.copy()
    annotations = (
        (box, _get_rfdetr_class_name(detections, index),
         float(confidences[index]) if confidences is not None else None)
        for index, box in enumerate(boxes)
    )
    return _draw_bounding_boxes(image, annotations)


def image_detection(image, conf_threshold, backend, yolo_variant, rfdetr_variant):
    if image is None:
        raise gr.Error("Upload an image before processing.")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if backend == "YOLOv26":
        model_image = get_yolo_model(yolo_variant)
        results = model_image(image_bgr, conf=conf_threshold)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        annotated_image = yolo_bounding_box(image_rgb, results)

        detected_classes = set(int(box.cls[0]) for box in results[0].boxes)
        class_names = results[0].names
        predictions = ", ".join(sorted([class_names[i] for i in detected_classes]))
        return annotated_image, predictions
    
    if backend != "RF-DETR":
        raise gr.Error(f"Unknown model family: {backend}")
    model_image = get_rfdetr_model(variant=rfdetr_variant)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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

    _remove_old_video_outputs()
    cap = cv2.VideoCapture(video_path)
    out = None
    output_path = None
    completed = False
    try:
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
                        all_classes.update(_get_rfdetr_class_name(detections, i) for i in range(len(boxes)))
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            out.write(annotated_frame)
            frame_count += 1

        if frame_count == 0:
            raise gr.Error("The video contains no readable frames.")
        predictions = ", ".join(sorted(all_classes)) if all_classes else "Nenhuma classe detectada"
        completed = True
        return str(output_path), predictions
    finally:
        cap.release()
        if out is not None:
            out.release()
        if not completed and output_path is not None:
            output_path.unlink(missing_ok=True)

def load_example_video(filename):
    EXAMPLE_VIDEO_CACHE_DIR.mkdir(exist_ok=True)
    return hf_hub_download(
        repo_id="carolinasoares/urban-disaster-examples",
        filename=filename,
        cache_dir=EXAMPLE_VIDEO_CACHE_DIR,
    )


with gr.Blocks(delete_cache=(3600, 86400)) as app:
    gr.HTML("""
        <h1 style='text-align: center'>Urban Disaster Monitor</h1>
        <h2 style='text-align: center'><a href='https://github.com/MariaCarolinass/urban-disaster-monitor' target='_blank'>GitHub</a></h2>
    """)

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an Image", type="pil")
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                backend = gr.Radio(["YOLOv26", "RF-DETR"], label="Model family", value="YOLOv26")
                yolo_variant = gr.Dropdown(["yolov26n", "yolov26s", "yolov26m", "yolov26l"], label="YOLOv26 variant", info="Used when the model family is YOLOv26.", value="yolov26m")
                rfdetr_variant = gr.Dropdown(["nano", "medium"], label="RF-DETR variant", info="Used when the model family is RF-DETR.", value="nano", visible=False)
                btn = gr.Button("Process Image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Processed Image")
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")

        btn.click(
            fn=image_detection,
            inputs=[image, conf_threshold, backend, yolo_variant, rfdetr_variant],
            outputs=[output_image, output_predictions],
            concurrency_id="model_inference",
            concurrency_limit=1,
        )

        def _toggle_variants(selected):
            return (gr.update(visible=selected == "YOLOv26"), gr.update(visible=selected != "YOLOv26"))

        backend.change(fn=_toggle_variants, inputs=[backend], outputs=[yolo_variant, rfdetr_variant])
    
        gr.Examples(
            examples=[[str(path)] for path in sorted((APP_DIR / "examples").glob("*"))
                      if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            inputs=image,
            label="Example Images",
        )
        
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload a Video", autoplay=True)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                backend = gr.Radio(["YOLOv26", "RF-DETR"], label="Model family", value="YOLOv26")
                yolo_variant = gr.Dropdown(["yolov26n", "yolov26s", "yolov26m", "yolov26l"], label="YOLOv26 variant", info="Used when the model family is YOLOv26.", value="yolov26m")
                rfdetr_variant = gr.Dropdown(["nano", "medium"], label="RF-DETR variant", info="Used when the model family is RF-DETR.", value="nano", visible=False)
                btn = gr.Button("Process Video", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Processed Video", autoplay=True)
                output_predictions = gr.Textbox(label="Predictions", placeholder="Predictions will appear here...")

        btn.click(
            fn=video_detection,
            inputs=[video, conf_threshold, backend, yolo_variant, rfdetr_variant],
            outputs=[output_video, output_predictions],
            concurrency_id="model_inference",
            concurrency_limit=1,
        )

        def _toggle_variants_video(selected):
            return (gr.update(visible=selected == "YOLOv26"), gr.update(visible=selected != "YOLOv26"))

        backend.change(fn=_toggle_variants_video, inputs=[backend], outputs=[yolo_variant, rfdetr_variant])

        with gr.Row():
            rescue_example = gr.Button("Load rescue simulation")
            rescuer_example = gr.Button("Load rescuer example")
        rescue_example.click(fn=lambda: load_example_video("rescue_simulation.mp4"), outputs=video)
        rescuer_example.click(fn=lambda: load_example_video("rescuer.mp4"), outputs=video)

if __name__ == "__main__":
    app.launch()
