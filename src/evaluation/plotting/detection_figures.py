"""Generate legible illustrative detection figures from saved YOLO26m weights.

Each run samples new validation images and records its seed and selection so it
can be reproduced. Images already used in the current selection cycle are
excluded. The examples are illustrative, not a quantitative sample.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..paths import ROOT, FIGURES_DIR, EXTERNAL_IMAGE
OUTPUT_DIR = FIGURES_DIR
DATASET = ROOT / "dataset/valid"
WEIGHTS = ROOT / "models/yolov26m/weights/best.pt"
MANIFEST = ROOT / "models/comparisons/qualitative-mosaic.json"
MOSAIC_CONFIDENCE = 0.70
INFERENCE_CONFIDENCE = 0.25
SAMPLE_COUNT = 120
PREVIEW_COUNT = 12
PANEL_SIZE = 640
GUTTER = 14
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
# Manual visual review found a cow labeled as dog in this source image.
EXCLUDED_IMAGES = {
    "imagem_012_1_png.rf.27667da9c4e2bee08cb848315fb4a1d2.jpg",
}
SYNTHETIC_PREFIXES = ("gemini_", "freepik", "imagem_", "image_", "20250924_")
COLORS = {
    "cat": (255, 0, 0), "civilian": (255, 255, 0),
    "cow": (0, 255, 0), "dog": (0, 0, 255),
    "horse": (255, 0, 255), "rescuer": (0, 255, 255),
}
TEXT_COLORS = {
    "cat": "white", "civilian": "black", "cow": "black",
    "dog": "white", "horse": "white", "rescuer": "black",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_boxes(path: Path, width: int, height: int) -> list[tuple[int, tuple[float, ...]]]:
    labels = DATASET / "labels" / f"{path.stem}.txt"
    result = []
    for line in labels.read_text().splitlines():
        class_id, xc, yc, bw, bh = [float(value) for value in line.split()[:5]]
        result.append((int(class_id), ((xc - bw / 2) * width, (yc - bh / 2) * height,
                                       (xc + bw / 2) * width, (yc + bh / 2) * height)))
    return result


def eligible_files() -> list[Path]:
    """Conservative pool: omit known synthetic and ambiguous image-name groups."""
    return [path for path in sorted((DATASET / "images").glob("*.jpg"))
            if path.name not in EXCLUDED_IMAGES
            and not path.name.lower().startswith(SYNTHETIC_PREFIXES)]


def lottery(count: int = PREVIEW_COUNT, seed: int | None = None,
            exclude_paths: list[str] | None = None) -> dict:
    """Draw a fast, model-free preview batch with some class variety."""
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    used = set(previous.get("selection_history", []))
    used.update(exclude_paths or [])
    files = [path for path in eligible_files() if str(path.relative_to(ROOT)) not in used]
    if len(files) < count:
        used = {row["path"] for row in previous.get("selected", [])}
        used.update(exclude_paths or [])
        files = [path for path in eligible_files() if str(path.relative_to(ROOT)) not in used]
    if len(files) < count:
        raise RuntimeError("There are too few eligible images for the preview")
    actual_seed = seed if seed is not None else secrets.randbits(64)
    rng = random.Random(actual_seed)
    by_class = {class_id: [] for class_id in range(6)}
    for path in files:
        label_path = DATASET / "labels" / f"{path.stem}.txt"
        classes = {int(line.split()[0]) for line in label_path.read_text().splitlines()}
        for class_id in classes:
            by_class[class_id].append(path)
    chosen = []
    for class_id in (2, 4, 3, 5, 1):
        options = [path for path in by_class[class_id] if path not in chosen]
        chosen.extend(rng.sample(options, min(2, len(options), count - len(chosen))))
    remaining = [path for path in files if path not in chosen]
    chosen.extend(rng.sample(remaining, count - len(chosen)))
    rng.shuffle(chosen)
    return {"seed": actual_seed, "paths": [str(path.relative_to(ROOT)) for path in chosen]}


def preview_sheet(drawn: dict) -> Image.Image:
    """Numbered contact sheet; no YOLO inference is needed to inspect images."""
    paths = [ROOT / name for name in drawn["paths"]]
    columns, cell_w, cell_h = 4, 240, 210
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    font = ImageFont.truetype(str(FONT), 21)
    drawer = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (cell_w - 12, cell_h - 36))
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
        drawer.text((x + 8, y + cell_h - 30), str(index), font=font, fill="black")
    return sheet


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def candidate(path: Path, result) -> dict | None:
    boxes = result.boxes
    if boxes is None or not 1 <= len(boxes) <= 3:
        return None
    height, width = result.orig_shape
    ground_truth = label_boxes(path, width, height)
    detections = []
    matched = []
    for box in boxes:
        coords = tuple(float(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        area_fraction = (coords[2] - coords[0]) * (coords[3] - coords[1]) / (width * height)
        agreement = max((iou(coords, gt) for gt_class, gt in ground_truth if gt_class == class_id), default=0.0)
        name = result.names[class_id]
        detections.append({"class": name, "confidence": round(confidence, 4),
                           "bbox_xyxy": [round(value, 1) for value in coords],
                           "annotation_iou": round(agreement, 3)})
        if confidence >= MOSAIC_CONFIDENCE and agreement >= 0.45 and area_fraction >= 0.025:
            matched.append((name, confidence))
    if not matched:
        return None
    return {"path": str(path.relative_to(ROOT)), "image_sha256": sha256(path),
            "detections": detections, "matched_classes": sorted({name for name, _ in matched}),
            "score": round(max(conf for _, conf in matched) + 0.025 * min(len(matched), 3)
                           - 0.08 * max(0, len(boxes) - 3), 4)}


def select(candidates: list[dict]) -> list[dict]:
    selected, covered = [], set()
    pool = candidates.copy()
    while len(selected) < 4:
        if not pool:
            raise RuntimeError("Increase SAMPLE_COUNT to find four eligible candidates")
        chosen = max(pool, key=lambda row: (row["score"] + 0.18 * len(set(row["matched_classes"]) - covered), row["path"]))
        selected.append(chosen)
        covered.update(chosen["matched_classes"])
        pool.remove(chosen)
    return selected


def draw_predictions(image: Image.Image, detections: list[dict], out_size: tuple[int, int] | None = None) -> Image.Image:
    """Draw larger boxes and opaque class/confidence labels in RGB colors."""
    image = image.convert("RGB")
    original_width, original_height = image.size
    if out_size is not None:
        image = image.resize(out_size, Image.Resampling.LANCZOS)
    width, height = image.size
    scale_x, scale_y = width / original_width, height / original_height
    font_size = max(23, round(min(width, height) * 0.038))
    font = ImageFont.truetype(str(FONT), font_size)
    drawer = ImageDraw.Draw(image)
    border = max(4, round(min(width, height) * 0.006))
    occupied_labels = []
    for item in sorted(detections, key=lambda row: row["confidence"], reverse=True):
        name = item["class"]
        color = COLORS.get(name, (255, 255, 255))
        coords = item["bbox_xyxy"]
        x1, y1, x2, y2 = (round(coords[0] * scale_x), round(coords[1] * scale_y),
                          round(coords[2] * scale_x), round(coords[3] * scale_y))
        drawer.rectangle((x1, y1, x2, y2), outline=color, width=border)
        label = f"{name} {item['confidence']:.2f}"
        text_box = drawer.textbbox((0, 0), label, font=font)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        label_width, label_height = text_width + 10, text_height + 8
        x_positions = (x1, x2 - label_width, x1 - label_width // 2)
        y_positions = (y1 - label_height - 3, y1 + border + 3,
                       y1 - 2 * label_height - 7, y1 + border + label_height + 7)
        positions = []
        for y_candidate in y_positions:
            for x_candidate in x_positions:
                label_x = max(0, min(x_candidate, width - label_width))
                label_y = max(0, min(y_candidate, height - label_height))
                rect = (label_x, label_y, label_x + label_width, label_y + label_height)
                if all(rect[2] + 3 <= old[0] or old[2] + 3 <= rect[0]
                       or rect[3] + 3 <= old[1] or old[3] + 3 <= rect[1]
                       for old in occupied_labels):
                    positions.append((rect, label_x, label_y))
        if positions:
            rect, label_x, label_y = positions[0]
        else:
            label_x = max(0, min(x1, width - label_width))
            label_y = max(0, min(y1 + border + 3, height - label_height))
            rect = (label_x, label_y, label_x + label_width, label_y + label_height)
        occupied_labels.append(rect)
        drawer.rectangle(rect, fill=color)
        drawer.text((label_x + 5, label_y + 3 - text_box[1]), label,
                    font=font, fill=TEXT_COLORS.get(name, "black"))
    return image


def mosaic(images: list[Image.Image]) -> Image.Image:
    if len(images) != 4:
        raise ValueError("O mosaico deve conter quatro imagens")
    side = 2 * PANEL_SIZE + 3 * GUTTER
    result = Image.new("RGB", (side, side), "white")
    for index, image in enumerate(images):
        x = GUTTER + (index % 2) * (PANEL_SIZE + GUTTER)
        y = GUTTER + (index // 2) * (PANEL_SIZE + GUTTER)
        result.paste(image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS), (x, y))
    return result


def detections_from_result(result) -> list[dict]:
    output = []
    for box in result.boxes:
        output.append({"class": result.names[int(box.cls[0])],
                       "confidence": float(box.conf[0]),
                       "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]})
    return output


def generate_external_and_video(model: YOLO, output_dir: Path) -> None:
    original = EXTERNAL_IMAGE
    result = model.predict(str(original), conf=INFERENCE_CONFIDENCE, imgsz=640, device="cpu", verbose=False)[0]
    draw_predictions(Image.open(original), detections_from_result(result)).save(output_dir / "imagem_externa_deteccao.png")

    video = cv2.VideoCapture(str(ROOT / "static/video/rescue_simulation.mp4"))
    if not video.isOpened():
        raise RuntimeError("Could not open the source video")
    try:
        selected = {240: "video_deteccao_visibilidade.png",
                    336: "video_deteccao_baixa_luminosidade.png"}
        # Decode in order, matching the complete-video measurement. Seeking can
        # return a neighboring frame in this variable-frame-rate source.
        for frame_number in range(max(selected) + 1):
            ok, frame = video.read()
            if not ok:
                raise RuntimeError(f"Could not read video frame {frame_number}")
            if frame_number not in selected:
                continue
            filename = selected[frame_number]
            result = model.predict(frame, conf=INFERENCE_CONFIDENCE, imgsz=640, device="cpu", verbose=False)[0]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_predictions(image, detections_from_result(result)).save(output_dir / filename)
    finally:
        video.release()


def generate(seed: int | None = None, avoid_history: bool = True,
             source_paths: list[str] | None = None,
             chosen_indices: list[int] | None = None, output_dir=None) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    from ultralytics import YOLO

    files = eligible_files()
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    history = list(previous.get("selection_history", [row["path"] for row in previous.get("selected", [])]))
    cycle = int(previous.get("selection_cycle", 1))
    used = set(history) if avoid_history else set()
    available = [path for path in files if str(path.relative_to(ROOT)) not in used]
    needed = len(source_paths) if source_paths is not None else SAMPLE_COUNT
    if len(available) < needed:
        history = [row["path"] for row in previous.get("selected", [])] if avoid_history else []
        used = set(history)
        cycle += 1
        available = [path for path in files if str(path.relative_to(ROOT)) not in used]
    actual_seed = seed if seed is not None else secrets.randbits(64)
    if seed is None:
        while actual_seed == previous.get("seed"):
            actual_seed = secrets.randbits(64)
    rng = random.Random(actual_seed)
    if source_paths is None:
        sampled = rng.sample(available, SAMPLE_COUNT)
    else:
        if len(source_paths) != len(set(source_paths)) or not source_paths:
            raise ValueError("The preview must contain distinct image paths")
        if any(ROOT / name not in files for name in source_paths):
            raise ValueError("The preview contains a file outside the eligible set")
        if chosen_indices:
            if len(chosen_indices) != 4 or len(set(chosen_indices)) != 4:
                raise ValueError("Choose four distinct indices")
            if any(index < 0 or index >= len(source_paths) for index in chosen_indices):
                raise ValueError("Index is outside the preview")
            sampled = [ROOT / source_paths[index] for index in chosen_indices]
        else:
            sampled = [ROOT / name for name in source_paths]
    model = YOLO(str(WEIGHTS))
    results = model.predict([str(path) for path in sampled], conf=MOSAIC_CONFIDENCE, imgsz=640,
                            batch=8, device="cpu", verbose=False)
    candidates = [entry for path, result in zip(sampled, results)
                  if (entry := candidate(path, result)) is not None]
    if chosen_indices:
        if len(candidates) != 4:
            raise ValueError("A selected image had no suitable detections. Choose different indices or use automatic selection")
        selected = candidates
    else:
        selected = select(candidates)
    raw, annotated = [], []
    for item in selected:
        image = Image.open(ROOT / item["path"]).convert("RGB")
        raw.append(image)
        annotated.append(draw_predictions(image, item["detections"], (PANEL_SIZE, PANEL_SIZE)))
    output_dir.mkdir(parents=True, exist_ok=True)
    mosaic(raw).save(output_dir / "mosaico_dataset.png")
    mosaic(annotated).save(output_dir / "mosaico_anotacoes.png")
    generate_external_and_video(model, output_dir)
    manifest = {
        "model": str(WEIGHTS.relative_to(ROOT)), "model_sha256": sha256(WEIGHTS),
        "split": "valid", "seed": actual_seed, "sample_count": len(sampled),
        "preview_paths": source_paths or [str(path.relative_to(ROOT)) for path in sampled],
        "sampled_paths": [str(path.relative_to(ROOT)) for path in sampled],
        "chosen_indices": chosen_indices or [],
        "selection_cycle": cycle,
        "selection_history": list(dict.fromkeys(history + [item["path"] for item in selected])),
        "excluded_images": sorted(EXCLUDED_IMAGES),
        "excluded_filename_prefixes": list(SYNTHETIC_PREFIXES),
        "mosaic_confidence_threshold": MOSAIC_CONFIDENCE,
        "external_video_confidence_threshold": INFERENCE_CONFIDENCE,
        "selection": "four illustrative candidates with high-confidence detections agreeing with dataset labels at IoU >= 0.45; diversity prioritized",
        "selected": selected,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
