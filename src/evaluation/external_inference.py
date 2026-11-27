"""Measure YOLO26m predictions on the external image and full rescue video.

No ground-truth annotations are available: detection counts are descriptive,
not precision, recall, AP, or counts of unique people.
"""

import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

from .paths import ROOT
OUTPUT = ROOT / "models/comparisons/external-inference.json"
from .paths import EXTERNAL_IMAGE
IMAGE = EXTERNAL_IMAGE
VIDEO = ROOT / "static/video/rescue_simulation.mp4"
WEIGHTS = ROOT / "models/yolov26m/weights/best.pt"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(samples):
    return {"samples": len(samples), "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)),
            "equivalent_images_per_second": 1000 / statistics.fmean(samples)}


def detections(result):
    return [{"class": result.names[int(b.cls[0])], "confidence": float(b.conf[0]),
             "bbox_xyxy": [float(x) for x in b.xyxy[0].tolist()]} for b in result.boxes]


def run():
    torch.set_num_threads(8)
    model = YOLO(str(WEIGHTS))

    def predict(image):
        return model.predict(image, conf=0.25, imgsz=640, device="cpu", verbose=False)[0]

    image = cv2.imread(str(IMAGE))
    if image is None:
        raise RuntimeError("Imagem externa indisponível")
    for _ in range(3):
        predict(image)
    samples = []
    for _ in range(30):
        start = time.perf_counter()
        result = predict(image)
        samples.append((time.perf_counter() - start) * 1000)
    image_result = {**stats(samples), "path": str(IMAGE.relative_to(ROOT)),
                    "sha256": digest(IMAGE), "samples_ms": samples,
                    "detections": detections(result), "shape_hw": list(image.shape[:2])}
    print("Imagem:", image_result["mean_ms"], "ms", flush=True)

    cap = cv2.VideoCapture(str(VIDEO))
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Vídeo sem quadros de aquecimento")
        predict(frame)
    cap.release()
    timing = []
    frames = []
    for repeat in range(3):
        cap = cv2.VideoCapture(str(VIDEO))
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                start = time.perf_counter()
                result = predict(frame)
                elapsed = (time.perf_counter() - start) * 1000
                timing.append({"repeat": repeat, "frame": index, "ms": elapsed})
                if repeat == 0:
                    frames.append({"frame": index, "timestamp_ms": timestamp_ms,
                                   "detections": detections(result)})
                index += 1
                if index % 64 == 0:
                    print(f"Vídeo: repetição {repeat + 1}/3, quadro {index}", flush=True)
        finally:
            cap.release()
        if index != len(frames):
            raise RuntimeError("Contagem de quadros diferente entre repetições")
    class_boxes = Counter()
    class_frames = Counter()
    for frame in frames:
        class_boxes.update(d["class"] for d in frame["detections"])
        class_frames.update({d["class"] for d in frame["detections"]})
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration", "-of", "json", str(VIDEO)
    ], text=True))["streams"][0]
    video_result = {**stats([row["ms"] for row in timing]),
                    "path": str(VIDEO.relative_to(ROOT)), "sha256": digest(VIDEO),
                    "metadata": metadata, "unique_frames": len(frames), "repeats": 3,
                    "frames_with_detection": sum(bool(f["detections"]) for f in frames),
                    "frames_per_class": dict(class_frames), "boxes_per_class": dict(class_boxes),
                    "prediction_time_per_pass_s": [sum(r["ms"] for r in timing if r["repeat"] == i) / 1000 for i in range(3)],
                    "timings": timing, "frames": frames}
    cpu_names = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")]
    payload = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
               "model": str(WEIGHTS.relative_to(ROOT)), "checkpoint_sha256": digest(WEIGHTS),
               "device": "cpu", "cpu": cpu_names[0] if cpu_names else platform.machine(),
               "platform": platform.platform(), "python_version": platform.python_version(),
               "torch_version": torch.__version__, "ultralytics_version": ultralytics_version,
               "torch_num_threads": torch.get_num_threads(), "confidence": 0.25, "imgsz": 640,
               "warmup_per_source": 3,
               "scope": "predict call: preprocessing, network and postprocessing; excludes decoding, loading, drawing and encoding",
               "image": image_result, "video": video_result}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("Salvo:", OUTPUT, flush=True)
    print("Vídeo:", stats([r["ms"] for r in timing]), flush=True)
    print("Quadros/classes:", video_result["frames_with_detection"], dict(class_frames), flush=True)


if __name__ == "__main__":
    run()
