"""Reevaluate best checkpoints for per-class and COCO comparison figures.

This CPU job can take several minutes. Use the resulting cache to generate comparison figures.
The resulting JSON is an auditable cache of the exact values plotted.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO, __version__ as ultralytics_version

from .paths import ROOT
OUTPUT = ROOT / "models/comparisons/checkpoint-validation.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    models = json.loads(OUTPUT.read_text()).get("models", {}) if OUTPUT.exists() else {}
    with tempfile.TemporaryDirectory(prefix="udm-yolo11-") as temp:
        temp = Path(temp)
        checkpoints = {}
        for variant in "nsml":
            weights = temp / f"yolo11{variant}.pt"
            with weights.open("wb") as target:
                subprocess.run(["git", "show", f"yolo11-v1.0.0:models/yolov11{variant}/weights/best.pt"],
                               cwd=ROOT, stdout=target, check=True)
            checkpoints[f"yolo11{variant}"] = (weights, ROOT / "dataset/data.yaml", None)
            checkpoints[f"yolo26{variant}"] = (ROOT / f"models/yolov26{variant}/weights/best.pt", ROOT / "dataset/data.yaml", None)
        checkpoints["custom_yolo26m"] = (ROOT / "models/yolov26m/weights/best.pt", ROOT / "dataset/data.yaml", None)
        checkpoints["coco_pretrained"] = (ROOT / "notebooks/generated/weights/yolo26m.pt",
                                           ROOT / "notebooks/generated/dataset_filtered/data_filtered.yaml",
                                           [0, 15, 16, 17, 19])
        for name, (weights, data, classes) in checkpoints.items():
            if name in models:
                continue
            kwargs = dict(data=str(data), split="val", imgsz=640, batch=16, workers=0,
                          plots=False, verbose=False, device="cpu", project="/tmp/udm-validation-runs",
                          name=name, exist_ok=True)
            if classes is not None:
                kwargs["classes"] = classes
            box = YOLO(str(weights)).val(**kwargs).box
            models[name] = {
                "precision": float(box.mp), "recall": float(box.mr),
                "map50": float(box.map50), "map50_95": float(box.map),
                "classes": np.asarray(box.ap_class_index, dtype=int).tolist(),
                "ap50": np.asarray(box.all_ap)[:, 0].tolist(),
                "weights": (f"yolo11-v1.0.0:models/yolov11{name[-1]}/weights/best.pt"
                            if name.startswith("yolo11") else str(weights.relative_to(ROOT))),
                "data": str(data.relative_to(ROOT)),
                "checkpoint_sha256": sha256(weights),
            }
            payload = {"protocol": "Ultralytics val, split=val, imgsz=640, batch=16, CPU",
                       "ultralytics_version": ultralytics_version, "dataset_images": 383,
                       "class_order": ["cat", "civilian", "cow", "dog", "horse", "rescuer"],
                       "coco_class_ids": [0, 15, 16, 17, 19], "models": models}
            OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
            print(name, models[name]["map50"], flush=True)


if __name__ == "__main__":
    run()
