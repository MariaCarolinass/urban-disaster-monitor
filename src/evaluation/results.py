"""Read saved model training artifacts."""
import csv
import io
import subprocess
from .paths import ROOT

VARIANTS = "nsml"
CLASSES = ["cat", "civilian", "cow", "dog", "horse", "rescuer"]
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def yolo_row(family: str, variant: str) -> dict:
    if family == "YOLO11":
        path = f"models/yolov11{variant}/results.csv"
        content = subprocess.check_output(
            ["git", "show", f"yolo11-v1.0.0:{path}"], cwd=ROOT, text=True
        )
        rows = list(csv.DictReader(io.StringIO(content)))
    else:
        rows = list(csv.DictReader((ROOT / f"models/yolov26{variant}/results.csv").open()))
    row = max(rows, key=lambda r: float(r[METRICS["map50"]]))
    result = {key: float(row[column]) for key, column in METRICS.items()}
    result["epoch"] = int(row["epoch"])
    result["time_min"] = float(rows[-1]["time"]) / 60 if family == "YOLO26" else None
    result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    return result

def rfdetr_row(variant: str) -> dict:
    rows = list(csv.DictReader((ROOT / f"models/rfdetr{variant}/metrics.csv").open()))
    row = max((r for r in rows if r["val/mAP_50"]), key=lambda r: float(r["val/mAP_50"]))
    result = {
        "epoch": int(row["epoch"]),
        "precision": float(row["val/precision"]),
        "recall": float(row["val/recall"]),
        "f1": float(row["val/F1"]),
        "map50": float(row["val/mAP_50"]),
        "map50_95": float(row["val/mAP_50_95"]),
        "ap_class": [float(row[f"val/AP/{name}"]) for name in CLASSES],
    }
    return result
