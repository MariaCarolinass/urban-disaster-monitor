"""Generate timing figures from recorded training and inference measurements."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ..paths import FIGURES_DIR, ROOT
from .style import COLORS, RF_COLORS, VARIANT_COLORS, decimal, figure, finish, integer, publication_style

NAMES = {**{f"yolov26{v}": f"YOLO26{v}" for v in "nsml"},
         "rfdetrnano": "RF-DETR Nano", "rfdetrmedium": "RF-DETR Medium"}


def generate(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    saved = json.loads((ROOT / "models/comparisons/model-inference-times.json").read_text())
    training = {}
    for variant in "nsml":
        rows = list(csv.DictReader((ROOT / f"models/yolov26{variant}/results.csv").open()))
        assert len(rows) == 50
        training[f"yolov26{variant}"] = float(rows[-1]["time"]) / 60
    for name, row in saved["models"].items():
        samples = row["samples_ms"]
        assert len(samples) == saved["samples_per_model"] == saved["images"] * saved["repeats"]
        for key, expected in (("mean_ms", statistics.fmean(samples)),
                              ("median_ms", statistics.median(samples)),
                              ("p95_ms", float(np.percentile(samples, 95)))):
            assert abs(row[key] - expected) < 1e-6, (name, key)
    publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("tempo_treinamento_yolo26", list(training), list(training.values()),
         COLORS, "Training duration (min)"),
        ("tempo_inferencia_cpu", list(NAMES),
         [saved["models"][name]["mean_ms"] for name in NAMES],
         [COLORS[0]] * 4 + [COLORS[1]] * 2, "Mean inference latency (ms/image)"),
    ]
    for filename, names, values, colors, ylabel in charts:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [NAMES[name].replace("RF-DETR ", "RF-DETR\n") for name in names]
        rects = ax.bar(labels, values, width=0.65, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.16)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(axis="y", color="#D9DEE4", alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.bar_label(rects, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{filename}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
    return {"training_minutes": training, "inference": saved}


def plot_training_duration(training, project_root):
    variants = [name[-1] for name, _ in training]
    values = [minutes for _, minutes in training]
    fig, ax = figure()
    bars = ax.bar([name for name, _ in training], values,
                  color=[VARIANT_COLORS[v] for v in variants], width=0.65)
    ax.bar_label(bars, labels=[decimal(value, 1) for value in values], padding=3)
    ax.set_ylabel("Training duration (min)")
    ax.set_ylim(bottom=0, top=max(values) * 1.13)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    finish(fig, project_root, "yolov26-training-duration")
    return fig


def plot_inference_times(saved, project_root):
    models = saved["models"]
    fig, axes = figure(height_cm=9, ncols=2)
    axis_max = max(data["mean_ms"] for data in models.values()) * 1.18
    for ax, family in zip(axes, ("YOLO", "RF-DETR")):
        rows = [(name, data["mean_ms"]) for name, data in models.items()
                if data["family"] == family]
        if family == "YOLO":
            rows.sort(key=lambda row: "nsml".index(row[0][-1]))
        else:
            rows.sort(key=lambda row: "nano" not in row[0])
        colors = ([VARIANT_COLORS[name[-1]] for name, _ in rows] if family == "YOLO"
                  else [RF_COLORS[name] for name, _ in rows])
        labels = ([f"YOLOv26{name[-1]}" for name, _ in rows]
                  if family == "YOLO" else
                  ["RF-DETR Nano" if "nano" in name else "RF-DETR Medium"
                   for name, _ in rows])
        bars = ax.barh(labels, [value for _, value in rows],
                       color=colors, height=0.65)
        ax.bar_label(bars, labels=[integer(value) for _, value in rows],
                     padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean inference latency (ms/image)")
        ax.set_title(family, fontsize=9.5)
        ax.set_xlim(left=0, right=axis_max)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: integer(value)))
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    finish(fig, project_root, "model-inference-times")
    return fig


if __name__ == "__main__":
    result = generate()
    for name in NAMES:
        row = result["inference"]["models"][name]
        print(NAMES[name], *(f"{row[k]:.1f}" for k in ("mean_ms", "median_ms", "p95_ms")))
