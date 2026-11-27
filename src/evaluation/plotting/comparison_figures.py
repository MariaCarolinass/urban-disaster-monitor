"""Regenerate comparison figures from saved metrics; no LaTeX dependency."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..paths import FIGURES_DIR, ROOT
from ..results import CLASSES, VARIANTS, rfdetr_row, yolo_row
from .style import (
    BASELINE_COLOR,
    COLORS,
    CUSTOM_COLOR,
    VARIANT_COLORS,
    decimal,
    figure,
    finish,
    publication_style,
    score_axis,
)

CLASS_LABELS = {
    "cat": "Cat",
    "civilian": "Civilian",
    "cow": "Cow",
    "dog": "Dog",
    "horse": "Horse",
    "rescuer": "Rescuer",
}


def save(fig, name: str, output_dir=None) -> None:
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def bars(name: str, labels: list[str], values: list[float], ylabel: str = "mAP@0.5", output_dir=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    rects = ax.bar(labels, values, width=0.65, color=COLORS[:len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    ax.grid(axis="y", color="#D9DEE4", alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.bar_label(rects, labels=[fmt(v) for v in values], padding=3, fontsize=9)
    save(fig, name, output_dir)


def grouped(name: str, labels: list[str], series: dict[str, list[float]], ylabel: str,
            figsize: tuple[int, int] = (8, 4), precision: int = 4, output_dir=None) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = (0.90 if len(series) == 2 else 0.76) / len(series)
    for i, (title, values) in enumerate(series.items()):
        rects = ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width,
                       label=title, color=COLORS[i])
        ax.bar_label(rects, labels=[f"{v:.{precision}f}" for v in values],
                     padding=2, fontsize=9, rotation=90 if len(series) >= 3 else 0)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.13 if len(series) >= 3 else 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(series), 4), frameon=False)
    ax.grid(axis="y", color="#D9DEE4", alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    save(fig, name, output_dir)


def generate(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    publication_style()
    y11 = {v: yolo_row("YOLO11", v) for v in VARIANTS}
    y26 = {v: yolo_row("YOLO26", v) for v in VARIANTS}
    rf = {v: rfdetr_row(v) for v in ("nano", "medium")}
    validation = json.loads((ROOT / "models/comparisons/checkpoint-validation.json").read_text())
    comparison = json.loads((ROOT / "models/comparisons/coco-vs-custom-yolo26m-common-test.json").read_text())
    for name, row in validation["models"].items():
        if name.startswith("yolo") or name == "custom_yolo26m":
            assert row["classes"] == list(range(len(CLASSES)))
            assert len(row["ap50"]) == len(CLASSES)
        assert all(0 <= value <= 1 for value in row["ap50"])
    y11map = [y11[v]['map50'] for v in VARIANTS]
    y26map = [y26[v]['map50'] for v in VARIANTS]
    bars('comparacao_yolo11_map50.png', [f'YOLO11{v}' for v in VARIANTS], y11map, output_dir=output_dir)
    bars('comparacao_yolo26_map50.png', [f'YOLO26{v}' for v in VARIANTS], y26map, output_dir=output_dir)
    grouped('comparacao_yolo11_classes.png', CLASSES, {f'YOLO11{v}': validation['models'][f'yolo11{v}']['ap50'] for v in VARIANTS}, 'AP@0.5', (8, 4.2), 3, output_dir=output_dir)
    grouped('comparacao_yolo26_classes.png', CLASSES, {f'YOLO26{v}': validation['models'][f'yolo26{v}']['ap50'] for v in VARIANTS}, 'AP@0.5', (8, 4.2), 3, output_dir=output_dir)
    grouped('comparacao_yolo11_yolo26.png', ['Nano', 'Small', 'Medium', 'Large'], {'YOLO11': y11map, 'YOLO26': y26map}, 'mAP@0.5', output_dir=output_dir)
    grouped('comparacao_rfdetr.png', ['Precision', 'Recall', 'F1-score', 'mAP@0.5'], {f'RF-DETR {name.title()}': [rf[name][k] for k in ('precision', 'recall', 'f1', 'map50')] for name in ('nano', 'medium')}, 'Metric value', output_dir=output_dir)
    grouped('comparacao_rfdetr_classes.png', CLASSES, {f'RF-DETR {name.title()}': rf[name]['ap_class'] for name in ('nano', 'medium')}, 'AP@0.5:0.95', precision=3, output_dir=output_dir)
    grouped('comparacao_rfdetr_yolo26m_map.png', ['mAP@0.5', 'mAP@0.5:0.95'], {'YOLO26m': [y26['m'][k] for k in ('map50', 'map50_95')], 'RF-DETR Medium': [rf['medium'][k] for k in ('map50', 'map50_95')]}, 'Metric value', (7, 4), output_dir=output_dir)
    grouped('comparacao_rfdetr_yolo26m_pr.png', ['Precision', 'Recall'], {'YOLO26m': [y26['m'][k] for k in ('precision', 'recall')], 'RF-DETR Medium': [rf['medium'][k] for k in ('precision', 'recall')]}, 'Metric value', (7, 4), output_dir=output_dir)
    from ..coco_comparison import plot_comparison
    plt.close(plot_comparison(comparison, output_dir=output_dir, filename="customyolo26"))
    return output_dir


def plot_variant_map50(variants, values, project_root, name="yolov26-map50"):
    fig, ax = figure()
    bars = ax.bar([f"YOLOv26{v}" for v in variants], values,
                  color=[VARIANT_COLORS[v] for v in variants], width=0.65)
    ax.bar_label(bars, labels=[decimal(value) for value in values], padding=3)
    ax.set_ylabel("mAP@0.5 (0–1)")
    score_axis(ax)
    finish(fig, project_root, name)
    return fig


def plot_per_class_map50(classes, variants, scores_by_variant, project_root):
    fig, ax = figure(height_cm=10.5)
    positions = np.arange(len(classes))
    height = 0.75 / len(variants)
    for index, variant in enumerate(variants):
        scores = [np.nan if value is None else value
                  for value in scores_by_variant[variant]]
        ax.barh(positions + (index - (len(variants) - 1) / 2) * height,
                scores, height, label=f"YOLOv26{variant}",
                color=VARIANT_COLORS[variant])
    ax.set_yticks(positions, [CLASS_LABELS.get(name, name) for name in classes])
    ax.invert_yaxis()
    ax.set_xlabel("mAP@0.5 (0–1)")
    score_axis(ax, orientation="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
              frameon=False)
    finish(fig, project_root, "yolov26-map50-by-class")
    return fig


def plot_coco_custom(metric_labels, custom_values, coco_values, project_root,
                     name="coco-vs-custom-yolov26m"):
    positions = np.arange(len(metric_labels))
    width = 0.36
    fig, ax = figure(height_cm=9)
    baseline = ax.bar(positions - width / 2, coco_values, width,
                      label="COCO-pretrained YOLO26m", color=BASELINE_COLOR)
    custom = ax.bar(positions + width / 2, custom_values, width,
                    label="Custom YOLO26m", color=CUSTOM_COLOR)
    for bars, values in ((baseline, coco_values), (custom, custom_values)):
        ax.bar_label(bars, labels=[decimal(value) for value in values],
                     padding=3, fontsize=8)
    ax.set_xticks(positions, metric_labels)
    ax.set_ylabel("Metric value (0–1)")
    score_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False)
    finish(fig, project_root, name)
    return fig


def plot_historical_map50(variants, yolo11, yolo26, project_root):
    positions = np.arange(len(variants))
    width = 0.36
    fig, ax = figure(height_cm=9)
    old = ax.bar(positions - width / 2, yolo11, width,
                 label="YOLO11", color=BASELINE_COLOR)
    new = ax.bar(positions + width / 2, yolo26, width,
                 label="YOLO26", color=CUSTOM_COLOR)
    for bars, values in ((old, yolo11), (new, yolo26)):
        ax.bar_label(bars, labels=[decimal(value) for value in values],
                     padding=3, fontsize=8)
    ax.set_xticks(positions, [dict(n="Nano", s="Small", m="Medium", l="Large")[v]
                              for v in variants])
    ax.set_ylabel("mAP@0.5 (0–1)")
    score_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False)
    finish(fig, project_root, "yolo11-vs-yolo26-map50")
    return fig


if __name__ == "__main__":
    print(generate())
