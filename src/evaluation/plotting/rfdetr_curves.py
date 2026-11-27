"""Plot the preserved RF-DETR validation history (epochs 40-49)."""

from pathlib import Path

import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from ..paths import ROOT, FIGURES_DIR
from .style import COLORS, publication_style as style


def generate(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    style()
    records = {}
    for variant in ("nano", "medium"):
        path = ROOT / f"models/rfdetr{variant}/metrics.csv"
        rows = [row for row in csv.DictReader(path.open()) if row.get("val/mAP_50")]
        epochs = [int(row["epoch"]) for row in rows]
        assert epochs == list(range(40, 50)), (variant, epochs)
        records[variant] = rows
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    selected = {}
    for index, (variant, rows) in enumerate(records.items()):
        chosen = max(rows, key=lambda r: float(r["val/mAP_50"]))
        selected[variant] = int(chosen["epoch"])
        for ax, metric in zip(axes, ("val/mAP_50", "val/mAP_50_95")):
            ax.plot([int(row["epoch"]) for row in rows], [float(row[metric]) for row in rows],
                    color=COLORS[index], marker=("o", "s")[index], markersize=4,
                    linewidth=1.5, label=f"RF-DETR {variant.title()}")
            ax.scatter(int(chosen["epoch"]), float(chosen[metric]), marker="*", s=150,
                       color=COLORS[index], edgecolors="black", linewidths=0.6, zorder=5)
    for ax, metric in zip(axes, ("mAP@0.5", "mAP@0.5:0.95")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_xticks(range(40, 50))
        ax.set_xlim(39.6, 49.4)
        ax.margins(y=0.16)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.grid(color="#D9DEE4", alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_title("(a) mAP@0.5", fontsize=10)
    axes[1].set_title("(b) mAP@0.5:0.95", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", linestyle="None", color="white",
                          markeredgecolor="black", markersize=11))
    labels.append("Epoch selected by mAP@0.5")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"evolucao_rfdetr_validacao.{extension}", dpi=300,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    assert selected == {"nano": 40, "medium": 43}, selected
    return selected


if __name__ == "__main__":
    print("Selected epochs:", generate())
