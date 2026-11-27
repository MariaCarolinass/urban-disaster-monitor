"""Plot complete YOLO11/YOLO26 validation histories using the publication style."""

from pathlib import Path

import csv
import io
import subprocess

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from ..paths import ROOT, FIGURES_DIR
from .style import COLORS, publication_style as style
from ..results import yolo_row


def generate(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    style()
    selected = {}
    for family in ("YOLO11", "YOLO26"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        for index, variant in enumerate("nsml"):
            if family == "YOLO11":
                content = subprocess.check_output([
                    "git", "show", f"yolo11-v1.0.0:models/yolov11{variant}/results.csv"
                ], cwd=ROOT, text=True)
            else:
                content = (ROOT / f"models/yolov26{variant}/results.csv").read_text()
            rows = list(csv.DictReader(io.StringIO(content)))
            assert [int(row["epoch"]) for row in rows] == list(range(1, 51))
            best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
            selected[family + variant] = int(best["epoch"])
            assert selected[family + variant] == yolo_row(family, variant)["epoch"]
            for ax, metric in zip(axes, ("metrics/mAP50(B)", "metrics/mAP50-95(B)")):
                ax.plot([int(r["epoch"]) for r in rows], [float(r[metric]) for r in rows],
                        color=COLORS[index], linestyle=("-", "--", "-.", ":")[index],
                        linewidth=1.4, label=family + variant)
                ax.scatter(int(best["epoch"]), float(best[metric]), marker="*", s=110,
                           color=COLORS[index], edgecolors="black", linewidths=0.5, zorder=5)
        for ax, metric in zip(axes, ("mAP@0.5", "mAP@0.5:0.95")):
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_xticks([1, 10, 20, 30, 40, 50])
            ax.set_xlim(0, 51)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
            ax.grid(color="#D9DEE4", alpha=0.7, linewidth=0.7)
            ax.set_axisbelow(True)
        axes[0].set_title("(a) mAP@0.5", fontsize=10)
        axes[1].set_title("(b) mAP@0.5:0.95", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], marker="*", linestyle="None", color="white",
                              markeredgecolor="black", markersize=11))
        labels.append("Epoch selected by mAP@0.5")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2)
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"evolucao_{family.lower()}_validacao.{extension}", dpi=300,
                        bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
    return selected


if __name__ == "__main__":
    print("Selected epochs:", generate())
