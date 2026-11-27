"""Shared, print-ready styling for evaluation notebook figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

VARIANT_COLORS = {
    "n": "#0072B2",
    "s": "#009E73",
    "m": "#D55E00",
    "l": "#CC79A7",
}
BASELINE_COLOR = "#0072B2"
CUSTOM_COLOR = "#D55E00"
RF_COLORS = {"rfdetrnano": "#4D4D4D", "rfdetrmedium": "#8064A2"}

# Palette used by the publication profile.
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def configure_figures():
    """Apply a readable style at the figure's final publication size."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def figure(width_cm=16, height_cm=8.5, **kwargs):
    return plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54), **kwargs)


def decimal(value, places=3):
    return f"{value:.{places}f}"


def integer(value):
    return f"{value:,.0f}"


def score_axis(ax, orientation="y"):
    ticks = np.linspace(0, 1, 6)
    if orientation == "y":
        ax.set_ylim(0, 1.05)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal(value, 1)))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    else:
        ax.set_xlim(0, 1.05)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal(value, 1)))
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)


def finish(fig, project_root, name):
    """Save the figure as an editable PDF and a 300 dpi PNG."""
    fig.tight_layout()
    output_dir = Path(project_root) / "notebooks" / "generated" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    return output_dir / f"{name}.pdf"


def publication_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
