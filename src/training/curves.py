# -*- coding: utf-8 -*-
"""Training curve plotting utilities."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURVE_DPI = 150
FIGURE_SIZE = (14, 10)
TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"


def plot_training_curves(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    _plot_metric(axes[0, 0], epochs, history, "loss", "Loss", ylim=None)
    _plot_metric(axes[0, 1], epochs, history, "acc", "Accuracy", ylim=(0, 1))
    _plot_metric(axes[1, 0], epochs, history, "f1", "F1 Weighted", ylim=(0, 1))
    _plot_metric(axes[1, 1], epochs, history, "f1_macro", "F1 Macro", ylim=(0, 1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=CURVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_metric(ax, epochs, history: dict, key: str, title: str, ylim) -> None:
    train_key = f"train_{key}"
    val_key = f"val_{key}"
    ax.plot(epochs, history[train_key], color=TRAIN_COLOR, linewidth=1.5, label="Train")
    if val_key in history and history[val_key]:
        ax.plot(epochs, history[val_key], color=VAL_COLOR, linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if ylim is not None:
        ax.set_ylim(*ylim)
