# -*- coding: utf-8 -*-
"""Training loop for task-level random holdout runs."""

import logging
from pathlib import Path

import torch

from src.training.curves import plot_training_curves


FINAL_MODEL_FILENAME = "final_model.pt"
BEST_MODEL_FILENAME = "best_model.pt"
TRAINING_CURVES_FILENAME = "training_curves.png"
CHECKPOINT_POLICY_FINAL = "final_epoch_random_holdout"
CHECKPOINT_POLICY_BEST = "best_validation_random_holdout"
MISSING_BEST_METRIC = float("-inf")


logger = logging.getLogger(__name__)


def train_full_data(trainer, train_loader, val_loader, config: dict, output_dir: Path) -> dict:
    epochs = config["training"]["epochs"]
    history = init_history(val_loader is not None)
    state = init_validation_state(config, val_loader)
    final_train_metrics = {}
    final_val_metrics = None

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch, epochs)
        append_history(history, "train", train_metrics)
        final_train_metrics = metric_summary(train_metrics)

        if val_loader is None:
            trainer.scheduler.step(train_metrics["loss"])
            continue

        val_metrics = trainer.evaluate(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        append_history(history, "val", val_metrics)
        final_val_metrics = metric_summary(val_metrics)
        trainer.scheduler.step(val_metrics["loss"])
        update_validation_checkpoint(trainer, val_metrics, state, epoch, output_dir)
        if state["patience_counter"] >= state["patience"]:
            logger.info("Early stop at epoch %s.", epoch)
            break

    save_final_checkpoint(trainer, output_dir / FINAL_MODEL_FILENAME, epoch, final_train_metrics)
    plot_training_curves(history, output_dir / TRAINING_CURVES_FILENAME)
    return build_train_result(history, final_train_metrics, final_val_metrics, state)


def init_history(include_validation: bool) -> dict:
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "train_f1_macro": []}
    if include_validation:
        history.update({"val_loss": [], "val_acc": [], "val_f1": [], "val_f1_macro": []})
    return history


def init_validation_state(config: dict, val_loader) -> dict:
    if val_loader is None:
        return {"enabled": False}
    training = config["training"]
    if "early_stop_metric" not in training or "patience" not in training:
        raise ValueError("validation training requires early_stop_metric and patience")
    return {
        "enabled": True,
        "early_stop_metric": training["early_stop_metric"],
        "patience": training["patience"],
        "patience_counter": 0,
        "best_val_metric": MISSING_BEST_METRIC,
        "best_epoch": 0,
    }


def append_history(history: dict, prefix: str, metrics: dict) -> None:
    history[f"{prefix}_loss"].append(metrics["loss"])
    history[f"{prefix}_acc"].append(metrics["accuracy"])
    history[f"{prefix}_f1"].append(metrics["f1_weighted"])
    history[f"{prefix}_f1_macro"].append(metrics["f1_macro"])


def update_validation_checkpoint(trainer, metrics: dict, state: dict, epoch: int, output_dir: Path):
    current_metric = metrics[state["early_stop_metric"]]
    if current_metric <= state["best_val_metric"]:
        state["patience_counter"] += 1
        return

    state["best_val_metric"] = current_metric
    state["best_epoch"] = epoch
    state["patience_counter"] = 0
    save_best_checkpoint(trainer, output_dir / BEST_MODEL_FILENAME, epoch, current_metric)


def save_best_checkpoint(trainer, path: Path, epoch: int, best_metric: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "checkpoint_policy": CHECKPOINT_POLICY_BEST,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def save_final_checkpoint(trainer, path: Path, epoch: int, final_metrics: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "checkpoint_policy": CHECKPOINT_POLICY_FINAL,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "final_train_metrics": final_metrics,
        },
        path,
    )


def metric_summary(metrics: dict) -> dict:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_macro": metrics["f1_macro"],
    }


def build_train_result(
    history: dict,
    final_train_metrics: dict,
    final_val_metrics,
    state: dict,
) -> dict:
    result = {"history": history, "final_train_metrics": final_train_metrics}
    if final_val_metrics is not None:
        result["final_val_metrics"] = final_val_metrics
        result["best_val_metric"] = state["best_val_metric"]
        result["best_epoch"] = state["best_epoch"]
    return result
