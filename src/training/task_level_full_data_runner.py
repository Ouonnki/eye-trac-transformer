# -*- coding: utf-8 -*-
"""Full-data task-level training runner."""

import json
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.augmentation import GazeAugmentation
from src.models.task_level_dataset import (
    TaskLevelGazeDataset,
    TaskLevelSequenceConfig,
    task_level_collate_fn,
)
from src.models.task_level_model import TaskLevelEncoder
from src.models.task_level_trainer import TaskLevelTrainer
from src.training.full_data import FullDataLoaderConfig, build_full_data_train_loader
from src.training.full_data import build_class_weights, build_ordinal_pos_weight
from src.training.task_level_full_data_loop import train_full_data


CONFIG_FILENAME = "config.json"
HISTORY_FILENAME = "history.json"
TRAIN_RESULTS_FILENAME = "train_results.json"
ORDINAL_HEAD = "ordinal"
RANDOM_HOLDOUT_MODE = "random_holdout"
ALL_SAMPLES_SCOPE = "all_samples"
SAMPLE_UNIT = "sample"
NO_TEST_FRACTION = 0.0
SPLIT_TOLERANCE = 1e-9
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
DEFAULT_NUM_WORKERS = 0
DEFAULT_BALANCED_MODE = "effective_num"
DEFAULT_BALANCED_BETA = 0.999
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_LABEL_SMOOTHING = 0.0


logger = logging.getLogger(__name__)


def run_full_data_training(
    config_path: Path,
    output_dir_override: Optional[str] = None,
) -> Path:
    config = load_config(config_path)
    validate_full_data_split(config)
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override

    set_seed(config["experiment"]["random_seed"])
    output_dir = create_output_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_data = load_processed_data(config)
    dataset = build_dataset(config, processed_data)
    loader_result = build_full_data_train_loader(
        dataset,
        build_loader_config(config),
        task_level_collate_fn,
    )
    configure_augmentation(dataset, config, loader_result.train_indices)
    train_labels = collect_labels(dataset, loader_result.train_indices)

    model = build_model(config, device)
    trainer = build_trainer(model, device, config, train_labels)
    train_result = train_full_data(
        trainer,
        loader_result.train_loader,
        loader_result.val_loader,
        config,
        output_dir,
    )

    save_json(output_dir / CONFIG_FILENAME, config)
    save_json(output_dir / HISTORY_FILENAME, train_result["history"])
    save_json(output_dir / TRAIN_RESULTS_FILENAME, train_result)
    logger.info("Training finished. Results saved to %s", output_dir)
    return output_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def validate_full_data_split(config: dict) -> None:
    split = config.get("split")
    if split is None:
        raise ValueError("training config requires split settings")
    if split.get("mode") != RANDOM_HOLDOUT_MODE:
        raise ValueError("training only supports split.mode=random_holdout")
    if split.get("scope") != ALL_SAMPLES_SCOPE or split.get("unit") != SAMPLE_UNIT:
        raise ValueError("random holdout split must use all_samples/sample")
    if split.get("test_fraction") != NO_TEST_FRACTION:
        raise ValueError("random holdout training requires split.test_fraction=0.0")
    total = split["train_fraction"] + split["validation_fraction"]
    if abs(total - 1.0) > SPLIT_TOLERANCE:
        raise ValueError("train_fraction + validation_fraction must equal 1.0")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def create_output_dir(config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_config = config["experiment"]
    output_dir = Path(exp_config["output_dir"]) / f"{exp_config['name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_processed_data(config: dict):
    data_path = Path(config["data"]["processed_data_path"])
    with data_path.open("rb") as file_obj:
        return pickle.load(file_obj)


def build_dataset(config: dict, processed_data) -> TaskLevelGazeDataset:
    seq_config = TaskLevelSequenceConfig(
        max_seq_len=config["sequence"]["max_seq_len"],
        max_segments=config["sequence"]["max_segments"],
        screen_width=config["sequence"].get("screen_width", DEFAULT_SCREEN_WIDTH),
        screen_height=config["sequence"].get("screen_height", DEFAULT_SCREEN_HEIGHT),
        input_dim=config["sequence"]["input_dim"],
    )
    return TaskLevelGazeDataset(
        processed_data=processed_data,
        config=seq_config,
        fit_normalizer=False,
    )


def collect_labels(dataset: TaskLevelGazeDataset, indices) -> np.ndarray:
    return np.asarray([dataset.samples[index]["label"] for index in indices])


def configure_augmentation(dataset: TaskLevelGazeDataset, config: dict, train_indices) -> None:
    aug_config = config.get("augmentation", {})
    if not aug_config:
        logger.info("No augmentation configured.")
        return

    dataset.augmentation = GazeAugmentation(aug_config)
    dataset.training_indices = set(train_indices)
    logger.info("Augmentation enabled for all training samples.")


def build_loader_config(config: dict) -> FullDataLoaderConfig:
    training = config["training"]
    return FullDataLoaderConfig(
        batch_size=training["batch_size"],
        num_workers=training.get("num_workers", DEFAULT_NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        use_balanced_sampler=training.get("use_balanced_sampler", False),
        validation_fraction=config["split"]["validation_fraction"],
        random_seed=config["experiment"]["random_seed"],
        balanced_sampler_mode=training.get("balanced_sampler_mode", DEFAULT_BALANCED_MODE),
        balanced_sampler_beta=training.get("balanced_sampler_beta", DEFAULT_BALANCED_BETA),
    )


def build_model(config: dict, device: torch.device) -> TaskLevelEncoder:
    sequence = config["sequence"]
    model_config = config["model"]
    segment_hidden = model_config.get(
        "segment_rnn_hidden_size",
        model_config["segment_d_model"],
    )
    return TaskLevelEncoder(
        input_dim=sequence["input_dim"],
        max_seq_len=sequence["max_seq_len"],
        max_segments=sequence["max_segments"],
        segment_d_model=model_config["segment_d_model"],
        segment_nhead=model_config["segment_nhead"],
        segment_num_layers=model_config["segment_num_layers"],
        segment_encoder_type=model_config["segment_encoder_type"],
        segment_rnn_hidden_size=segment_hidden,
        segment_rnn_layers=model_config.get("segment_rnn_layers", 1),
        segment_rnn_dropout=model_config.get("segment_rnn_dropout", 0.0),
        attention_dim=model_config["attention_dim"],
        task_embedding_dim=model_config["task_embedding_dim"],
        use_task_embedding=model_config["use_task_embedding"],
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
        head_type=model_config.get("head_type", "classification"),
    ).to(device)


def build_trainer(model, device: torch.device, config: dict, train_labels: np.ndarray):
    training = config["training"]
    model_config = config["model"]
    class_weights = build_class_weights(training, train_labels)
    ordinal_pos_weight = None
    if model_config.get("head_type", "classification") == ORDINAL_HEAD:
        ordinal_pos_weight = build_ordinal_pos_weight(
            training,
            train_labels,
            model_config["num_classes"],
        )

    return TaskLevelTrainer(
        model=model,
        device=device,
        num_classes=model_config["num_classes"],
        head_type=model_config.get("head_type", "classification"),
        class_weights=class_weights,
        lr=training["lr"],
        weight_decay=training["weight_decay"],
        grad_clip=training["grad_clip"],
        use_focal_loss=training.get("use_focal_loss", False),
        focal_loss_alpha=training.get("focal_loss_alpha"),
        focal_loss_gamma=training.get("focal_loss_gamma", DEFAULT_FOCAL_GAMMA),
        label_smoothing=training.get("label_smoothing", DEFAULT_LABEL_SMOOTHING),
        ordinal_thresholds=training.get("ordinal_thresholds"),
        ordinal_pos_weight=ordinal_pos_weight,
    )
