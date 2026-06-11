# -*- coding: utf-8 -*-
"""Strict model loading and prediction for new-data inference."""

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.inference.new_data_dataset import (
    NewDataInferenceDataset,
    inference_collate_fn,
)
from src.inference.new_data_types import InferenceSample, PredictionResult
from src.models.losses import decode_ordinal_logits, ordinal_probs_to_class_probs
from src.models.task_level_model import TaskLevelEncoder


CLASS_LABELS = ("低", "中", "高")
EXPECTED_CLASS_COUNT = 3
ORDINAL_HEAD = "ordinal"


def load_inference_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TaskLevelEncoder, Dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, config_path)
    model = TaskLevelEncoder(**_model_options(config)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path}: checkpoint 缺少 model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def predict_samples(
    model: TaskLevelEncoder,
    config: Dict,
    samples: Sequence[InferenceSample],
    device: torch.device,
) -> Tuple[PredictionResult, ...]:
    sequence = config["sequence"]
    dataset = NewDataInferenceDataset(
        samples=samples,
        max_seq_len=sequence["max_seq_len"],
        input_dim=sequence["input_dim"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=inference_collate_fn,
    )
    results = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            logits = _forward_batch(model, batch, device)
            predictions, probabilities = decode_ordinal_predictions(logits, config)
            for index in range(len(predictions)):
                results.append(
                    _prediction_result(
                        samples[offset + index],
                        predictions[index],
                        probabilities[index],
                    )
                )
            offset += len(predictions)
    return tuple(results)


def decode_ordinal_predictions(
    logits: torch.Tensor,
    config: Dict,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[float, float, float], ...]]:
    thresholds = torch.tensor(
        config["training"]["ordinal_thresholds"],
        dtype=logits.dtype,
        device=logits.device,
    )
    predictions = decode_ordinal_logits(logits, thresholds)
    probabilities = ordinal_probs_to_class_probs(torch.sigmoid(logits))
    prediction_values = tuple(int(value) for value in predictions.cpu().tolist())
    probability_values = tuple(
        tuple(float(value) for value in row)
        for row in probabilities.cpu().tolist()
    )
    return prediction_values, probability_values


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境不可用")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求了 MPS，但当前环境不可用")
    return device


def _validate_config(config: Dict, config_path: Path) -> None:
    for section in ("sequence", "model", "training"):
        if section not in config:
            raise ValueError(f"{config_path}: 缺少 {section} 配置")
    model_config = config["model"]
    if model_config.get("head_type") != ORDINAL_HEAD:
        raise ValueError(f"{config_path}: 仅支持 ordinal 推理模型")
    if model_config.get("num_classes") != EXPECTED_CLASS_COUNT:
        raise ValueError(f"{config_path}: num_classes 必须为3")
    thresholds = config["training"].get("ordinal_thresholds")
    if not isinstance(thresholds, list) or len(thresholds) != EXPECTED_CLASS_COUNT - 1:
        raise ValueError(f"{config_path}: ordinal_thresholds 必须包含2个值")


def _model_options(config: Dict) -> Dict:
    sequence = config["sequence"]
    model = config["model"]
    return {
        "input_dim": sequence["input_dim"],
        "max_seq_len": sequence["max_seq_len"],
        "max_segments": sequence["max_segments"],
        "segment_d_model": model["segment_d_model"],
        "segment_nhead": model["segment_nhead"],
        "segment_num_layers": model["segment_num_layers"],
        "segment_encoder_type": model["segment_encoder_type"],
        "segment_rnn_hidden_size": model.get(
            "segment_rnn_hidden_size", model["segment_d_model"]
        ),
        "segment_rnn_layers": model.get("segment_rnn_layers", 1),
        "segment_rnn_dropout": model.get("segment_rnn_dropout", 0.0),
        "segment_cnn_channels": model.get("segment_cnn_channels"),
        "segment_cnn_kernel_sizes": model.get("segment_cnn_kernel_sizes"),
        "attention_dim": model["attention_dim"],
        "task_embedding_dim": model["task_embedding_dim"],
        "use_task_embedding": model["use_task_embedding"],
        "task_embedding_type": model.get("task_embedding_type", "independent"),
        "use_conditional_pooling": model.get("use_conditional_pooling", False),
        "dropout": model["dropout"],
        "num_classes": model["num_classes"],
        "head_type": model["head_type"],
        "use_gradient_checkpointing": model.get(
            "use_gradient_checkpointing", False
        ),
    }


def _forward_batch(model, batch: Dict, device: torch.device) -> torch.Tensor:
    return model(
        batch["segments"].to(device),
        batch["segment_mask"].to(device),
        batch["task_conditions"].to(device),
        batch["segment_seq_mask"].to(device),
    )


def _prediction_result(
    sample: InferenceSample,
    prediction: int,
    probabilities: Tuple[float, float, float],
) -> PredictionResult:
    return PredictionResult(
        sample=sample,
        predicted_index=prediction,
        predicted_label=CLASS_LABELS[prediction],
        probabilities=probabilities,
    )
