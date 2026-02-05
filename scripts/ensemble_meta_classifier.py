# -*- coding: utf-8 -*-
"""
模型融合：片段模型 + 层级模型 元分类器

最小侵入：不改训练器，仅加载已训练模型，
基于被试级概率做拼接特征，训练一个小的元分类器。
"""

import os
import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 项目路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import UnifiedConfig
from src.models.dl_trainer import DeepLearningTrainer
from src.models.segment_trainer import SegmentTrainer
from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset, segment_collate_fn
from experiments.dl_transformer_experiment import LightweightGazeDataset
from src.data.split_strategy import TwoByTwoSplitter


def load_processed_data(data_path: str) -> List[Dict]:
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def load_model_weights(trainer, model_path: str) -> None:
    """加载模型权重（按形状匹配过滤，避免结构改动导致的不兼容）"""
    checkpoint = torch.load(model_path, map_location=trainer.device, weights_only=False)
    trainer.model = trainer._create_model()

    model_state = trainer.model.state_dict()
    ckpt_state = checkpoint.get('model_state_dict', {})
    filtered = {
        k: v for k, v in ckpt_state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = trainer.model.load_state_dict(filtered, strict=False)
    print(f'[load_model_weights] loaded={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}')


def build_subject_label_map(data: List[Dict]) -> Dict[str, int]:
    label_map = {}
    for d in data:
        if 'category' in d:
            label_map[d['subject_id']] = int(d['category']) - 1
        else:
            # 回归任务不在此脚本支持
            label_map[d['subject_id']] = int(d['label'])
    return label_map


def split_train_val(train_data: List[Dict], seed: int = 42, val_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_data))
    val_size = max(1, int(len(train_data) * val_ratio))
    val_indices = set(indices[:val_size])
    train_split = [train_data[i] for i in indices[val_size:]]
    val_split = [train_data[i] for i in indices[:val_size]]
    return train_split, val_split


def get_hier_subject_probs(
    trainer: DeepLearningTrainer,
    data_split: List[Dict],
    seq_config: SequenceConfig,
) -> Dict[str, np.ndarray]:
    dataset = LightweightGazeDataset(
        data=data_split,
        config=seq_config,
        fit_normalizer=False,
        normalizer_stats=None,
        task_type=trainer.config.task.type,
        use_task_embedding=trainer.config.model.use_task_embedding,
    )
    preds, labels, _ = trainer.predict(dataset)
    # preds: (N, C) logits
    if preds.ndim == 1:
        preds = preds[:, None]
    probs = torch.softmax(torch.tensor(preds), dim=-1).numpy()

    subject_ids = [d['subject_id'] for d in data_split]
    return {sid: prob for sid, prob in zip(subject_ids, probs)}


def get_segment_subject_probs(
    trainer: SegmentTrainer,
    data_split: List[Dict],
    seq_config: SequenceConfig,
) -> Dict[str, np.ndarray]:
    dataset = SegmentGazeDataset.from_processed_data(data_split, seq_config)
    loader = DataLoader(
        dataset,
        batch_size=trainer.config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=segment_collate_fn,
    )

    trainer.model.eval()
    subject_probs: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(trainer.device, non_blocking=True)
            lengths = batch['length'].to(trainer.device, non_blocking=True)

            task_conditions = None
            if 'task_conditions' in batch and batch['task_conditions'] is not None:
                task_conditions = {k: v.to(trainer.device, non_blocking=True) for k, v in batch['task_conditions'].items()}

            outputs = trainer.model(features, lengths, task_conditions=task_conditions)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()

            for sid, p in zip(batch['subject_ids'], probs):
                subject_probs.setdefault(sid, []).append(p)

    # 聚合为被试级概率（平均）
    return {sid: np.mean(ps, axis=0) for sid, ps in subject_probs.items()}


def build_feature_matrix(
    subject_ids: List[str],
    hier_probs: Dict[str, np.ndarray],
    seg_probs: Dict[str, np.ndarray],
    label_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = []
    labels = []
    kept_ids = []
    for sid in subject_ids:
        if sid not in hier_probs or sid not in seg_probs or sid not in label_map:
            continue
        feat = np.concatenate([seg_probs[sid], hier_probs[sid]], axis=0)
        feats.append(feat)
        labels.append(label_map[sid])
        kept_ids.append(sid)
    return np.array(feats), np.array(labels), kept_ids


def evaluate_meta(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        'accuracy': float(accuracy_score(y, preds)),
        'f1': float(f1_score(y, preds, average='macro')),
    }


def main():
    parser = argparse.ArgumentParser(description='片段+层级元分类器融合')
    parser.add_argument('--data', type=str, default='data/processed/processed_data.pkl')
    parser.add_argument('--hier-config', type=str, required=True)
    parser.add_argument('--hier-model', type=str, required=True)
    parser.add_argument('--seg-config', type=str, required=True)
    parser.add_argument('--seg-model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = load_processed_data(args.data)
    label_map = build_subject_label_map(data)

    # split
    config = UnifiedConfig.from_json(args.hier_config)
    seq_config = SequenceConfig()
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )
    splits = splitter.split(data)
    train_data, val_data = split_train_val(splits['train'], seed=args.seed, val_ratio=0.2)

    # load hierarchical model
    hier_config = UnifiedConfig.from_json(args.hier_config)
    hier_trainer = DeepLearningTrainer(hier_config, seq_config)
    load_model_weights(hier_trainer, args.hier_model)

    # load segment model
    seg_config = UnifiedConfig.from_json(args.seg_config)
    seg_trainer = SegmentTrainer(seg_config, seq_config)
    load_model_weights(seg_trainer, args.seg_model)

    # probabilities
    hier_train = get_hier_subject_probs(hier_trainer, train_data, seq_config)
    hier_val = get_hier_subject_probs(hier_trainer, val_data, seq_config)
    hier_test1 = get_hier_subject_probs(hier_trainer, splits['test1'], seq_config)
    hier_test2 = get_hier_subject_probs(hier_trainer, splits['test2'], seq_config)
    hier_test3 = get_hier_subject_probs(hier_trainer, splits['test3'], seq_config)

    seg_train = get_segment_subject_probs(seg_trainer, train_data, seq_config)
    seg_val = get_segment_subject_probs(seg_trainer, val_data, seq_config)
    seg_test1 = get_segment_subject_probs(seg_trainer, splits['test1'], seq_config)
    seg_test2 = get_segment_subject_probs(seg_trainer, splits['test2'], seq_config)
    seg_test3 = get_segment_subject_probs(seg_trainer, splits['test3'], seq_config)

    # feature matrices
    train_ids = [d['subject_id'] for d in train_data]
    val_ids = [d['subject_id'] for d in val_data]
    test1_ids = [d['subject_id'] for d in splits['test1']]
    test2_ids = [d['subject_id'] for d in splits['test2']]
    test3_ids = [d['subject_id'] for d in splits['test3']]

    X_train, y_train, _ = build_feature_matrix(train_ids, hier_train, seg_train, label_map)
    X_val, y_val, _ = build_feature_matrix(val_ids, hier_val, seg_val, label_map)
    X_test1, y_test1, _ = build_feature_matrix(test1_ids, hier_test1, seg_test1, label_map)
    X_test2, y_test2, _ = build_feature_matrix(test2_ids, hier_test2, seg_test2, label_map)
    X_test3, y_test3, _ = build_feature_matrix(test3_ids, hier_test3, seg_test3, label_map)

    # meta classifier
    meta = LogisticRegression(max_iter=1000, class_weight='balanced')
    meta.fit(X_train, y_train)

    print('Meta Val:', evaluate_meta(meta, X_val, y_val))
    print('Meta Test1:', evaluate_meta(meta, X_test1, y_test1))
    print('Meta Test2:', evaluate_meta(meta, X_test2, y_test2))
    print('Meta Test3:', evaluate_meta(meta, X_test3, y_test3))


if __name__ == '__main__':
    main()
