# -*- coding: utf-8 -*-
"""Full-data task-level training loader tests."""

import json
import unittest
import sys
from pathlib import Path

from torch.utils.data import WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.full_data import FullDataLoaderConfig, build_full_data_train_loader


class TinyDataset:
    def __init__(self, labels):
        self.samples = [
            {"label": label, "subject_id": index, "task_id": index}
            for index, label in enumerate(labels)
        ]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def keep_batch(batch):
    return batch


class FullDataTrainingLoaderTest(unittest.TestCase):
    def test_uses_every_sample_for_training(self):
        dataset = TinyDataset(labels=[0, 1, 2, 1, 0])
        options = FullDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            use_balanced_sampler=False,
        )

        result = build_full_data_train_loader(dataset, options, keep_batch)

        self.assertEqual(result.train_indices, (0, 1, 2, 3, 4))
        self.assertEqual(len(result.train_loader.dataset), len(dataset))
        self.assertNotIsInstance(result.train_loader.sampler, WeightedRandomSampler)

    def test_balanced_sampler_covers_all_samples(self):
        dataset = TinyDataset(labels=[0, 1, 1, 1, 2])
        options = FullDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            use_balanced_sampler=True,
        )

        result = build_full_data_train_loader(dataset, options, keep_batch)

        self.assertIsInstance(result.train_loader.sampler, WeightedRandomSampler)
        self.assertEqual(result.train_loader.sampler.num_samples, len(dataset))
        self.assertEqual(result.train_indices, tuple(range(len(dataset))))

    def test_full_data_config_keeps_best_architecture(self):
        config_path = PROJECT_ROOT / "configs" / "task_level_ordinal_manual11_bilstm_full_data.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = config["model"]

        self.assertEqual(model_config["head_type"], "ordinal")
        self.assertEqual(model_config["segment_encoder_type"], "bilstm")
        self.assertTrue(model_config["use_task_embedding"])

    def test_full_data_config_declares_no_validation_split(self):
        config_path = PROJECT_ROOT / "configs" / "task_level_ordinal_manual11_bilstm_full_data.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["split"]["mode"], "full_data")
        self.assertEqual(config["split"]["train_fraction"], 1.0)
        self.assertEqual(config["split"]["validation_fraction"], 0.0)
        self.assertNotIn("train_val_split", config["training"])
        self.assertNotIn("patience", config["training"])
        self.assertNotIn("early_stop_metric", config["training"])

    def test_full_data_script_does_not_use_2x2_split(self):
        script_path = PROJECT_ROOT / "scripts" / "train_task_level_full_data.py"
        runner_path = PROJECT_ROOT / "src" / "training" / "task_level_full_data_runner.py"

        script_text = script_path.read_text(encoding="utf-8")
        runner_text = runner_path.read_text(encoding="utf-8")
        combined_text = script_text + runner_text

        self.assertIn("build_full_data_train_loader", combined_text)
        self.assertNotIn("split_2x2", combined_text)


if __name__ == "__main__":
    unittest.main()
