# -*- coding: utf-8 -*-
"""Full-data task-level training loader tests."""

import json
import tempfile
import unittest
import sys
from pathlib import Path

from torch.utils.data import WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.curves import plot_training_curves
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
            validation_fraction=0.0,
            random_seed=42,
        )

        result = build_full_data_train_loader(dataset, options, keep_batch)

        self.assertEqual(result.train_indices, (0, 1, 2, 3, 4))
        self.assertEqual(result.val_indices, ())
        self.assertEqual(len(result.train_loader.dataset), len(dataset))
        self.assertIsNone(result.val_loader)
        self.assertNotIsInstance(result.train_loader.sampler, WeightedRandomSampler)

    def test_validation_split_uses_holdout_without_2x2(self):
        dataset = TinyDataset(labels=[0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
        options = FullDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            use_balanced_sampler=False,
            validation_fraction=0.1,
            random_seed=42,
        )

        result = build_full_data_train_loader(dataset, options, keep_batch)

        self.assertEqual(len(result.train_indices), 9)
        self.assertEqual(len(result.val_indices), 1)
        self.assertEqual(set(result.train_indices) | set(result.val_indices), set(range(10)))
        self.assertEqual(set(result.train_indices) & set(result.val_indices), set())
        self.assertIsNotNone(result.val_loader)

    def test_balanced_sampler_covers_all_samples(self):
        dataset = TinyDataset(labels=[0, 1, 1, 1, 2])
        options = FullDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            use_balanced_sampler=True,
            validation_fraction=0.0,
            random_seed=42,
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

    def test_full_data_config_declares_random_validation_split(self):
        config_path = PROJECT_ROOT / "configs" / "task_level_ordinal_manual11_bilstm_full_data.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["split"]["mode"], "random_holdout")
        self.assertEqual(config["split"]["scope"], "all_samples")
        self.assertEqual(config["split"]["unit"], "sample")
        self.assertEqual(config["split"]["train_fraction"], 0.9)
        self.assertEqual(config["split"]["validation_fraction"], 0.1)
        self.assertEqual(config["training"]["early_stop_metric"], "f1_macro")
        self.assertEqual(config["training"]["patience"], 20)
        self.assertNotIn("train_val_split", config["training"])

    def test_curve_plot_writes_png(self):
        history = {
            "train_loss": [1.0, 0.8],
            "train_acc": [0.4, 0.6],
            "train_f1": [0.3, 0.5],
            "train_f1_macro": [0.2, 0.4],
            "val_loss": [1.1, 0.9],
            "val_acc": [0.35, 0.55],
            "val_f1": [0.25, 0.45],
            "val_f1_macro": [0.15, 0.35],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "training_curves.png"
            plot_training_curves(history, output_path)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_runner_saves_validation_result_summary(self):
        runner_path = PROJECT_ROOT / "src" / "training" / "task_level_full_data_runner.py"

        runner_text = runner_path.read_text(encoding="utf-8")

        self.assertIn("save_json(output_dir / TRAIN_RESULTS_FILENAME, train_result)", runner_text)

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
