# -*- coding: utf-8 -*-
"""Regression tests for reproducible task-level training reruns."""

import argparse
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import train_task_level as training


class GridDataset:
    def __init__(self):
        self.samples = [
            {"subject_id": f"subject-{subject:03d}", "task_id": task, "label": 0}
            for subject in range(1, 158)
            for task in range(1, 31)
        ]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class FakeParameter:
    requires_grad = True

    def numel(self):
        return 1


class FakeModel:
    def to(self, _device):
        return self

    def parameters(self):
        return (FakeParameter(),)


class FakeScheduler:
    def step(self, _loss):
        return None


class FakeTrainer:
    instances = []

    def __init__(self, **_kwargs):
        self.scheduler = FakeScheduler()
        self.calls = []
        type(self).instances.append(self)

    def train_epoch(self, _loader, epoch, _total_epochs):
        return {
            "loss": 1.0 / epoch,
            "accuracy": 0.5,
            "f1_weighted": 0.5,
            "f1_macro": 0.5,
        }

    def evaluate(self, _loader, desc):
        metric = 0.8 if "Epoch 1" in desc else 0.7
        return {
            "loss": 1.0 - metric,
            "accuracy": metric,
            "f1_weighted": metric,
            "f1_macro": metric,
        }

    def save_checkpoint(self, path, epoch, best_metric):
        Path(path).write_text("checkpoint", encoding="utf-8")
        self.calls.append(("save", Path(path).name, epoch, best_metric))

    def load_checkpoint(self, path):
        self.calls.append(("load", Path(path).name))
        return 1, 0.8


class FixedDateTime:
    @classmethod
    def now(cls):
        return cls()

    def strftime(self, _format):
        return "20260802_010203"


def sample_key(sample):
    return str(sample["subject_id"]), int(sample["task_id"])


def manifest_entries(dataset, indices):
    return [
        {"subject_id": str(dataset.samples[index]["subject_id"]), "task_id": int(dataset.samples[index]["task_id"])}
        for index in indices
    ]


class TrainTaskLevelRerunTest(unittest.TestCase):
    def test_fixed_sample_holdout_resolves_manifest_keys_in_manifest_order(self):
        dataset = GridDataset()
        legacy_splits = training.split_2x2(dataset)
        split_names = ("train", "val", "test1", "test2", "test3")
        manifest = {
            "schema_version": 1,
            "dataset_fingerprint": training.dataset_fingerprint(dataset),
            **{
                name: manifest_entries(dataset, indices)[::-1]
                for name, indices in zip(split_names, legacy_splits)
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "fixed_holdout.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            resolved_splits = training.split_2x2(
                dataset,
                split_manifest_path=manifest_path,
            )

        for name, indices in zip(split_names, resolved_splits):
            self.assertEqual(
                [sample_key(dataset.samples[index]) for index in indices],
                [(entry["subject_id"], entry["task_id"]) for entry in manifest[name]],
            )

    def test_fixed_sample_holdout_rejects_invalid_schema_fingerprint_and_duplicate_keys(self):
        dataset = GridDataset()
        legacy_splits = training.split_2x2(dataset)
        manifest = {
            "schema_version": 1,
            "dataset_fingerprint": training.dataset_fingerprint(dataset),
            "train": manifest_entries(dataset, legacy_splits[0]),
            "val": manifest_entries(dataset, legacy_splits[1]),
            "test1": manifest_entries(dataset, legacy_splits[2]),
            "test2": manifest_entries(dataset, legacy_splits[3]),
            "test3": manifest_entries(dataset, legacy_splits[4]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "fixed_holdout.json"
            bad_schema = {**manifest, "schema_version": 2}
            manifest_path.write_text(json.dumps(bad_schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                training.split_2x2(dataset, split_manifest_path=manifest_path)

            bad_fingerprint = {**manifest, "dataset_fingerprint": "wrong"}
            manifest_path.write_text(json.dumps(bad_fingerprint), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset fingerprint"):
                training.split_2x2(dataset, split_manifest_path=manifest_path)

            duplicate_train = manifest["train"][:]
            duplicate_train[0] = duplicate_train[1]
            duplicate_key = {**manifest, "train": duplicate_train}
            manifest_path.write_text(json.dumps(duplicate_key), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                training.split_2x2(dataset, split_manifest_path=manifest_path)

    def test_default_split_manifest_path_preserves_legacy_subject_split(self):
        dataset = GridDataset()

        expected = training.split_2x2(dataset)
        actual = training.split_2x2(dataset, split_manifest_path=None)

        self.assertEqual(actual, expected)

    def test_event_stream_reports_progress_and_saves_final_checkpoint_before_best_reload(self):
        output, calls = self._run_main(event_stream=True)

        event_prefix = "@@PAPER_EXPERIMENT@@"
        events = [
            json.loads(line[len(event_prefix):])
            for line in output.splitlines()
            if line.startswith(event_prefix)
        ]

        self.assertEqual(
            [event["event"] for event in events],
            ["output_dir", "epoch_end", "epoch_end", "completed"],
        )
        self.assertEqual(events[1]["epoch"], 1)
        self.assertEqual(events[2]["epoch"], 2)
        self.assertEqual(calls, [
            ("save", "best_model.pt", 1, 0.8),
            ("save", "final_model.pt", 2, 0.8),
            ("load", "best_model.pt"),
        ])

    def test_default_output_does_not_emit_event_records(self):
        output, _calls = self._run_main(event_stream=False)

        self.assertNotIn("@@PAPER_EXPERIMENT@@", output)

    def _run_main(self, event_stream):
        dataset = GridDataset()
        FakeTrainer.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_data_path = root / "processed_data.pkl"
            processed_data_path.write_bytes(pickle.dumps([]))
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "data": {"processed_data_path": str(processed_data_path)},
                "sequence": {"max_seq_len": 4, "max_segments": 1, "input_dim": 7},
                "experiment": {
                    "random_seed": 42,
                    "name": "rerun",
                    "output_dir": str(root / "outputs"),
                    "train_subjects": 100,
                    "train_tasks": 20,
                },
                "model": {
                    "segment_d_model": 4,
                    "segment_nhead": 1,
                    "segment_num_layers": 1,
                    "segment_encoder_type": "bilstm",
                    "attention_dim": 4,
                    "task_embedding_dim": 1,
                    "dropout": 0.0,
                    "num_classes": 3,
                },
                "training": {
                    "train_val_split": 0.9,
                    "use_class_weights": False,
                    "use_balanced_sampler": False,
                    "batch_size": 8,
                    "num_workers": 0,
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "grad_clip": 1.0,
                    "epochs": 2,
                    "patience": 2,
                    "early_stop_metric": "f1_macro",
                    "skip_test_evaluation": True,
                },
            }), encoding="utf-8")
            args = argparse.Namespace(
                config=str(config_path),
                use_task_embedding=None,
                experiment_name=None,
                output_dir=None,
                event_stream=event_stream,
            )
            stdout = StringIO()
            with (
                patch.object(training, "parse_args", return_value=args),
                patch.object(training, "datetime", FixedDateTime),
                patch.object(training, "TaskLevelGazeDataset", return_value=dataset),
                patch.object(training, "TaskLevelEncoder", return_value=FakeModel()),
                patch.object(training, "TaskLevelTrainer", FakeTrainer),
                patch.object(training, "plot_training_curves"),
                redirect_stdout(stdout),
            ):
                training.main()

        return stdout.getvalue(), FakeTrainer.instances[0].calls


if __name__ == "__main__":
    unittest.main()
