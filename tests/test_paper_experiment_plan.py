from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from paper_experiment_rerun.experiment_plan import (
    ConfigGenerationOptions,
    RUN_SPECS,
    build_shared_split_manifest,
    build_shared_split_manifest_from_data,
    generate_derived_configs,
    write_shared_split_manifest,
)
from scripts.train_task_level import dataset_fingerprint, split_2x2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUN_IDS = (
    "BASE_TRANSFORMER",
    "BASE_CNN1D",
    "BASE_RNN",
    "BASE_LSTM",
    "BASE_GRU",
    "BASE_BILSTM",
    "FULL_TRANSFORMER",
    "FULL_CNN1D",
    "FULL_RNN",
    "FULL_LSTM",
    "FULL_GRU",
    "FULL_BILSTM",
    "TASK_BILSTM",
    "ORDINAL_BILSTM",
)


@dataclass(frozen=True)
class FakeDataset:
    samples: Tuple[Dict[str, Union[int, str]], ...]


def make_dataset() -> FakeDataset:
    return FakeDataset(
        tuple(
            {"subject_id": f"S{subject:03d}", "task_id": task}
            for subject in range(1, 111)
            for task in range(1, 31)
        )
    )


class PaperExperimentPlanTests(unittest.TestCase):
    def test_run_specs_have_exact_order_unique_ids_and_templates(self) -> None:
        self.assertEqual(tuple(spec.run_id for spec in RUN_SPECS), EXPECTED_RUN_IDS)
        self.assertEqual(len({spec.run_id for spec in RUN_SPECS}), 14)
        for spec in RUN_SPECS:
            self.assertTrue((PROJECT_ROOT / spec.template_path).is_file())

    def test_derived_configs_preserve_templates_except_approved_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            options = ConfigGenerationOptions(
                project_root=PROJECT_ROOT,
                manifest_path=temporary_path / "shared-split.json",
                output_dir=temporary_path / "outputs",
                epochs_override=3,
            )

            derived_configs = generate_derived_configs(options)

            self.assertEqual(tuple(derived_configs), EXPECTED_RUN_IDS)
            for spec in RUN_SPECS:
                template_path = PROJECT_ROOT / spec.template_path
                expected = json.loads(template_path.read_text(encoding="utf-8"))
                expected["experiment"].update(
                    {
                        "name": spec.run_id,
                        "output_dir": str(options.output_dir.resolve()),
                        "random_seed": 42,
                    }
                )
                expected["model"]["use_task_embedding"] = spec.use_task_embedding
                expected["training"]["epochs"] = 3
                expected["split"] = {
                    "mode": "fixed_sample_holdout",
                    "manifest_path": str(options.manifest_path.resolve()),
                }
                self.assertEqual(derived_configs[spec.run_id], expected)

    def test_manifest_uses_canonical_identities_and_preserves_test_partitions(self) -> None:
        dataset = make_dataset()

        manifest = build_shared_split_manifest(dataset)
        reversed_manifest = build_shared_split_manifest(
            FakeDataset(tuple(reversed(dataset.samples)))
        )

        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(len(manifest.train), 1800)
        self.assertEqual(len(manifest.val), 200)
        expected_pool = tuple(
            (f"S{subject:03d}", task)
            for subject in range(1, 101)
            for task in range(1, 21)
        )
        permutation = np.random.RandomState(42).permutation(len(expected_pool))
        self.assertEqual(
            tuple((sample.subject_id, sample.task_id) for sample in manifest.train),
            tuple(expected_pool[index] for index in permutation[:1800]),
        )
        self.assertEqual(
            tuple((sample.subject_id, sample.task_id) for sample in manifest.val),
            tuple(expected_pool[index] for index in permutation[1800:]),
        )
        self.assertEqual(len(manifest.test1), 200)
        self.assertEqual(len(manifest.test2), 1000)
        self.assertEqual(len(manifest.test3), 100)
        self.assertEqual(
            {(sample.subject_id, sample.task_id) for sample in manifest.test1},
            {(f"S{subject:03d}", task) for subject in range(101, 111) for task in range(1, 21)},
        )
        self.assertEqual(
            {(sample.subject_id, sample.task_id) for sample in manifest.test2},
            {(f"S{subject:03d}", task) for subject in range(1, 101) for task in range(21, 31)},
        )
        self.assertEqual(
            {(sample.subject_id, sample.task_id) for sample in manifest.test3},
            {(f"S{subject:03d}", task) for subject in range(101, 111) for task in range(21, 31)},
        )
        self.assertEqual(len(manifest.dataset_fingerprint), 64)
        json.dumps(manifest.to_json(), sort_keys=True)

    def test_manifest_writer_emits_serializable_fixed_holdout_data(self) -> None:
        manifest = build_shared_split_manifest(make_dataset())
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "shared-split.json"

            written_path = write_shared_split_manifest(manifest, manifest_path)

            payload = json.loads(written_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["dataset_fingerprint"], manifest.dataset_fingerprint)
            self.assertEqual(len(payload["train"]), 1800)
            self.assertEqual(len(payload["val"]), 200)
            self.assertEqual(len(payload["test1"]), 200)
            self.assertEqual(len(payload["test2"]), 1000)
            self.assertEqual(len(payload["test3"]), 100)

    def test_emitted_manifest_resolves_through_training_fixed_sample_holdout(self) -> None:
        dataset = make_dataset()
        manifest = build_shared_split_manifest(dataset)
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "shared-split.json"
            write_shared_split_manifest(manifest, manifest_path)

            resolved_splits = split_2x2(dataset, split_manifest_path=str(manifest_path))

            self.assertEqual(dataset_fingerprint(dataset), manifest.dataset_fingerprint)
            self.assertEqual(len(resolved_splits[0]), 1800)
            self.assertEqual(len(resolved_splits[1]), 200)
            self.assertEqual(
                tuple(
                    (dataset.samples[index]["subject_id"], dataset.samples[index]["task_id"])
                    for index in resolved_splits[0]
                ),
                tuple((sample.subject_id, sample.task_id) for sample in manifest.train),
            )
            self.assertEqual(
                tuple(
                    (dataset.samples[index]["subject_id"], dataset.samples[index]["task_id"])
                    for index in resolved_splits[1]
                ),
                tuple((sample.subject_id, sample.task_id) for sample in manifest.val),
            )

    def test_manifest_from_processed_data_uses_task_level_sample_construction(self) -> None:
        processed_data = [
            {
                "subject_id": f"S{subject:03d}",
                "tasks": [
                    {"task_id": task, "segments": [], "task_label": 0}
                    for task in range(1, 31)
                ],
            }
            for subject in range(1, 111)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            data_path = temporary_path / "processed.pkl"
            manifest_path = temporary_path / "shared-split.json"
            with data_path.open("wb") as data_file:
                pickle.dump(processed_data, data_file)

            written_path = build_shared_split_manifest_from_data(data_path, manifest_path)

            payload = json.loads(written_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["train"]), 1800)
            self.assertEqual(len(payload["val"]), 200)


if __name__ == "__main__":
    unittest.main()
