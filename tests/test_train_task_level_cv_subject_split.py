from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from paper_experiment_rerun.experiment_plan import build_shared_split_manifest
from scripts.train_task_level_cv import make_subject_folds, write_fold_manifest


@dataclass
class FakeDataset:
    samples: List[Dict[str, Union[int, str]]]


def make_dataset() -> FakeDataset:
    return FakeDataset(
        [
            {"subject_id": f"S{subject:03d}", "task_id": task, "label": task % 3}
            for subject in range(1, 101)
            for task in range(1, 21)
        ]
    )


class SubjectFoldTests(unittest.TestCase):
    def test_subject_folds_hold_out_ten_complete_subjects(self) -> None:
        dataset = make_dataset()
        pool_indices = list(range(len(dataset.samples)))

        folds = make_subject_folds(pool_indices, dataset, n_folds=10, seed=42)

        self.assertEqual(len(folds), 10)
        all_val_positions = set()
        for train_positions, val_positions in folds:
            self.assertEqual(len(train_positions), 1800)
            self.assertEqual(len(val_positions), 200)
            train_subjects = {
                dataset.samples[pool_indices[int(position)]]["subject_id"]
                for position in train_positions
            }
            val_subjects = {
                dataset.samples[pool_indices[int(position)]]["subject_id"]
                for position in val_positions
            }
            self.assertEqual(len(train_subjects), 90)
            self.assertEqual(len(val_subjects), 10)
            self.assertFalse(train_subjects & val_subjects)
            all_val_positions.update(int(position) for position in val_positions)
        self.assertEqual(all_val_positions, set(range(2000)))

    def test_first_cv_fold_matches_subject_pipeline_holdout(self) -> None:
        dataset = make_dataset()
        pool_indices = list(range(len(dataset.samples)))

        folds = make_subject_folds(pool_indices, dataset, n_folds=10, seed=42)
        pipeline_manifest = build_shared_split_manifest(
            dataset,
            val_split_unit="subject",
        )
        first_val_subjects = {
            dataset.samples[pool_indices[int(position)]]["subject_id"]
            for position in folds[0][1]
        }

        self.assertEqual(
            first_val_subjects,
            {sample.subject_id for sample in pipeline_manifest.val},
        )

    def test_fold_manifest_records_subject_membership(self) -> None:
        dataset = make_dataset()
        pool_indices = list(range(len(dataset.samples)))
        folds = make_subject_folds(pool_indices, dataset, n_folds=10, seed=42)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "split_manifest.json"

            write_fold_manifest(
                path,
                dataset,
                pool_indices,
                folds,
                fold_by="subject",
                seed=42,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["fold_by"], "subject")
            self.assertEqual(len(payload["folds"]), 10)
            self.assertTrue(
                all(len(fold["val_subject_ids"]) == 10 for fold in payload["folds"])
            )
            self.assertTrue(
                all(len(fold["val"]) == 200 for fold in payload["folds"])
            )


if __name__ == "__main__":
    unittest.main()
