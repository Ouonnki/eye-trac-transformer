from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Protocol, TypedDict, Union

import numpy as np


JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List["JsonValue"], Dict[str, "JsonValue"]]
JsonObject = Dict[str, JsonValue]
SEED: Final = 42
TRAIN_SUBJECT_COUNT: Final = 100
TRAIN_TASK_COUNT: Final = 20
TOTAL_TASK_COUNT: Final = 30
TRAIN_SAMPLE_COUNT: Final = 1800
VAL_SAMPLE_COUNT: Final = 200


class TaskSample(TypedDict):
    subject_id: str
    task_id: int


class SampleDataset(Protocol):
    samples: Sequence[TaskSample]


@dataclass(frozen=True, order=True)
class SampleIdentity:
    subject_id: str
    task_id: int

    def to_json(self) -> JsonObject:
        return {"subject_id": self.subject_id, "task_id": self.task_id}


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    dataset_fingerprint: str
    train: tuple[SampleIdentity, ...]
    val: tuple[SampleIdentity, ...]
    test1: tuple[SampleIdentity, ...]
    test2: tuple[SampleIdentity, ...]
    test3: tuple[SampleIdentity, ...]

    def to_json(self) -> JsonObject:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "train": _serialize_samples(self.train),
            "val": _serialize_samples(self.val),
            "test1": _serialize_samples(self.test1),
            "test2": _serialize_samples(self.test2),
            "test3": _serialize_samples(self.test3),
        }


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    template_path: Path
    use_task_embedding: bool


@dataclass(frozen=True)
class ConfigGenerationOptions:
    project_root: Path
    manifest_path: Path
    output_dir: Path
    epochs_override: Optional[int] = None


@dataclass(frozen=True)
class ExperimentPlanError(Exception):
    resource: Path
    issue: str

    def __str__(self) -> str:
        return f"{self.resource}: {self.issue}"


RUN_SPECS: Final[tuple[RunSpec, ...]] = (
    RunSpec("BASE_TRANSFORMER", Path("configs/task_level_classification_manual11_transformer.json"), False),
    RunSpec("BASE_CNN1D", Path("configs/task_level_classification_manual11_cnn1d.json"), False),
    RunSpec("BASE_RNN", Path("configs/task_level_classification_manual11_rnn.json"), False),
    RunSpec("BASE_LSTM", Path("configs/task_level_classification_manual11_lstm.json"), False),
    RunSpec("BASE_GRU", Path("configs/task_level_classification_manual11_gru.json"), False),
    RunSpec("BASE_BILSTM", Path("configs/task_level_classification_manual11_bilstm.json"), False),
    RunSpec("FULL_TRANSFORMER", Path("configs/task_level_small_lr_ordinal_pw_manual11.json"), True),
    RunSpec("FULL_CNN1D", Path("configs/task_level_ordinal_manual11_cnn1d.json"), True),
    RunSpec("FULL_RNN", Path("configs/task_level_ordinal_manual11_rnn.json"), True),
    RunSpec("FULL_LSTM", Path("configs/task_level_ordinal_manual11_lstm.json"), True),
    RunSpec("FULL_GRU", Path("configs/task_level_ordinal_manual11_gru.json"), True),
    RunSpec("FULL_BILSTM", Path("configs/task_level_ordinal_manual11_bilstm.json"), True),
    RunSpec("TASK_BILSTM", Path("configs/task_level_classification_manual11_bilstm.json"), True),
    RunSpec("ORDINAL_BILSTM", Path("configs/task_level_ordinal_manual11_bilstm.json"), False),
)


def generate_derived_configs(options: ConfigGenerationOptions) -> dict[str, JsonObject]:
    """Build all validated run configurations without changing templates."""
    if options.epochs_override is not None and options.epochs_override <= 0:
        raise ExperimentPlanError(options.output_dir, "epochs_override must be positive")
    manifest_path = str(options.manifest_path.resolve())
    output_dir = str(options.output_dir.resolve())
    generated: dict[str, JsonObject] = {}
    for spec in RUN_SPECS:
        if spec.run_id in generated:
            raise ExperimentPlanError(options.project_root, f"duplicate run ID {spec.run_id}")
        template_path = options.project_root / spec.template_path
        config = _load_template(template_path)
        experiment = _section(config, "experiment", template_path)
        model = _section(config, "model", template_path)
        training = _section(config, "training", template_path)
        _section(config, "sequence", template_path)
        _section(config, "data", template_path)
        experiment.update(name=spec.run_id, output_dir=output_dir, random_seed=SEED)
        model["use_task_embedding"] = spec.use_task_embedding
        if options.epochs_override is not None:
            training["epochs"] = options.epochs_override
        config["split"] = {
            "mode": "fixed_sample_holdout",
            "manifest_path": manifest_path,
        }
        generated[spec.run_id] = config
    return generated


def build_shared_split_manifest(dataset: SampleDataset) -> SplitManifest:
    """Create the seed-42 fixed-sample manifest for the canonical 2x2 pools."""
    identities = tuple(sorted(SampleIdentity(str(sample["subject_id"]), int(sample["task_id"])) for sample in dataset.samples))
    if len(set(identities)) != len(identities):
        raise ExperimentPlanError(Path("dataset"), "duplicate subject/task sample identity")
    subjects = tuple(sorted({identity.subject_id for identity in identities}))
    if len(subjects) < TRAIN_SUBJECT_COUNT:
        raise ExperimentPlanError(Path("dataset"), "fewer than 100 subjects")
    train_subjects = frozenset(subjects[:TRAIN_SUBJECT_COUNT])
    train_tasks = frozenset(range(1, TRAIN_TASK_COUNT + 1))
    test_tasks = frozenset(range(TRAIN_TASK_COUNT + 1, TOTAL_TASK_COUNT + 1))
    train_pool = tuple(identity for identity in identities if identity.subject_id in train_subjects and identity.task_id in train_tasks)
    if len(train_pool) != TRAIN_SAMPLE_COUNT + VAL_SAMPLE_COUNT:
        raise ExperimentPlanError(Path("dataset"), "expected 2000 samples in the training pool")
    permutation = np.random.RandomState(SEED).permutation(len(train_pool))
    train = tuple(train_pool[index] for index in permutation[:TRAIN_SAMPLE_COUNT])
    val = tuple(train_pool[index] for index in permutation[TRAIN_SAMPLE_COUNT:])
    test_subjects = frozenset(subjects[TRAIN_SUBJECT_COUNT:])
    test1 = tuple(identity for identity in identities if identity.subject_id in test_subjects and identity.task_id in train_tasks)
    test2 = tuple(identity for identity in identities if identity.subject_id in train_subjects and identity.task_id in test_tasks)
    test3 = tuple(identity for identity in identities if identity.subject_id in test_subjects and identity.task_id in test_tasks)
    if len(train) != TRAIN_SAMPLE_COUNT or len(val) != VAL_SAMPLE_COUNT:
        raise ExperimentPlanError(Path("dataset"), "expected 1800 train and 200 validation samples")
    return SplitManifest(SEED, _dataset_fingerprint(identities), train, val, test1, test2, test3)


def write_shared_split_manifest(manifest: SplitManifest, manifest_path: Path) -> Path:
    """Serialize one fixed-split manifest and return its absolute path."""
    absolute_path = manifest_path.resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return absolute_path


def build_shared_split_manifest_from_data(processed_data_path: Path, manifest_path: Path) -> Path:
    """Construct the task-level dataset exactly as training does, then write its manifest."""
    from src.models.task_level_dataset import TaskLevelGazeDataset, TaskLevelSequenceConfig

    with processed_data_path.open("rb") as source:
        processed_data = pickle.load(source)
    dataset = TaskLevelGazeDataset(
        processed_data=processed_data,
        config=TaskLevelSequenceConfig(max_seq_len=300, max_segments=25),
        fit_normalizer=False,
    )
    return write_shared_split_manifest(build_shared_split_manifest(dataset), manifest_path)


def _load_template(template_path: Path) -> JsonObject:
    if not template_path.is_file():
        raise ExperimentPlanError(template_path, "template does not exist")
    try:
        config = json.loads(template_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExperimentPlanError(template_path, "template is not valid JSON") from error
    if not isinstance(config, dict):
        raise ExperimentPlanError(template_path, "template root must be an object")
    return config


def _section(config: JsonObject, name: str, template_path: Path) -> JsonObject:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ExperimentPlanError(template_path, f"missing object section {name}")
    return section


def _serialize_samples(samples: Sequence[SampleIdentity]) -> List[JsonValue]:
    return [sample.to_json() for sample in samples]


def _dataset_fingerprint(identities: Sequence[SampleIdentity]) -> str:
    keys = sorted([list((identity.subject_id, identity.task_id)) for identity in identities])
    payload = json.dumps(keys, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
