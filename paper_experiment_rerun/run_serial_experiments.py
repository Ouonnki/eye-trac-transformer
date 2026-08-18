"""Run paper experiments serially with reproducible artifacts and reports.

Examples:
  python paper_experiment_rerun/run_serial_experiments.py
  python paper_experiment_rerun/run_serial_experiments.py --mode dry-run
  python paper_experiment_rerun/run_serial_experiments.py --mode injected-failure
  python paper_experiment_rerun/run_serial_experiments.py --mode smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiment_rerun.build_results_workbook import (
    IncompleteRunRecordError,
    RerunRecordError,
    SourceRecord,
    build_results_workbook,
)
from paper_experiment_rerun.experiment_plan import (
    ConfigGenerationOptions,
    ExperimentPlanError,
    RUN_SPECS,
    ValSplitUnit,
    build_shared_split_manifest_from_data,
    generate_derived_configs,
)
from paper_experiment_rerun.tui import (
    BatchView,
    DashboardState,
    RunStatus,
    RunView,
    SerialDashboard,
)


EVENT_PREFIX = "@@PAPER_EXPERIMENT@@"
SCHEMA_VERSION = 1
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_task_level.py"
REQUIRED_ARTIFACTS = ("best_model.pt", "final_model.pt", "config.json", "test_results.json")
COMPLETE_SPLITS = frozenset(("Val", "Test1", "Test2", "Test3"))


@dataclass(frozen=True)
class LaunchRequest:
    """Immutable child invocation inputs used by real and fake launchers."""

    command: Tuple[str, ...]
    run_id: str
    output_parent: Path
    config_path: Path


@dataclass(frozen=True)
class LaunchResult:
    """Child process status observed through the shared line-event protocol."""

    exit_code: int


@dataclass(frozen=True)
class RunOutcome:
    """One completed or failed attempt persisted in manifests and reports."""

    run_id: str
    ordinal: int
    status: str
    reason: Optional[str]
    exit_code: Optional[int]
    log_path: Path
    output_path: Path
    config_path: Path
    test_results_path: Optional[Path]


LineConsumer = Callable[[str], None]
Launcher = Callable[[LaunchRequest, LineConsumer], LaunchResult]


class RunnerError(Exception):
    """Expected runner boundary error with a stable operator-facing message."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse supported serial execution modes."""
    parser = argparse.ArgumentParser(
        description="Serial fixed-split paper experiment rerunner.",
        epilog=(
            "Examples:\n"
            "  python paper_experiment_rerun/run_serial_experiments.py\n"
            "  python paper_experiment_rerun/run_serial_experiments.py --mode dry-run\n"
            "  python paper_experiment_rerun/run_serial_experiments.py --mode injected-failure\n"
            "  python paper_experiment_rerun/run_serial_experiments.py --mode smoke"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("run", "dry-run", "injected-failure", "smoke"),
        default="run",
        help="run all experiments, validate commands only, exercise fake failure, or run FULL_BILSTM smoke",
    )
    parser.add_argument(
        "--val-split-unit",
        choices=("sample", "subject"),
        default="sample",
        help=(
            "sample randomly holds out 200 subject-task samples; subject randomly "
            "holds out 10 complete subjects (200 samples)"
        ),
    )
    return parser.parse_args(argv)


def _batch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _runtime_paths(root: Path, batch_id: str) -> Tuple[Path, Path, Path, Path]:
    return (
        root / "configs" / batch_id,
        root / "runs" / batch_id,
        root / "logs" / batch_id,
        root / "reports" / batch_id,
    )


def _generated_configs(config_dir: Path, runs_dir: Path, epochs_override: Optional[int]) -> Dict[str, Dict]:
    options = ConfigGenerationOptions(
        project_root=PROJECT_ROOT,
        manifest_path=config_dir / "shared_split_manifest.json",
        output_dir=runs_dir,
        epochs_override=epochs_override,
    )
    configs = generate_derived_configs(options)
    expected_ids = tuple(spec.run_id for spec in RUN_SPECS)
    if tuple(configs) != expected_ids:
        raise RunnerError("generated config order does not match fixed RUN_SPECS order")
    if len(configs) != 14:
        raise RunnerError("generated config matrix must contain exactly 14 runs")
    return configs


def _data_path(config: Dict) -> Path:
    data = config.get("data")
    if not isinstance(data, dict):
        raise RunnerError("first generated config has no data object")
    raw_path = data.get("processed_data_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RunnerError("first generated config has no data.processed_data_path")
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_configs(configs: Dict[str, Dict], config_dir: Path, smoke: bool) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for run_id, config in configs.items():
        if smoke:
            training = config.get("training")
            if not isinstance(training, dict):
                raise RunnerError("smoke config has no training object")
            training["patience"] = 1
        path = config_dir / (run_id.lower() + ".json")
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[run_id] = path
    return paths


def _command(config_path: Path, output_parent: Path) -> Tuple[str, ...]:
    return (
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_parent),
        "--event-stream",
    )


def _subprocess_launcher(request: LaunchRequest, consume: LineConsumer) -> LaunchResult:
    """Run one training child serially and forward every stdout/stderr line."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        request.command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        text=True,
        bufsize=1,
        shell=False,
    )
    try:
        if process.stdout is None:
            raise RunnerError("child stdout pipe was not created")
        while True:
            line = process.stdout.readline()
            if line:
                consume(line.rstrip("\r\n"))
                continue
            if process.poll() is not None:
                break
        return LaunchResult(process.wait())
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise


def _parse_event(line: str) -> Optional[Dict]:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _complete_result_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    found = set()
    for raw_name, split in payload.items():
        if not isinstance(raw_name, str) or not isinstance(split, dict):
            return False
        name = _canonical_split(raw_name)
        if name in found:
            return False
        if name in COMPLETE_SPLITS:
            labels = split.get("labels")
            predictions = split.get("predictions")
            probabilities = split.get("probabilities")
            if not isinstance(labels, list) or not isinstance(predictions, list) or not isinstance(probabilities, list):
                return False
            if not labels or len(labels) != len(predictions) or len(labels) != len(probabilities):
                return False
            found.add(name)
    return found == COMPLETE_SPLITS


def _canonical_split(name: str) -> str:
    normalized = name.lower()
    if "val" in normalized or "同分布" in name:
        return "Val"
    if "test1" in normalized:
        return "Test1"
    if "test2" in normalized:
        return "Test2"
    if "test3" in normalized:
        return "Test3"
    return name


def _valid_output(output_path: Optional[Path]) -> Tuple[bool, str]:
    if output_path is None:
        return False, "missing output_dir event"
    if not output_path.is_dir():
        return False, "output directory does not exist"
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_path / name).is_file()]
    if missing:
        return False, "missing artifacts: " + ", ".join(missing)
    if not _complete_result_file(output_path / "test_results.json"):
        return False, "test_results.json lacks four complete splits"
    return True, ""


def _dashboard_state(
    batch_id: str,
    total: int,
    started: datetime,
    run_id: str,
    ordinal: int,
    config: Dict,
    status: RunStatus,
    epoch: int,
    log_path: Path,
    succeeded: int,
    failed: int,
    latest: str,
) -> DashboardState:
    model = config["model"]
    training = config["training"]
    current = RunView(
        ordinal=ordinal,
        source=next(spec.template_path for spec in RUN_SPECS if spec.run_id == run_id),
        encoder=str(model.get("segment_encoder_type", "unknown")),
        head=str(model.get("head_type", "unknown")),
        use_task_embedding=bool(model.get("use_task_embedding", False)),
        status=status,
        epoch=epoch,
        total_epochs=int(training.get("epochs", 0)),
        elapsed=datetime.now() - started,
        log_path=log_path,
    )
    completed = succeeded + failed
    pending = max(0, total - completed - (1 if status == RunStatus.RUNNING else 0))
    return DashboardState(
        batch=BatchView(batch_id, total, datetime.now() - started),
        current_run=current,
        succeeded=succeeded,
        failed=failed,
        pending=pending,
        latest_message=latest,
    )


def _attempt(
    request: LaunchRequest,
    config: Dict,
    ordinal: int,
    total: int,
    batch_id: str,
    started: datetime,
    log_path: Path,
    launcher: Launcher,
    dashboard: SerialDashboard,
    succeeded: int,
    failed: int,
) -> RunOutcome:
    actual_output: Optional[Path] = None
    completed = False
    epoch = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        def consume(line: str) -> None:
            nonlocal actual_output, completed, epoch
            log_file.write(line + "\n")
            log_file.flush()
            event = _parse_event(line)
            if event is not None:
                raw_output = event.get("output_dir")
                if isinstance(raw_output, str) and raw_output:
                    actual_output = Path(raw_output)
                if event.get("event") == "completed":
                    completed = True
                raw_epoch = event.get("epoch")
                if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
                    epoch = raw_epoch
            dashboard.update(
                _dashboard_state(
                    batch_id, total, started, request.run_id, ordinal, config, RunStatus.RUNNING,
                    epoch, log_path, succeeded, failed, line,
                )
            )

        dashboard.update(
            _dashboard_state(
                batch_id, total, started, request.run_id, ordinal, config, RunStatus.RUNNING,
                epoch, log_path, succeeded, failed, "starting child",
            )
        )
        result = launcher(request, consume)

    valid, detail = _valid_output(actual_output)
    if result.exit_code == 0 and completed and valid:
        return RunOutcome(
            request.run_id, ordinal, "success", None, result.exit_code, log_path,
            actual_output, request.config_path,
            actual_output / "test_results.json",
        )
    reasons: List[str] = []
    if result.exit_code != 0:
        reasons.append("exit code " + str(result.exit_code))
    if not completed:
        reasons.append("missing completed event")
    if not valid:
        reasons.append(detail)
    return RunOutcome(
        request.run_id, ordinal, "failed", "; ".join(reasons), result.exit_code, log_path,
        actual_output or request.output_parent,
        request.config_path, None,
    )


def _record_json(outcome: RunOutcome) -> Dict:
    return {
        "run_id": outcome.run_id,
        "ordinal": outcome.ordinal,
        "status": outcome.status,
        "reason": outcome.reason,
        "exit_code": outcome.exit_code,
        "log_path": str(outcome.log_path),
        "output_path": str(outcome.output_path),
        "config_path": str(outcome.config_path),
        "test_results_path": None if outcome.test_results_path is None else str(outcome.test_results_path),
    }


def _source_record(outcome: RunOutcome) -> SourceRecord:
    if outcome.status == "success":
        return SourceRecord(
            outcome.run_id, "success", outcome.output_path, outcome.config_path,
            outcome.test_results_path, None,
        )
    return SourceRecord(
        outcome.run_id, "failed", outcome.output_path, outcome.config_path, None,
        outcome.reason or "run did not complete",
    )


def _write_reports(
    reports_dir: Path,
    batch_id: str,
    mode: str,
    outcomes: Sequence[RunOutcome],
    planned_ids: Sequence[str],
) -> None:
    by_id = {outcome.run_id: outcome for outcome in outcomes}
    completed_records = [_source_record(by_id[run_id]) for run_id in planned_ids if run_id in by_id]
    for ordinal, run_id in enumerate(planned_ids, start=1):
        if run_id not in by_id:
            completed_records.append(
                SourceRecord(run_id, "failed", None, None, None, "not run")
            )
    attempted_failures = [outcome for outcome in outcomes if outcome.status == "failed"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "mode": mode,
        "counts": {
            "planned": len(planned_ids),
            "attempted": len(outcomes),
            "succeeded": sum(outcome.status == "success" for outcome in outcomes),
            "failed": len(attempted_failures),
            "not_run": len(planned_ids) - len(outcomes),
        },
        "runs": [
            _record_json(by_id[run_id])
            if run_id in by_id
            else {
                "run_id": run_id,
                "ordinal": ordinal,
                "status": "not_run",
                "reason": "not run",
                "exit_code": None,
                "log_path": None,
                "output_path": None,
                "config_path": None,
                "test_results_path": None,
            }
            for ordinal, run_id in enumerate(planned_ids, start=1)
        ],
    }
    errors = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "errors": [_record_json(outcome) for outcome in attempted_failures],
    }
    (reports_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports_dir / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_results_workbook(completed_records, reports_dir / "final_results.xlsx")


def _fake_results() -> Dict:
    split = {"labels": [0, 1], "predictions": [0, 1], "probabilities": [[0.9, 0.1], [0.1, 0.9]]}
    return {name: split for name in ("Val", "Test1", "Test2", "Test3")}


def _injected_launcher() -> Tuple[Launcher, List[int]]:
    attempts: List[int] = []

    def launch(request: LaunchRequest, consume: LineConsumer) -> LaunchResult:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            consume("fake child failure")
            return LaunchResult(17)
        output = request.output_parent / (request.run_id.lower() + "_fake")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("best_model.pt", "final_model.pt"):
            (output / name).write_bytes(b"fake")
        (output / "config.json").write_text("{}\n", encoding="utf-8")
        (output / "test_results.json").write_text(json.dumps(_fake_results()), encoding="utf-8")
        consume(EVENT_PREFIX + json.dumps({"event": "output_dir", "output_dir": str(output)}))
        consume(EVENT_PREFIX + json.dumps({"event": "completed", "output_dir": str(output)}))
        return LaunchResult(0)

    return launch, attempts


def _prepare_temporary_batch(
    temp_root: Path,
    smoke: bool,
    skip_data: bool = False,
    val_split_unit: ValSplitUnit = "sample",
) -> Tuple[str, Path, Path, Path, Path, Dict[str, Dict], Dict[str, Path]]:
    batch_id = _batch_id()
    config_dir, runs_dir, logs_dir, reports_dir = _runtime_paths(temp_root, batch_id)
    configs = _generated_configs(config_dir, runs_dir, 1 if smoke else None)
    for path in (config_dir, runs_dir, logs_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    if skip_data:
        (config_dir / "shared_split_manifest.json").write_text("{\"schema_version\":1}\n", encoding="utf-8")
    else:
        data_path = _data_path(configs[RUN_SPECS[0].run_id])
        if not data_path.is_file():
            raise RunnerError("Preflight failed: processed data does not exist: " + str(data_path))
        build_shared_split_manifest_from_data(
            data_path,
            config_dir / "shared_split_manifest.json",
            val_split_unit=val_split_unit,
        )
    config_paths = _write_configs(configs, config_dir, smoke)
    return batch_id, config_dir, runs_dir, logs_dir, reports_dir, configs, config_paths


def _run_batch(
    mode: str,
    runtime_root: Path,
    launcher: Launcher,
    selected_ids: Sequence[str],
    smoke: bool,
    skip_data: bool = False,
    val_split_unit: ValSplitUnit = "sample",
) -> int:
    batch_id, config_dir, runs_dir, logs_dir, reports_dir, configs, config_paths = _prepare_temporary_batch(
        runtime_root,
        smoke,
        skip_data,
        val_split_unit=val_split_unit,
    )
    del config_dir
    started = datetime.now()
    outcomes: List[RunOutcome] = []
    interrupted = False
    with SerialDashboard() as dashboard:
        for ordinal, run_id in enumerate(selected_ids, start=1):
            request = LaunchRequest(
                _command(config_paths[run_id], runs_dir / run_id.lower()),
                run_id,
                runs_dir / run_id.lower(),
                config_paths[run_id],
            )
            try:
                outcome = _attempt(
                    request, configs[run_id], ordinal, len(selected_ids), batch_id, started,
                    logs_dir / (run_id.lower() + ".log"), launcher, dashboard,
                    sum(item.status == "success" for item in outcomes),
                    sum(item.status == "failed" for item in outcomes),
                )
            except KeyboardInterrupt:
                interrupted = True
                outcomes.append(
                    RunOutcome(run_id, ordinal, "failed", "interrupted", None,
                               logs_dir / (run_id.lower() + ".log"), runs_dir / run_id.lower(),
                               config_paths[run_id], None)
                )
                break
            outcomes.append(outcome)
            dashboard.update(
                _dashboard_state(
                    batch_id, len(selected_ids), started, run_id, ordinal, configs[run_id],
                    RunStatus.SUCCESS if outcome.status == "success" else RunStatus.FAILED, 0,
                    outcome.log_path, sum(item.status == "success" for item in outcomes),
                    sum(item.status == "failed" for item in outcomes), outcome.reason or "completed",
                )
            )
    try:
        _write_reports(reports_dir, batch_id, mode, outcomes, tuple(spec.run_id for spec in RUN_SPECS))
    except (IncompleteRunRecordError, RerunRecordError, OSError, KeyError, ValueError) as error:
        print("Report failed: " + str(error), file=sys.stderr)
        return 2
    if interrupted:
        return 130
    if any(outcome.status == "failed" for outcome in outcomes):
        return 1
    return 0


def _dry_run() -> int:
    batch_id = _batch_id()
    config_dir, runs_dir, _, _ = _runtime_paths(PROJECT_ROOT / "paper_experiment_rerun", batch_id)
    configs = _generated_configs(config_dir, runs_dir, None)
    for spec in RUN_SPECS:
        config_path = config_dir / (spec.run_id.lower() + ".json")
        print(spec.run_id + ": " + subprocess.list2cmdline(_command(config_path, runs_dir / spec.run_id.lower())))
    if len(configs) != len(RUN_SPECS):
        raise RunnerError("dry-run config count mismatch")
    return 0


def _run_injected_failure() -> int:
    with TemporaryDirectory(prefix="paper-experiment-injected-") as temporary:
        launcher, attempts = _injected_launcher()
        exit_code = _run_batch(
            "injected-failure", Path(temporary), launcher,
            tuple(spec.run_id for spec in RUN_SPECS), False, skip_data=True,
        )
        failures = 1
        if len(attempts) != 14:
            raise RunnerError("injected-failure must attempt each of 14 runs exactly once")
        print("injected-failure attempts=" + str(len(attempts)) + " failures=" + str(failures))
        return 0 if exit_code == 1 else exit_code


def _run_smoke(val_split_unit: ValSplitUnit = "sample") -> int:
    configs = _generated_configs(Path("smoke-configs"), Path("smoke-runs"), 1)
    data_path = _data_path(configs[RUN_SPECS[0].run_id])
    if not data_path.is_file():
        print("Preflight failed: processed data does not exist: " + str(data_path), file=sys.stderr)
        return 2
    with TemporaryDirectory(prefix="paper-experiment-smoke-") as temporary:
        return _run_batch(
            "smoke",
            Path(temporary),
            _subprocess_launcher,
            ("FULL_BILSTM",),
            True,
            val_split_unit=val_split_unit,
        )


def _run_production(val_split_unit: ValSplitUnit = "sample") -> int:
    batch_id = _batch_id()
    config_dir, runs_dir, _, _ = _runtime_paths(PROJECT_ROOT / "paper_experiment_rerun", batch_id)
    try:
        configs = _generated_configs(config_dir, runs_dir, None)
        data_path = _data_path(configs[RUN_SPECS[0].run_id])
    except (ExperimentPlanError, OSError, RunnerError) as error:
        print("Preflight failed: " + str(error), file=sys.stderr)
        return 2
    if not data_path.is_file():
        print("Preflight failed: processed data does not exist: " + str(data_path), file=sys.stderr)
        return 2
    return _run_batch(
        "run",
        PROJECT_ROOT / "paper_experiment_rerun",
        _subprocess_launcher,
        tuple(spec.run_id for spec in RUN_SPECS),
        False,
        val_split_unit=val_split_unit,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute requested mode; only this CLI boundary handles unexpected errors."""
    try:
        args = parse_args(argv)
        mode = args.mode
        val_split_unit: ValSplitUnit = args.val_split_unit
        if mode == "dry-run":
            return _dry_run()
        if mode == "injected-failure":
            return _run_injected_failure()
        if mode == "smoke":
            return _run_smoke(val_split_unit)
        return _run_production(val_split_unit)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # CLI boundary records user-visible failure as exit 2.
        print("Runner failed: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
