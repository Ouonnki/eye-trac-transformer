"""End-to-end contracts for serial paper-experiment runner modes."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "paper_experiment_rerun" / "run_serial_experiments.py"


class PaperExperimentRunnerTests(unittest.TestCase):
    def _run_runner(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--mode", mode],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            encoding="utf-8",
            text=True,
            check=False,
        )

    def test_dry_run_when_templates_are_valid_prints_fourteen_commands(self) -> None:
        # Given: repository templates and no runtime batch writes.
        # When: dry-run is requested.
        result = self._run_runner("dry-run")

        # Then: all fixed-order child commands are printed without execution.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("train_task_level.py"), 14)
        self.assertIn("BASE_TRANSFORMER", result.stdout)
        self.assertIn("ORDINAL_BILSTM", result.stdout)

    def test_injected_failure_when_first_child_fails_continues_all_attempts(self) -> None:
        # Given: deterministic injected child protocol.
        # When: injected-failure mode is requested.
        result = self._run_runner("injected-failure")

        # Then: one failure is retained while all fourteen attempts complete.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("attempts=14", result.stdout)
        self.assertIn("failures=1", result.stdout)

    def test_run_when_processed_data_is_missing_returns_preflight_exit(self) -> None:
        # Given: checked-out repository has no canonical processed dataset.
        # When: production run is requested.
        result = self._run_runner("run")

        # Then: no training child starts and preflight failure is explicit.
        self.assertEqual(result.returncode, 2)
        self.assertIn("Preflight failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
