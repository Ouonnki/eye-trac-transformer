"""Tests for serial paper-experiment terminal dashboard."""

from __future__ import annotations

import dataclasses
import io
import unittest
from datetime import timedelta
from pathlib import Path

from rich.cells import cell_len
from rich.console import Console

from paper_experiment_rerun.tui import (
    BatchView,
    DashboardState,
    RunStatus,
    RunView,
    SerialDashboard,
)


def build_run() -> RunView:
    return RunView(
        ordinal=3,
        source=Path("配置/序数模型.json"),
        encoder="bilstm",
        head="ordinal",
        use_task_embedding=True,
        status=RunStatus.RUNNING,
        epoch=4,
        total_epochs=20,
        elapsed=timedelta(minutes=7, seconds=8),
        log_path=Path("日志/第03次/训练.log"),
    )


def build_state() -> DashboardState:
    return DashboardState(
        batch=BatchView(
            identity="paper-rerun-2026-08-02",
            total_runs=14,
            elapsed=timedelta(hours=1, minutes=2, seconds=3),
        ),
        current_run=build_run(),
        succeeded=2,
        failed=0,
        pending=11,
        latest_message="正在训练第 4 轮",
    )


class PaperExperimentDashboardTest(unittest.TestCase):
    def test_views_are_immutable_when_constructed(self) -> None:
        state = build_state()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.latest_message = "changed"

    def test_tty_dashboard_renders_complete_cjk_state(self) -> None:
        output = io.StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="standard",
            width=120,
            record=True,
        )

        with SerialDashboard(console=console) as dashboard:
            dashboard.update(build_state())

        plain = console.export_text()
        for expected in (
            "paper-rerun-2026-08-02",
            "3 / 14",
            str(Path("配置/序数模型.json")),
            "bilstm",
            "ordinal",
            "Enabled",
            "RUNNING",
            "Epoch",
            "4/20",
            "Overall",
            "2/14",
            "Success 2",
            "Failed 0",
            "Pending 11",
            "01:02:03",
            "00:07:08",
            str(Path("日志/第03次/训练.log")),
            "正在训练第 4 轮",
        ):
            self.assertIn(expected, plain)
        self.assertIn("\x1b[", output.getvalue())

    def test_non_tty_emits_only_meaningful_state_transitions(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, color_system=None, width=120)
        epoch_refresh = dataclasses.replace(
            build_state(),
            current_run=dataclasses.replace(build_run(), epoch=5),
            latest_message="高频刷新不应输出",
        )
        completed = dataclasses.replace(
            build_state(),
            current_run=dataclasses.replace(
                build_run(),
                status=RunStatus.SUCCESS,
                epoch=20,
            ),
            succeeded=3,
            pending=10,
            latest_message="第 3 次运行完成",
        )

        with SerialDashboard(console=console) as dashboard:
            dashboard.update(build_state())
            dashboard.update(epoch_refresh)
            dashboard.update(completed)

        rendered = output.getvalue()
        unwrapped = rendered.replace("\n", "")
        self.assertEqual(rendered.count("[paper-rerun-2026-08-02]"), 2)
        self.assertIn("RUNNING", rendered)
        self.assertIn("SUCCESS", rendered)
        self.assertIn(str(Path("配置/序数模型.json")), unwrapped)
        self.assertIn("encoder=bilstm", rendered)
        self.assertIn("head=ordinal", rendered)
        self.assertIn("task=enabled", rendered)
        self.assertIn("run_elapsed=00:07:08", rendered)
        self.assertIn(str(Path("日志/第03次/训练.log")), unwrapped)
        self.assertIn("第 3 次运行完成", unwrapped)
        self.assertNotIn("高频刷新不应输出", unwrapped)

    def test_failed_status_remains_textual_without_color(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        state = dataclasses.replace(
            build_state(),
            current_run=dataclasses.replace(build_run(), status=RunStatus.FAILED),
            failed=1,
            pending=10,
            latest_message="进程退出码 1",
        )

        with SerialDashboard(console=console) as dashboard:
            dashboard.update(state)

        rendered = output.getvalue()
        self.assertIn("FAILED", rendered)
        self.assertIn("failed=1", rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_cjk_dashboard_fits_narrow_windows_terminal(self) -> None:
        output = io.StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="standard",
            width=72,
            record=True,
        )

        dashboard = SerialDashboard(console=console)
        dashboard.update(build_state())

        rendered = console.export_text()
        self.assertLessEqual(max(cell_len(line) for line in rendered.splitlines()), 72)
        self.assertNotIn("�", rendered)


if __name__ == "__main__":
    unittest.main()
