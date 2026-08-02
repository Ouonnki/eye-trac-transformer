"""Rich dashboard for serial paper-experiment reruns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Final, Mapping, Optional, Tuple, Type

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text


_ACCENT: Final = "bold cyan"
_ACTIVE: Final = "bold blue"
_SUCCESS: Final = "bold green"
_FAILURE: Final = "bold red"
_PENDING: Final = "yellow"
_MUTED: Final = "dim"
_PANEL_PADDING: Final = (0, 1)
_TABLE_PADDING: Final = (0, 2)


class RunStatus(str, Enum):
    """Closed set of serial-run lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


_STATUS_LABELS: Final[Mapping[RunStatus, str]] = MappingProxyType(
    {
        RunStatus.PENDING: "PENDING",
        RunStatus.RUNNING: "RUNNING",
        RunStatus.SUCCESS: "SUCCESS",
        RunStatus.FAILED: "FAILED",
    }
)
_STATUS_STYLES: Final[Mapping[RunStatus, str]] = MappingProxyType(
    {
        RunStatus.PENDING: _PENDING,
        RunStatus.RUNNING: _ACTIVE,
        RunStatus.SUCCESS: _SUCCESS,
        RunStatus.FAILED: _FAILURE,
    }
)


@dataclass(frozen=True)
class BatchView:
    """Batch identity and timing shown by dashboard."""

    __slots__ = ("identity", "total_runs", "elapsed")

    identity: str
    total_runs: int
    elapsed: timedelta


@dataclass(frozen=True)
class RunView:
    """Current run identity, configuration, and progress."""

    __slots__ = (
        "ordinal",
        "source",
        "encoder",
        "head",
        "use_task_embedding",
        "status",
        "epoch",
        "total_epochs",
        "elapsed",
        "log_path",
    )

    ordinal: int
    source: Path
    encoder: str
    head: str
    use_task_embedding: bool
    status: RunStatus
    epoch: int
    total_epochs: int
    elapsed: timedelta
    log_path: Path


@dataclass(frozen=True)
class DashboardState:
    """Immutable snapshot consumed by dashboard."""

    __slots__ = (
        "batch",
        "current_run",
        "succeeded",
        "failed",
        "pending",
        "latest_message",
    )

    batch: BatchView
    current_run: Optional[RunView]
    succeeded: int
    failed: int
    pending: int
    latest_message: str

    @property
    def completed(self) -> int:
        return self.succeeded + self.failed


def _status_label(status: RunStatus) -> str:
    return _STATUS_LABELS[status]


def _status_style(status: RunStatus) -> str:
    return _STATUS_STYLES[status]


def _duration(value: timedelta) -> str:
    seconds = max(0, int(value.total_seconds()))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _details(state: DashboardState) -> Panel:
    table = Table.grid(padding=_TABLE_PADDING, expand=True)
    table.add_column(style=_MUTED, no_wrap=True)
    table.add_column(ratio=1, overflow="fold")
    run = state.current_run
    if run is None:
        table.add_row("Run", Text(f"- / {state.batch.total_runs}", style=_PENDING))
        table.add_row("Status", Text("PENDING", style=_PENDING))
    else:
        task_flag = Text("Enabled", style=_SUCCESS) if run.use_task_embedding else Text("Disabled", style=_MUTED)
        table.add_row("Run", Text(f"{run.ordinal} / {state.batch.total_runs}", style=_ACCENT))
        table.add_row("Source", Text(str(run.source)))
        table.add_row("Encoder / head", Text(f"{run.encoder} / {run.head}"))
        table.add_row("Task embedding", task_flag)
        table.add_row("Status", Text(_status_label(run.status), style=_status_style(run.status)))
        table.add_row("Elapsed", Text(f"batch {_duration(state.batch.elapsed)}  |  run {_duration(run.elapsed)}"))
        table.add_row("Log", Text(str(run.log_path), overflow="fold"))
    return Panel(
        table,
        title="[bold]Current run[/bold]",
        border_style="cyan",
        padding=_PANEL_PADDING,
        box=box.ROUNDED,
    )


def _progress(state: DashboardState) -> Panel:
    progress = Progress(
        TextColumn("{task.description}", style="bold", justify="right"),
        BarColumn(complete_style="blue", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}", style=_MUTED),
        auto_refresh=False,
        expand=True,
    )
    run = state.current_run
    epoch = 0 if run is None else run.epoch
    total_epochs = 1 if run is None else run.total_epochs
    progress.add_task("Epoch", total=total_epochs, completed=epoch)
    counts = f"Overall  [green]Success {state.succeeded}[/]  [red]Failed {state.failed}[/]  [yellow]Pending {state.pending}[/]"
    progress.add_task(counts, total=state.batch.total_runs, completed=state.completed)
    return Panel(
        progress,
        title="[bold]Progress[/bold]",
        border_style="blue",
        padding=_PANEL_PADDING,
        box=box.ROUNDED,
    )


def _render(state: DashboardState) -> Group:
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text.assemble(("SERIAL EXPERIMENT  ", _MUTED), (state.batch.identity, _ACCENT)),
        Text(f"{state.completed} / {state.batch.total_runs} complete", style="bold"),
    )
    return Group(
        Panel(header, border_style="cyan", padding=_PANEL_PADDING, box=box.HEAVY),
        _details(state),
        _progress(state),
        Panel(
            Text(state.latest_message or "Waiting for runner output", overflow="fold"),
            title="[bold]Latest message[/bold]",
            border_style="bright_black",
            padding=_PANEL_PADDING,
            box=box.ROUNDED,
        ),
    )


class SerialDashboard:
    """Mutable Live lifecycle around immutable dashboard snapshots."""

    def __init__(self, console: Optional[Console] = None, refresh_per_second: float = 8.0) -> None:
        self._console = console or Console(highlight=False)
        self._refresh_per_second = refresh_per_second
        self._live: Optional[Live] = None
        self._active = False
        self._transition: Optional[
            Tuple[str, Optional[int], Optional[RunStatus], int, int, int]
        ] = None

    def __enter__(self) -> SerialDashboard:
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._active = False

    def update(self, state: DashboardState) -> None:
        """Display latest snapshot without accumulating refresh lines."""
        if self._console.is_terminal:
            self._update_live(state)
        else:
            self._update_plain(state)

    def _update_live(self, state: DashboardState) -> None:
        renderable = _render(state)
        if self._live is not None:
            self._live.update(renderable, refresh=True)
        elif self._active:
            self._live = Live(
                renderable,
                console=self._console,
                refresh_per_second=self._refresh_per_second,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="ellipsis",
            )
            self._live.start(refresh=True)
        else:
            self._console.print(renderable)

    def _update_plain(self, state: DashboardState) -> None:
        run = state.current_run
        ordinal = None if run is None else run.ordinal
        status = None if run is None else run.status
        transition = (state.batch.identity, ordinal, status, state.succeeded, state.failed, state.pending)
        if transition == self._transition:
            return
        self._transition = transition
        status_text = "PENDING" if status is None else _status_label(status)
        run_text = "-" if ordinal is None else str(ordinal)
        if run is None:
            run_details = "source=- encoder=- head=- task=- epoch=- run_elapsed=- log=-"
        else:
            task_flag = "enabled" if run.use_task_embedding else "disabled"
            run_details = (
                f"source={run.source} encoder={run.encoder} head={run.head} task={task_flag} "
                f"epoch={run.epoch}/{run.total_epochs} run_elapsed={_duration(run.elapsed)} log={run.log_path}"
            )
        line = (
            f"[{state.batch.identity}] {status_text} run={run_text}/{state.batch.total_runs} "
            f"{run_details} "
            f"done={state.completed} success={state.succeeded} failed={state.failed} pending={state.pending} "
            f"batch_elapsed={_duration(state.batch.elapsed)} | {state.latest_message}"
        )
        self._console.print(Text(line))
