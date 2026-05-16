"""Rich terminal monitor for Prism evolution progress."""

from __future__ import annotations

import time
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

_NEW_BEST_SUFFIX = "  *NEW BEST*"


class TerminalMonitor:
    """Displays generation-level evolution stats using Rich tables."""

    def __init__(self) -> None:
        self.console = Console()
        self.start_time = time.time()
        self.best_ever: float | None = None

    def on_generation(
        self,
        gen: int,
        total: int,
        best_quality: float,
        avg_quality: float,
        families_active: Sequence[str],
        population_size: int,
        elapsed: float,
    ) -> None:
        """Log a completed generation."""
        new_best = self._update_best(best_quality)

        table = self._generation_table(gen, total)

        best_str = f"{best_quality:.6f}"
        if new_best:
            best_str += _NEW_BEST_SUFFIX
        table.add_row("Best Quality", best_str)
        table.add_row("Avg Quality", f"{avg_quality:.6f}")
        table.add_row("Best Ever", f"{self.best_ever:.6f}")
        table.add_row("Population", str(population_size))
        table.add_row("Families", _format_families(families_active))
        table.add_row("Elapsed", _format_time(elapsed))

        self.console.print(table)

    def on_complete(
        self,
        best_quality: float,
        total_gens: int,
        elapsed: float,
    ) -> None:
        """Print final evolution summary."""
        self.console.print()
        self.console.rule("[bold green]Evolution Complete")
        self.console.print()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="green")
        table.add_row("Best Quality", f"{best_quality:.6f}")
        table.add_row("Generations", str(total_gens))
        table.add_row("Total Time", _format_time(elapsed))
        table.add_row(
            "Avg per Gen",
            _format_time(elapsed / total_gens) if total_gens > 0 else "N/A",
        )
        self.console.print(table)
        self.console.print()

    def on_info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f"[bold blue]INFO:[/bold blue] {message}")

    def on_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"[bold red]ERROR:[/bold red] {message}")

    def _update_best(self, best_quality: float) -> bool:
        """Update the tracked best quality and report whether it improved."""
        if self.best_ever is not None and best_quality <= self.best_ever:
            return False
        self.best_ever = best_quality
        return True

    @staticmethod
    def _generation_table(gen: int, total: int) -> Table:
        """Create the standard generation metrics table."""
        table = Table(
            title=f"Generation {gen + 1}/{total}",
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Metric", style="cyan", min_width=18)
        table.add_column("Value", style="green", min_width=24)
        return table


def _format_families(families: Sequence[str]) -> str:
    """Format active families for display."""
    return ", ".join(sorted(families)) or "none"


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"
