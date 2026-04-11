"""Rich terminal monitor for evolution progress."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table
from rich.text import Text


class TerminalMonitor:
    """Displays generation-level evolution stats using Rich tables."""

    def __init__(self, verbose: bool = False) -> None:
        self.console = Console()
        self.start_time = time.time()
        self.best_ever: float | None = None
        self.verbose = verbose
        self._gen_count = 0

    def on_generation(
        self,
        generation: int,
        total: int,
        best_fitness: float,
        avg_fitness: float,
        worst_fitness: float,
        phase: str = "evolve",
        population_size: int = 0,
        archive_fill: float | None = None,
        scheduler_stats: dict | None = None,
    ) -> None:
        """Log a completed generation."""
        self._gen_count += 1
        new_best = False
        if self.best_ever is None or best_fitness < self.best_ever:
            self.best_ever = best_fitness
            new_best = True

        elapsed = time.time() - self.start_time

        table = Table(
            title=f"Generation {generation + 1}/{total}",
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Metric", style="cyan", min_width=18)
        table.add_column("Value", style="green", min_width=24)

        # Best fitness with highlight
        best_str = f"{best_fitness:.6f}"
        if new_best:
            best_str += "  *NEW BEST*"
        table.add_row("Best Fitness", best_str)
        table.add_row("Avg Fitness", f"{avg_fitness:.6f}")
        table.add_row("Worst Fitness", f"{worst_fitness:.6f}")
        table.add_row("Best Ever", f"{self.best_ever:.6f}")
        table.add_row("Phase", phase)
        if population_size:
            table.add_row("Population", str(population_size))
        if archive_fill is not None:
            table.add_row("Archive Fill", f"{archive_fill:.1%}")

        # Scheduler stats (benchmark pool rotation, multi-fidelity, etc.)
        if scheduler_stats:
            for key, val in scheduler_stats.items():
                table.add_row(key, str(val))

        table.add_row("Elapsed", _format_time(elapsed))
        self.console.print(table)

    def on_evaluation(
        self, generation: int, genome_idx: int, fitness: float,
    ) -> None:
        """Log a single genome evaluation (verbose mode only)."""
        if not self.verbose:
            return
        self.console.print(
            f"  [dim]gen {generation} | genome {genome_idx:3d} | "
            f"fitness {fitness:.6f}[/dim]"
        )

    def on_complete(
        self, best_fitness: float, total_generations: int, elapsed: float | None = None,
    ) -> None:
        """Print final evolution summary."""
        if elapsed is None:
            elapsed = time.time() - self.start_time

        self.console.print()
        self.console.rule("[bold green]Evolution Complete")
        self.console.print()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column(style="green")
        table.add_row("Best Fitness", f"{best_fitness:.6f}")
        table.add_row("Generations", str(total_generations))
        table.add_row("Total Time", _format_time(elapsed))
        table.add_row(
            "Avg per Gen",
            _format_time(elapsed / total_generations) if total_generations > 0 else "N/A",
        )
        self.console.print(table)
        self.console.print()

    def on_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"[bold red]ERROR:[/bold red] {message}")

    def on_info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f"[bold blue]INFO:[/bold blue] {message}")


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
