from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Primitive-first evolutionary search for EvoNN.")
console = Console()


@app.callback(invoke_without_command=True)
def main() -> None:
    """Show a compact overview when called without a subcommand."""
    table = Table(title="Primordia")
    table.add_column("Area")
    table.add_column("Status")
    table.add_row("Package scaffold", "ready")
    table.add_row("Primitive genome", "planned")
    table.add_row("Cheap evaluation lane", "planned")
    table.add_row("Motif bank export", "planned")
    console.print(table)
    console.print("See EvoNN-Primordia/VISION.md and IMPLEMENTATION_PLAN.md for roadmap.")


if __name__ == "__main__":
    app()
