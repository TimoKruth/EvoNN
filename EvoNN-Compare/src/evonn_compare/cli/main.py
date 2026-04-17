"""EvoNN-Compare CLI."""

from __future__ import annotations

import typer

from evonn_compare.cli.campaign import campaign
from evonn_compare.cli.compare import compare
from evonn_compare.cli.hybrid import run as hybrid_run
from evonn_compare.cli.validate import validate

app = typer.Typer(
    name="evonn-compare",
    help="Compare Prism, Topograph, and fused hybrid runs via normalized export contracts.",
)

app.command("compare")(compare)
app.command("validate")(validate)
app.command("campaign")(campaign)

hybrid_app = typer.Typer(help="Fused hybrid commands")
hybrid_app.command("run")(hybrid_run)
app.add_typer(hybrid_app, name="hybrid")
