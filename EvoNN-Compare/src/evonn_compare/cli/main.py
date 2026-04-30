"""EvoNN-Compare CLI."""

from __future__ import annotations

import typer

from evonn_compare.cli.campaign import campaign
from evonn_compare.cli.campaign_state import campaign_inspect, campaign_stop
from evonn_compare.cli.compare import compare
from evonn_compare.cli.dashboard import dashboard
from evonn_compare.cli.fair_matrix import fair_matrix
from evonn_compare.cli.historical_baseline import historical_baseline
from evonn_compare.cli.hybrid import run as hybrid_run
from evonn_compare.cli.seeded_compare import seeded_compare
from evonn_compare.cli.transfer_regimes import transfer_regimes
from evonn_compare.cli.trend_report import trend_report
from evonn_compare.cli.validate import validate
from evonn_compare.cli.workspace_report import workspace_report

app = typer.Typer(
    name="evonn-compare",
    help="Compare Prism, Topograph, Stratograph, Primordia, and baselines via normalized export contracts.",
)

app.command("compare")(compare)
app.command("validate")(validate)
app.command("campaign")(campaign)
app.command("campaign-inspect")(campaign_inspect)
app.command("campaign-stop")(campaign_stop)
app.command("fair-matrix")(fair_matrix)
app.command("seeded-compare")(seeded_compare)
app.command("transfer-regimes")(transfer_regimes)
app.command("historical-baseline")(historical_baseline)
app.command("trend-report")(trend_report)
app.command("dashboard")(dashboard)
app.command("workspace-report")(workspace_report)

hybrid_app = typer.Typer(help="Fused hybrid commands")
hybrid_app.command("run")(hybrid_run)
app.add_typer(hybrid_app, name="hybrid")
