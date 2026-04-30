"""HTML renderer for the canonical performance review payload."""

from __future__ import annotations

import html
from typing import Any


def render_performance_dashboard_html(payload: dict[str, Any]) -> str:
    comparison = payload["primary_comparison"]
    history_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['label']))}</td>"
        f"<td>{html.escape(str(row['outcome']))}</td>"
        f"<td>{html.escape(', '.join(row.get('candidate_accounting_tags') or ['full_budget']))}</td>"
        f"<td>{html.escape(str(row['verdict']))}</td>"
        f"<td>{html.escape(str(row['matched_case_count']))}</td>"
        f"<td>{_float_cell(row['median_wall_clock_delta_pct'], suffix='%')}</td>"
        f"<td>{_float_cell(row['median_evals_per_second_delta_pct'], suffix='%')}</td>"
        f"<td>{html.escape(str(row['quality_regression_count']))}</td>"
        f"<td>{html.escape(str(row['trust_regression_count']))}</td>"
        "</tr>"
        for row in payload["optimization_history"]
    )
    dataset_cards = "\n".join(
        "<article class='metric-card'>"
        f"<p class='eyebrow'>{html.escape(str(dataset['label']))}</p>"
        f"<h3>{html.escape(str(dataset['outcome']))}</h3>"
        f"<p><strong>Rows</strong> {html.escape(str(dataset['row_count']))}</p>"
        f"<p><strong>Accounting</strong> {html.escape(', '.join(dataset.get('accounting_tags') or ['full_budget']))}</p>"
        f"<p><strong>Source</strong> <code>{html.escape(str(dataset['source_path']))}</code></p>"
        "</article>"
        for dataset in payload["datasets"]
    )
    slice_rows = []
    for dataset in payload["datasets"]:
        for row in dataset["slices"]:
            slice_rows.append(
                "<tr>"
                f"<td>{html.escape(str(dataset['label']))}</td>"
                f"<td>{html.escape(str(row['system']))}</td>"
                f"<td>{html.escape(str(row['backend_label']))}</td>"
                f"<td>{html.escape(str(row['pack_name']))}</td>"
                f"<td>{html.escape(str(row['budget']))}</td>"
                f"<td>{html.escape(str(row['cache_mode']))}</td>"
                f"<td>{html.escape(', '.join(row.get('accounting_tags') or ['full_budget']))}</td>"
                f"<td>{_float_cell(row['median_wall_clock_seconds'])}</td>"
                f"<td>{_float_cell(row['median_evals_per_second'])}</td>"
                f"<td>{html.escape(str(row['total_failure_count']))}</td>"
                f"<td>{_float_cell(row['median_quality_delta_vs_baseline'])}</td>"
                "</tr>"
            )
    delta_rows = []
    for row in comparison["deltas"]:
        delta_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['system']))}</td>"
            f"<td>{html.escape(str(row['backend_label']))}</td>"
            f"<td>{html.escape(str(row['pack_name']))}</td>"
            f"<td>{html.escape(str(row['budget']))}</td>"
            f"<td>{html.escape(str(row['cache_mode']))}</td>"
            f"<td>{html.escape(', '.join(row.get('candidate_accounting_tags') or ['full_budget']))}</td>"
            f"<td>{html.escape(str(row['matched_case_count']))}</td>"
            f"<td>{_float_cell(row['wall_clock_delta_pct'], suffix='%')}</td>"
            f"<td>{_float_cell(row['evals_per_second_delta_pct'], suffix='%')}</td>"
            f"<td>{html.escape(str(row['failure_count_delta']))}</td>"
            f"<td>{html.escape(str(row['verdict']))}</td>"
            "</tr>"
        )
    summary = comparison["summary"]
    compare_label = comparison["candidate_label"] or "none"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>EvoNN Performance Review</title>
    <style>
      :root {{
        color-scheme: light;
        font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
        color: #12202f;
        background: #f5f0e7;
      }}
      body {{
        margin: 0;
        background:
          radial-gradient(circle at top left, rgba(181, 108, 36, 0.20), transparent 33%),
          linear-gradient(180deg, #fbf6ef 0%, #efe1cf 100%);
      }}
      main {{
        max-width: 1240px;
        margin: 0 auto;
        padding: 2rem 1.25rem 3rem;
      }}
      .hero {{
        background: rgba(255, 251, 245, 0.92);
        border: 1px solid #d7c2a3;
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 20px 40px rgba(61, 41, 13, 0.10);
      }}
      .eyebrow {{
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        color: #8a5320;
      }}
      .hero-grid, .cards {{
        display: grid;
        gap: 1rem;
      }}
      .hero-grid {{
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}
      .cards {{
        margin-top: 1rem;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }}
      .metric-card, .panel {{
        background: rgba(255, 251, 245, 0.92);
        border: 1px solid #d7c2a3;
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        vertical-align: top;
        border-bottom: 1px solid #e5d7c4;
        padding: 0.65rem 0.45rem;
      }}
      code {{
        font-family: "Iosevka", "SFMono-Regular", monospace;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <p class="eyebrow">EvoNN Compare</p>
        <h1>Performance Review Surface</h1>
        <p>This dashboard keeps raw <code>perf_rows.jsonl</code> as the source of truth while surfacing grouped runtime slices and before/after deltas for optimization review.</p>
        <div class="hero-grid">
          <div><strong>Generated</strong><br>{html.escape(str(payload['generated_at']))}</div>
          <div><strong>Baseline</strong><br>{html.escape(str(payload['baseline_label']))}</div>
          <div><strong>Primary Compare</strong><br>{html.escape(str(compare_label))}</div>
          <div><strong>History Entries</strong><br>{html.escape(str(len(payload['optimization_history'])))}</div>
        </div>
      </section>

      <section class="cards">
        {dataset_cards}
      </section>

      <section class="panel">
        <h2>Grouped Runtime Slices</h2>
        <p>Each row is grouped by backend, budget, pack, and cache mode so runtime, cache, failure, and quality fields can be reviewed without leaving the canonical schema.</p>
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>System</th>
              <th>Backend</th>
              <th>Pack</th>
              <th>Budget</th>
              <th>Cache</th>
              <th>Accounting</th>
              <th>Median Wall (s)</th>
              <th>Median Eval/s</th>
              <th>Failures</th>
              <th>Quality Delta</th>
            </tr>
          </thead>
          <tbody>
            {' '.join(slice_rows)}
          </tbody>
        </table>
      </section>

      <section class="panel">
        <h2>Before/After Delta View</h2>
        <p><strong>Verdict:</strong> {html.escape(str(summary['verdict']))}</p>
        <p><strong>Baseline Accounting:</strong> {html.escape(', '.join(summary.get('baseline_accounting_tags') or ['full_budget']))}</p>
        <p><strong>Candidate Accounting:</strong> {html.escape(', '.join(summary.get('candidate_accounting_tags') or ['full_budget']))}</p>
        <p><strong>Median Wall Delta:</strong> {_float_cell(summary['median_wall_clock_delta_pct'], suffix='%')}</p>
        <p><strong>Median Eval/s Delta:</strong> {_float_cell(summary['median_evals_per_second_delta_pct'], suffix='%')}</p>
        <p><strong>Candidate Deduplicated Slots:</strong> {html.escape(str(summary.get('candidate_deduplicated_evaluations') or 0))} |
        <strong>Proxy-Screened Slots:</strong> {html.escape(str(summary.get('candidate_screened_evaluations') or 0))} |
        <strong>Reduced-Fidelity Slots:</strong> {html.escape(str(summary.get('candidate_reduced_fidelity_evaluations') or 0))}</p>
        <p><strong>Quality Regressions:</strong> {html.escape(str(summary['quality_regression_count']))} | <strong>Trust Regressions:</strong> {html.escape(str(summary['trust_regression_count']))}</p>
        <table>
          <thead>
            <tr>
              <th>System</th>
              <th>Backend</th>
              <th>Pack</th>
              <th>Budget</th>
              <th>Cache</th>
              <th>Candidate Accounting</th>
              <th>Matched Cases</th>
              <th>Wall Delta %</th>
              <th>Eval/s Delta %</th>
              <th>Failure Delta</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {' '.join(delta_rows) if delta_rows else '<tr><td colspan="11">No candidate comparison selected.</td></tr>'}
          </tbody>
        </table>
      </section>

      <section class="panel">
        <h2>Optimization History</h2>
        <p>Accepted and scrapped optimization branches stay visible here so fake wins do not get rediscovered later from anecdotal timing.</p>
        <table>
          <thead>
            <tr>
              <th>Label</th>
              <th>Outcome</th>
              <th>Accounting</th>
              <th>Verdict</th>
              <th>Matched Cases</th>
              <th>Wall Delta %</th>
              <th>Eval/s Delta %</th>
              <th>Quality Regressions</th>
              <th>Trust Regressions</th>
            </tr>
          </thead>
          <tbody>
            {history_rows or '<tr><td colspan="9">No optimization history loaded.</td></tr>'}
          </tbody>
        </table>
      </section>
    </main>
  </body>
</html>
"""


def _float_cell(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "---"
    return f"{float(value):.2f}{suffix}"
