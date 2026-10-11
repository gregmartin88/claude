#!/usr/bin/env python3
"""Audit investigation-queue routing by zip_cluster.

Reads the CSV from generate_claims.py and reports, per zip_cluster:

    investigation_rate        share of claims routed to special_investigation
    false_investigation_rate  share of investigated claims whose final_outcome
                              is "approved" (paid without adjustment)
    ratio_to_highest          investigation_rate / highest cluster's rate,
                              flagged when below 0.8 (four-fifths rule)
    adverse_impact_ratio      lowest cluster's rate / investigation_rate.
                              Investigation is an adverse outcome, so this is
                              the conventional four-fifths reading: below 0.8
                              means the cluster is investigated
                              disproportionately often.

Usage:
    python audit_claims.py                       # reads claims.csv
    python audit_claims.py data.csv --out-dir audit_output

Writes to --out-dir:
    cluster_audit.csv, cluster_audit.md
    investigation_rate.png, false_investigation_rate.png, ratio_to_highest.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INVESTIGATION_QUEUE = "special_investigation"
PAID_IN_FULL = "approved"
FOUR_FIFTHS = 0.8

# Reference palette (dataviz skill): one series per chart, so one hue.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def audit(df):
    df = df.assign(
        investigated=df["assigned_queue"] == INVESTIGATION_QUEUE,
        false_investigation=(df["assigned_queue"] == INVESTIGATION_QUEUE)
        & (df["final_outcome"] == PAID_IN_FULL),
    )
    t = df.groupby("zip_cluster").agg(
        claims=("claim_id", "size"),
        investigated=("investigated", "sum"),
        false_investigations=("false_investigation", "sum"),
    )
    t["investigation_rate"] = t["investigated"] / t["claims"]
    t["false_investigation_rate"] = t["false_investigations"] / t["investigated"]

    max_rate = t["investigation_rate"].max()
    min_rate = t["investigation_rate"].min()
    t["ratio_to_highest"] = t["investigation_rate"] / max_rate
    t["below_four_fifths"] = t["ratio_to_highest"] < FOUR_FIFTHS
    t["adverse_impact_ratio"] = min_rate / t["investigation_rate"]
    t["adverse_below_four_fifths"] = t["adverse_impact_ratio"] < FOUR_FIFTHS
    return t.sort_index().reset_index()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_chart(table, column, title, subtitle, path, pct=True, hatch_mask=None, ref_line=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    values = table[column]
    bars = ax.bar(table["zip_cluster"], values, width=0.6, color=SERIES,
                  edgecolor=SURFACE, linewidth=2)
    if hatch_mask is not None:
        for bar, flag in zip(bars, hatch_mask):
            if flag:
                bar.set_hatch("///")
                bar.set_facecolor("none")
                bar.set_edgecolor(SERIES)
                bar.set_linewidth(1.5)

    fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.2f}")
    for bar, v in zip(bars, values):
        ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)

    if ref_line is not None:
        ax.axhline(ref_line, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.annotate(f"four-fifths threshold ({ref_line:g})", (0.01, ref_line),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=TEXT_SECONDARY)

    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(values.max(), ref_line or 0) * 1.18)
    ax.set_xlabel("zip_cluster", color=TEXT_SECONDARY)
    fig.text(0.02, 0.96, title, fontsize=12, weight="bold", color=TEXT_PRIMARY, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=9, color=TEXT_SECONDARY, va="top")
    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.13)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def to_markdown(table):
    cols = ["zip_cluster", "claims", "investigated", "investigation_rate",
            "false_investigations", "false_investigation_rate",
            "ratio_to_highest", "below_four_fifths",
            "adverse_impact_ratio", "adverse_below_four_fifths"]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table[cols].iterrows():
        cells = [
            r.zip_cluster, f"{r.claims:,}", f"{r.investigated:,}",
            f"{r.investigation_rate:.1%}", f"{r.false_investigations:,}",
            f"{r.false_investigation_rate:.1%}", f"{r.ratio_to_highest:.2f}",
            "yes" if r.below_four_fifths else "no",
            f"{r.adverse_impact_ratio:.2f}",
            "yes" if r.adverse_below_four_fifths else "no",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", nargs="?", default="claims.csv")
    parser.add_argument("--out-dir", default="audit_output")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = audit(pd.read_csv(args.csv))
    table.round(4).to_csv(out / "cluster_audit.csv", index=False)
    (out / "cluster_audit.md").write_text(to_markdown(table))

    n = int(table["claims"].sum())
    top = table.loc[table["investigation_rate"].idxmax(), "zip_cluster"]
    bar_chart(table, "investigation_rate",
              "Investigation-queue selection rate by zip cluster",
              f"Share of claims routed to special_investigation (n = {n:,} claims)",
              out / "investigation_rate.png")
    bar_chart(table, "false_investigation_rate",
              "False-investigation rate by zip cluster",
              "Share of investigated claims later approved without adjustment",
              out / "false_investigation_rate.png")
    bar_chart(table, "ratio_to_highest",
              f"Investigation rate relative to highest-rate cluster ({top})",
              "Hatched bars fall below the four-fifths (0.8) threshold",
              out / "ratio_to_highest.png", pct=False,
              hatch_mask=table["below_four_fifths"], ref_line=FOUR_FIFTHS)

    print(to_markdown(table))
    print(f"Wrote table and charts to {out}/")


if __name__ == "__main__":
    main()
