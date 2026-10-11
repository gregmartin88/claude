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

Drift check: each month with three earlier months of data is compared with
those three months pooled, per cluster and across all clusters. A
plain-language summary of the most recent month is printed and saved.

Usage:
    python audit_claims.py                       # reads claims.csv
    python audit_claims.py data.csv --out-dir audit_output

Writes to --out-dir:
    cluster_audit.csv, cluster_audit.md
    investigation_rate.png, false_investigation_rate.png, ratio_to_highest.png
    drift_check.csv (every month vs. its baseline), drift_summary.txt
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


# --- Drift check -------------------------------------------------------------
# Each month is compared with the three months before it, pooled. Rate metrics
# use a two-proportion z-test and must also move by a practically meaningful
# amount; with ~100-200 claims per cluster per month, smaller moves are noise.
# Ratio metrics are noisy month to month (the lowest- or highest-rate cluster
# can change from noise alone), so a crossing of the four-fifths line counts
# only when the cluster's own investigation rate also passed the rate test,
# and takes that test's severity. Unbacked crossings stay in the CSV as
# "crossed_not_significant".

BASELINE_MONTHS = 3
RATE_METRICS = {
    # metric: (numerator column, denominator column, minimum absolute change)
    "investigation_rate": ("investigated", "claims", 0.05),
    "false_investigation_rate": ("false_investigations", "investigated", 0.15),
}
RATIO_METRICS = {
    "ratio_to_highest": "below_four_fifths",
    "adverse_impact_ratio": "adverse_below_four_fifths",
}
Z_WATCH, Z_ALERT = 1.96, 2.58
MIN_DENOMINATOR = 15
ALL_CLUSTERS = "All clusters"


def with_total(table):
    total = table[["claims", "investigated", "false_investigations"]].sum()
    total["zip_cluster"] = ALL_CLUSTERS
    total["investigation_rate"] = total["investigated"] / total["claims"]
    total["false_investigation_rate"] = total["false_investigations"] / total["investigated"]
    return pd.concat([table, total.to_frame().T], ignore_index=True).set_index("zip_cluster")


def two_proportion_z(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    return 0.0 if se == 0 else (x1 / n1 - x2 / n2) / se


def drift(df):
    """Compare every month that has a full baseline with its prior three months."""
    months = sorted(df["claim_month"].unique())
    rows = []
    for i in range(BASELINE_MONTHS, len(months)):
        month, baseline = months[i], months[i - BASELINE_MONTHS:i]
        cur = with_total(audit(df[df["claim_month"] == month]))
        base = with_total(audit(df[df["claim_month"].isin(baseline)]))
        for cluster in base.index:
            c, b = cur.loc[cluster], base.loc[cluster]
            common = {"month": month, "baseline": f"{baseline[0]}..{baseline[-1]}",
                      "zip_cluster": cluster}
            for metric, (num, den, min_change) in RATE_METRICS.items():
                if c[den] < MIN_DENOMINATOR:
                    status, z = "too_few", float("nan")
                else:
                    z = two_proportion_z(c[num], c[den], b[num], b[den])
                    big = abs(c[metric] - b[metric]) >= min_change
                    status = ("alert" if big and abs(z) >= Z_ALERT
                              else "watch" if big and abs(z) >= Z_WATCH else "ok")
                rows.append({**common, "metric": metric,
                             "baseline_value": b[metric], "current_value": c[metric],
                             "change": c[metric] - b[metric], "z": z,
                             "current_n": int(c[den]), "status": status})
            if cluster == ALL_CLUSTERS:
                continue
            rate_status = next(r["status"] for r in reversed(rows)
                               if r["metric"] == "investigation_rate")
            for metric, below_col in RATIO_METRICS.items():
                crossed = bool(c[below_col]) != bool(b[below_col])
                status = ("ok" if not crossed
                          else rate_status if rate_status in ("alert", "watch")
                          else "crossed_not_significant")
                rows.append({**common, "metric": metric,
                             "baseline_value": b[metric], "current_value": c[metric],
                             "change": c[metric] - b[metric], "z": float("nan"),
                             "current_n": int(c["claims"]), "status": status})
    return pd.DataFrame(rows)


def month_name(ym):
    return pd.Period(ym, "M").strftime("%B %Y")


def pct(v):
    return f"{v:.0%}"


def area(cluster):
    return "Across all areas" if cluster == ALL_CLUSTERS else f"In area {cluster}"


def describe(r):
    direction = "rose" if r.change > 0 else "fell"
    points = f"{abs(r.change) * 100:.0f} percentage points"
    if r.metric == "investigation_rate":
        return (f"{area(r.zip_cluster)}, the share of claims sent for investigation "
                f"{direction} from {pct(r.baseline_value)} to {pct(r.current_value)}, "
                f"a change of {points}.")
    if r.metric == "false_investigation_rate":
        meaning = ("more investigations found nothing to adjust" if r.change > 0
                   else "fewer investigations found nothing to adjust")
        return (f"{area(r.zip_cluster)}, the share of investigated claims that were "
                f"then paid in full {direction} from {pct(r.baseline_value)} to "
                f"{pct(r.current_value)}, a change of {points}. In other words, "
                f"{meaning}.")
    now_below = r.current_value < FOUR_FIFTHS
    if r.metric == "adverse_impact_ratio":
        return (f"Area {r.zip_cluster} is now investigated far more often than the "
                f"least-investigated area, which breaks the four-fifths fairness "
                f"benchmark. It met the benchmark over the previous three months."
                if now_below else
                f"Area {r.zip_cluster} now meets the four-fifths fairness benchmark "
                f"against the least-investigated area. Over the previous three months "
                f"it did not.")
    return (f"Area {r.zip_cluster} is now investigated at less than four-fifths of "
            f"the rate of the most-investigated area. Over the previous three months "
            f"it was within that range."
            if now_below else
            f"Area {r.zip_cluster} is now investigated at a rate close to the "
            f"most-investigated area (within four-fifths). Over the previous three "
            f"months it was well below it.")


def plain_summary(drift_table):
    """A summary written to be read aloud, covering the most recent month."""
    latest = drift_table["month"].max()
    d = drift_table[drift_table["month"] == latest]
    b0, b1 = d["baseline"].iloc[0].split("..")
    period = f"{month_name(b0)} to {month_name(b1)}"

    alerts = d[d["status"] == "alert"]
    watches = d[d["status"] == "watch"]
    too_few = d[d["status"] == "too_few"]
    persistent = d[(d["metric"] == "adverse_impact_ratio")
                   & (d["current_value"] < FOUR_FIFTHS)
                   & (d["baseline_value"] < FOUR_FIFTHS)]["zip_cluster"].tolist()

    lines = [f"Claims investigation drift check: {month_name(latest)} compared with "
             f"the previous three months, {period}.", ""]
    if alerts.empty and watches.empty:
        lines.append("No meaningful changes were found this month. Every measure is "
                     "within its normal month-to-month range.")
        lines.append("")
    if not alerts.empty:
        lines.append(f"Needs attention ({len(alerts)}). These changes are large and "
                     "very unlikely to be ordinary month-to-month variation:")
        lines += [f"  {i}. {describe(r)}" for i, r in enumerate(alerts.itertuples(), 1)]
        lines.append("")
    if not watches.empty:
        lines.append(f"Worth watching ({len(watches)}). These changes are noticeable "
                     "but could still be chance. We will see whether they repeat "
                     "next month:")
        lines += [f"  {i}. {describe(r)}" for i, r in enumerate(watches.itertuples(), 1)]
        lines.append("")
    if persistent:
        lines.append("Ongoing, not new: areas " + ", ".join(persistent) + " were "
                     "investigated far more often than the least-investigated area in "
                     "both this month and the previous three months. This breaks the "
                     "four-fifths fairness benchmark. It is a continuing pattern, not "
                     "a change this month.")
        lines.append("")
    if not too_few.empty:
        names = ", ".join(sorted(set(too_few["zip_cluster"])))
        lines.append(f"Not enough data this month to judge the share of investigations "
                     f"paid in full for: {names}. Fewer than {MIN_DENOMINATOR} claims "
                     f"were investigated in each.")
    return "\n".join(lines).rstrip() + "\n"


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

    df = pd.read_csv(args.csv)
    table = audit(df)
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

    drift_table = drift(df)
    if drift_table.empty:
        print(f"Drift check skipped: needs at least {BASELINE_MONTHS + 1} months of data.")
    else:
        drift_table.round(4).to_csv(out / "drift_check.csv", index=False)
        summary = plain_summary(drift_table)
        (out / "drift_summary.txt").write_text(summary)
        print(summary)
    print(f"Wrote tables, charts and drift check to {out}/")


if __name__ == "__main__":
    main()
