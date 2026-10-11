#!/usr/bin/env python3
"""Generate synthetic auto claims with a planted ZIP-tier routing bias (Queue A/B).

Test fixture for bias-detection and fairness-audit tooling. Every row is
synthetic; no real people, policies or ZIP codes are represented.

Output columns:
    claim_id, claim_date, claim_month, claim_amount, zip3, zip_tier,
    vehicle_age, policy_tenure_years, adjuster_queue, final_outcome

    adjuster_queue: A = Fast-Track, B = Special Investigation / High-Scrutiny
    final_outcome:  approved / partial / denied

Usage:
    python generate_tiered_claims.py                    # writes claims_tiered.csv
    python generate_tiered_claims.py --rows 8000 --seed 7 --out data.csv
    python generate_tiered_claims.py --verify           # print bias diagnostics

-------------------------------------------------------------------------------
HOW THE BIAS IS BUILT (ground truth for later detection)
-------------------------------------------------------------------------------
No column flags the bias; it lives only in the routing rule below.

1. zip_tier (1..5) is drawn with weights [0.24, 0.24, 0.20, 0.18, 0.14].
   Each tier owns four synthetic 3-digit prefixes (zip3), so tier can also be
   recovered from zip3 alone (ZIP3_BY_TIER below).

2. vehicle_age (years, 0..25) leans older in higher tiers, so a naive
   tier comparison is partly confounded:
       vehicle_age ~ Gamma(shape=3, scale=(4 + 1.0*(tier-1))/3)

3. policy_tenure_years ~ Gamma(2, 3), clipped [0.1, 30]; independent of tier.

4. claim_amount (USD) is lognormal and independent of tier except through
   vehicle_age:
       log(amount) ~ N(8.2 - 0.015*vehicle_age, 0.8), clipped [200, 80000]

5. Latent legitimate risk r (never emitted) depends only on legitimate
   features, never on tier:
       r = -2.4 + 0.06*vehicle_age + 0.55*log10(amount/1000)
           - 0.05*tenure + N(0, 0.6)

6. adjuster_queue: P(Queue B) = sigmoid(r + 0.4 + BIAS * [zip_tier in {4, 5}])
   with BIAS = 1.1 by default (--bias). That is the planted direct effect:
   at the same amount, vehicle age and tenure, a tier 4/5 claim has
   ~e^1.1 = 3x the odds of being sent to Queue B.

7. final_outcome depends on the latent risk r plus a queue effect, never on
   tier:
       denied  logit = -2.6 + 1.6*r_centered + 0.5*[queue B]
       partial logit = -1.5 + 0.5*r_centered + 0.4*[queue B]
       approved logit = 0
   The extra tier 4/5 claims sent to Queue B are not riskier, so Queue B in
   tiers 4/5 shows a lower denial rate (a higher "investigated then paid"
   rate) than Queue B in tiers 1-3. That is a second signature of
   unwarranted routing.

What you should be able to detect (seed 42, default settings; run --verify):
  - Queue B rate jumps between tiers 3 and 4 (17% -> 39%), not gradually
    like vehicle age.
  - A logistic regression of queue B on log amount, vehicle age and tenure
    leaves a tier 4/5 coefficient of ~1.02 (odds ratio ~2.8, z ~15).
  - Within all 20 amount-quintile x vehicle-age-band cells, tier 4/5 claims
    go to Queue B more often (mean gap ~21 points).
  - Within Queue B, tier 4/5 claims are denied less often (15.9% vs 17.6%),
    even though their older vehicles would push the other way; with
    --bias 0 the order flips (19.8% vs 17.6%). This signal is modest.
  - Control: --bias 0 gives a tier 4/5 odds ratio of ~0.97 (z ~ -0.4).
-------------------------------------------------------------------------------
"""

import argparse

import numpy as np
import pandas as pd

N_ROWS = 6000
START, END = "2026-01-01", "2026-06-30"
TIER_WEIGHTS = [0.24, 0.24, 0.20, 0.18, 0.14]
ZIP3_BY_TIER = {
    1: ["101", "114", "127", "133"],
    2: ["205", "218", "222", "239"],
    3: ["304", "311", "326", "348"],
    4: ["402", "417", "429", "436"],
    5: ["508", "515", "521", "547"],
}
BIASED_TIERS = (4, 5)
DEFAULT_BIAS = 1.1
OUTCOMES = np.array(["approved", "partial", "denied"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=N_ROWS, seed=42, bias=DEFAULT_BIAS):
    rng = np.random.default_rng(seed)

    tier = rng.choice(np.arange(1, 6), size=n, p=TIER_WEIGHTS)
    zip3 = np.array([rng.choice(ZIP3_BY_TIER[t]) for t in tier])

    vehicle_age = np.clip(np.round(rng.gamma(3, (4 + (tier - 1)) / 3)), 0, 25).astype(int)
    tenure = np.round(np.clip(rng.gamma(2, 3, n), 0.1, 30), 1)
    amount = np.round(
        np.clip(np.exp(rng.normal(8.2 - 0.015 * vehicle_age, 0.8)), 200, 80_000), 2
    )

    # Latent legitimate risk (step 5) - no tier term.
    log_amt = np.log10(amount / 1000)
    risk = -2.4 + 0.06 * vehicle_age + 0.55 * log_amt - 0.05 * tenure + rng.normal(0, 0.6, n)

    # Routing (step 6) - the only place tier enters.
    biased = np.isin(tier, BIASED_TIERS)
    queue_b = rng.random(n) < sigmoid(risk + 0.4 + bias * biased)
    queue = np.where(queue_b, "B", "A")

    # Outcome (step 7) - risk and queue only.
    rc = risk - risk.mean()
    logits = np.column_stack(
        [np.zeros(n), -1.5 + 0.5 * rc + 0.4 * queue_b, -2.6 + 1.6 * rc + 0.5 * queue_b]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    outcome = OUTCOMES[(probs.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)]

    days = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1
    dates = pd.Timestamp(START) + pd.to_timedelta(rng.integers(0, days, n), unit="D")

    df = pd.DataFrame(
        {
            "claim_date": dates.strftime("%Y-%m-%d"),
            "claim_month": dates.strftime("%Y-%m"),
            "claim_amount": amount,
            "zip3": zip3,
            "zip_tier": tier,
            "vehicle_age": vehicle_age,
            "policy_tenure_years": tenure,
            "adjuster_queue": queue,
            "final_outcome": outcome,
        }
    ).sort_values("claim_date", kind="stable").reset_index(drop=True)
    df.insert(0, "claim_id", [f"TC-{i:06d}" for i in range(1, n + 1)])
    return df


def logit_fit(X, y, iters=50):
    """Logistic regression by IRLS. Returns coefficients and standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ beta)
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def verify(df):
    b = (df["adjuster_queue"] == "B").astype(float)
    hi = df["zip_tier"].isin(BIASED_TIERS)

    print("\nQueue B rate and mean covariates by zip_tier:")
    print(
        df.assign(queue_b=b)
        .groupby("zip_tier")
        .agg(
            claims=("claim_id", "size"),
            queue_b_rate=("queue_b", "mean"),
            mean_amount=("claim_amount", "mean"),
            mean_vehicle_age=("vehicle_age", "mean"),
        )
        .round(3)
        .to_string()
    )

    X = np.column_stack(
        [
            np.ones(len(df)),
            np.log10(df["claim_amount"] / 1000),
            df["vehicle_age"],
            df["policy_tenure_years"],
            hi.astype(float),
        ]
    )
    names = ["intercept", "log10(amount/1000)", "vehicle_age", "tenure", "tier_4_or_5"]
    beta, se = logit_fit(X, b.to_numpy())
    print("\nLogit P(Queue B) ~ amount + vehicle_age + tenure + tier_4_or_5:")
    for name, coef, s in zip(names, beta, se):
        print(f"  {name:<20} {coef:+.3f}  (se {s:.3f}, z {coef / s:+.1f})")
    print(f"  => tier 4/5 odds ratio, controlling for the rest: {np.exp(beta[-1]):.2f}")

    cells = df.assign(
        queue_b=b,
        hi=hi,
        amt_q=pd.qcut(df["claim_amount"], 5, labels=False),
        age_band=pd.cut(df["vehicle_age"], [-1, 3, 7, 12, 25], labels=False),
    )
    strat = cells.pivot_table(
        index=["amt_q", "age_band"], columns="hi", values="queue_b", aggfunc="mean"
    ).dropna()
    gap = strat[True] - strat[False]
    print(
        f"\nStratified (amount quintile x vehicle-age band, {len(gap)} cells): "
        f"tier 4/5 Queue B rate higher in {(gap > 0).sum()}/{len(gap)} cells, "
        f"mean gap {gap.mean():+.3f}"
    )

    qb = df[b == 1]
    denied = (qb["final_outcome"] == "denied").groupby(qb["zip_tier"].isin(BIASED_TIERS)).mean()
    print(
        f"\nDenial rate within Queue B: tiers 1-3 {denied[False]:.3f}, "
        f"tiers 4-5 {denied[True]:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="claims_tiered.csv")
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bias", type=float, default=DEFAULT_BIAS,
                        help="log-odds boost toward Queue B for tiers 4-5 (0 = none)")
    parser.add_argument("--verify", action="store_true", help="print bias diagnostics")
    args = parser.parse_args()

    df = generate(args.rows, args.seed, args.bias)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} ({df['claim_date'].min()} to {df['claim_date'].max()})")
    if args.verify:
        verify(df)


if __name__ == "__main__":
    main()
