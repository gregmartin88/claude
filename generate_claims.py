#!/usr/bin/env python3
"""Generate synthetic auto insurance claims data with a planted proxy-bias pathway.

Output columns (default):
    claim_id, claim_month, claim_amount, zip_cluster, vehicle_age,
    policy_tenure_years, assigned_queue, final_outcome

Usage:
    python generate_claims.py                         # writes claims.csv
    python generate_claims.py --out data.csv --seed 7
    python generate_claims.py --expose-prior-claims   # also emits prior_claims_3y

-------------------------------------------------------------------------------
HOW THE PROXY-BIAS PATHWAY IS BUILT (ground truth for later detection)
-------------------------------------------------------------------------------
Causal structure:

    zip_cluster ──> latent "disadvantage" index d ──┬──> vehicle_age
                                                    └──> prior_claims_3y (hidden)
    vehicle_age, prior_claims_3y, claim_amount, policy_tenure_years
        ──> assigned_queue ──> final_outcome

zip_cluster is NEVER used directly in the queue or outcome logic. Any disparity
in assigned_queue / final_outcome across zip clusters flows entirely through
vehicle_age and prior claim frequency, i.e. those features act as proxies.

1. Six zip clusters (Z01..Z06) are drawn with sampling weights
   [0.22, 0.20, 0.18, 0.16, 0.14, 0.10]. Each cluster gets a fixed latent
   disadvantage index d (CLUSTER_PROFILE below):
       Z01=0.05, Z02=0.15, Z03=0.35, Z04=0.55, Z05=0.75, Z06=0.90
   Each claim's d is jittered with N(0, 0.08) and clipped to [0, 1], so the
   cluster signal is strong on average but individual rows overlap.

2. vehicle_age (years, integer 0..25):
       vehicle_age ~ Gamma(shape=3, scale=(3 + 7*d)/3), rounded, clipped
   => mean age is ~3 years at d=0 and ~10 years at d=1.
   Observed means (seed 42): Z01 ~3.6, Z03 ~5.3, Z06 ~9.3.

3. prior_claims_3y (count of prior claims in last 3 years, hidden by default):
       prior_claims_3y ~ Poisson(lambda = 0.25 + 1.35*d)
   => mean ~0.3 at Z01 rising to ~1.5 at Z06 (seed 42: 0.34 -> 1.55).
   Given d, vehicle_age and prior_claims_3y are drawn independently, so their
   correlation exists only between clusters (within-cluster r ~ 0).
   Not emitted unless --expose-prior-claims is passed, so by default it is an
   unobserved confounder you would have to infer (e.g. via queue residuals).

4. policy_tenure_years is drawn independently of zip_cluster:
       Gamma(shape=2, scale=3), clipped to [0.1, 30], 1 decimal place.
   It is a "clean" control variable with no cluster link.

5. claim_amount (USD) is lognormal, independent of zip_cluster except
   through vehicle_age (older vehicles -> slightly lower repair cost):
       log(amount) ~ N(8.3 - 0.02*vehicle_age, 0.75), clipped to [250, 75000]

6. assigned_queue is chosen via a multinomial-logit score that uses ONLY
   vehicle_age, prior_claims_3y, claim_amount and tenure:
       s_siu  = -3.2 + 0.11*vehicle_age + 0.70*prior_claims_3y
                + 0.45*log10(amount/1000) - 0.05*tenure
       s_fast = 0.8 - 0.09*vehicle_age - 0.55*prior_claims_3y
                - 0.90*log10(amount/1000) + 0.04*tenure
       s_std  = 0
   Queues: fast_track / standard / special_investigation.

7. final_outcome (approved / partial / denied) depends on queue plus the same
   proxy features (never zip_cluster):
       base probs by queue:
           fast_track: approved .88, partial .09, denied .03
           standard:   approved .70, partial .20, denied .10
           special_investigation: approved .40, partial .28, denied .32
       then logit shift on "denied" of +0.04*vehicle_age + 0.25*prior_claims_3y,
       renormalised.

What you should be able to detect:
  - Strong association zip_cluster <-> vehicle_age (and prior_claims if exposed).
  - SIU routing and denial rates rising monotonically Z01 -> Z06
    (seed 42: SIU 5.0% -> 23.7%, denied 9.3% -> 26.2%).
  - Conditional on vehicle_age and prior_claims_3y, zip_cluster carries ~no
    additional predictive power for queue/outcome (the pathway is fully
    mediated). If you only condition on vehicle_age, a residual cluster effect
    remains because prior_claims_3y is still hidden.
-------------------------------------------------------------------------------
"""

import argparse

import numpy as np
import pandas as pd

N_ROWS = 5000
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

# zip_cluster -> (sampling weight, latent disadvantage index d)
CLUSTER_PROFILE = {
    "Z01": (0.22, 0.05),
    "Z02": (0.20, 0.15),
    "Z03": (0.18, 0.35),
    "Z04": (0.16, 0.55),
    "Z05": (0.14, 0.75),
    "Z06": (0.10, 0.90),
}

QUEUES = np.array(["fast_track", "standard", "special_investigation"])
OUTCOMES = np.array(["approved", "partial", "denied"])
OUTCOME_BASE = {
    "fast_track": [0.88, 0.09, 0.03],
    "standard": [0.70, 0.20, 0.10],
    "special_investigation": [0.40, 0.28, 0.32],
}


def softmax(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sample_rows(probs, rng):
    """Sample one column index per row from a row-stochastic matrix."""
    u = rng.random((probs.shape[0], 1))
    return (probs.cumsum(axis=1) < u).sum(axis=1)


def generate(n=N_ROWS, seed=42, expose_prior_claims=False):
    rng = np.random.default_rng(seed)

    clusters = np.array(list(CLUSTER_PROFILE))
    weights = np.array([w for w, _ in CLUSTER_PROFILE.values()])
    d_by_cluster = np.array([d for _, d in CLUSTER_PROFILE.values()])

    cluster_idx = rng.choice(len(clusters), size=n, p=weights)
    zip_cluster = clusters[cluster_idx]
    d = np.clip(d_by_cluster[cluster_idx] + rng.normal(0, 0.08, n), 0, 1)

    # Proxy features driven by d (see docstring steps 2-3).
    vehicle_age = np.clip(np.round(rng.gamma(3, (3 + 7 * d) / 3)), 0, 25).astype(int)
    prior_claims = rng.poisson(0.25 + 1.35 * d)

    # Cluster-independent features (steps 4-5).
    tenure = np.round(np.clip(rng.gamma(2, 3, n), 0.1, 30), 1)
    amount = np.round(
        np.clip(np.exp(rng.normal(8.3 - 0.02 * vehicle_age, 0.75)), 250, 75_000), 2
    )

    # Queue assignment (step 6) - no zip_cluster term.
    log_amt = np.log10(amount / 1000)
    s_fast = 0.8 - 0.09 * vehicle_age - 0.55 * prior_claims - 0.90 * log_amt + 0.04 * tenure
    s_std = np.zeros(n)
    s_siu = -3.2 + 0.11 * vehicle_age + 0.70 * prior_claims + 0.45 * log_amt - 0.05 * tenure
    queue = QUEUES[sample_rows(softmax(np.column_stack([s_fast, s_std, s_siu])), rng)]

    # Outcome (step 7) - no zip_cluster term.
    base = np.array([OUTCOME_BASE[q] for q in queue])
    logits = np.log(base)
    logits[:, 2] += 0.04 * vehicle_age + 0.25 * prior_claims
    outcome = OUTCOMES[sample_rows(softmax(logits), rng)]

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:06d}" for i in range(1, n + 1)],
            "claim_month": rng.choice(MONTHS, size=n),
            "claim_amount": amount,
            "zip_cluster": zip_cluster,
            "vehicle_age": vehicle_age,
            "policy_tenure_years": tenure,
            "assigned_queue": queue,
            "final_outcome": outcome,
        }
    )
    if expose_prior_claims:
        df["prior_claims_3y"] = prior_claims
    return df.sort_values(["claim_month", "claim_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="claims.csv")
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expose-prior-claims", action="store_true")
    args = parser.parse_args()

    df = generate(args.rows, args.seed, args.expose_prior_claims)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
