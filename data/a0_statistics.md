# P20 Statistical Rigor (A0) — Stability Analysis

_Generated offline from `votes_census_percase.csv` (no LLM/API calls). Seed=20260618; bootstrap resamples=10,000; permutations=10,000; Monte-Carlo draws/cell=4,000._

**Label-free caveat.** There is NO ground truth in this corpus, so every quantity below measures **stability/self-consistency**, not accuracy. `verdict_flip=1` means a (model,caseId)'s repetitions were not unanimous; `caught_flip` is the subset of flips the harness flagged. Verdict→block mapping for the aggregation sweep: `consistent`=ALLOW, `drifting`/`hijacked`=BLOCK.

**Unit of analysis.** Repetitions and effort-rows within a caseId are not independent. We report Wilson CIs (naive, row-level Bernoulli — the conventional headline) AND cluster-bootstrap CIs that resample whole caseId clusters; the **cluster-bootstrap CI is the one to quote**. The matched base deck has **12 caseId clusters** (['adv-1', 'adv-10', 'adv-11', 'adv-12', 'adv-2', 'adv-3', 'adv-4', 'adv-5', 'adv-6', 'adv-7', 'adv-8', 'adv-9']).

## Task 1 — Confidence intervals on headline rates

| Rate | Point | Wilson 95% CI | Cluster-bootstrap 95% CI | n rows | #clusters |
|---|---|---|---|---|---|
| Corpus verdict-flip (all 1437) | 12.1% | [10.5, 13.9] | [7.0, 17.3] | 1437 | 73 |
| Corpus caught-flip (all 1437) | 7.3% | [6.1, 8.8] | [3.6, 11.1] | 1437 | 73 |
| Matched-deck verdict-flip: Opus 4.7 | 29.5% | [22.4, 37.8] | [15.2, 47.0] | 132 | 12 |
| Matched-deck verdict-flip: Sonnet 4.6 | 42.7% | [33.3, 52.7] | [21.9, 63.5] | 96 | 12 |
| Matched-deck verdict-flip: Haiku 4.5 | 55.2% | [45.3, 64.8] | [37.5, 71.9] | 96 | 12 |
| Matched-deck caught-flip: Opus 4.7 | 16.7% | [11.3, 23.9] | [3.8, 34.8] | 132 | 12 |
| Matched-deck caught-flip: Sonnet 4.6 | 31.2% | [22.9, 41.1] | [14.6, 49.0] | 96 | 12 |
| Matched-deck caught-flip: Haiku 4.5 | 35.4% | [26.6, 45.4] | [19.8, 52.1] | 96 | 12 |
| Matched-deck flip @ T=0.1 (Sonnet+Haiku) | 12.5% | [5.9, 24.7] | [0.0, 25.0] | 48 | 12 |
| Matched-deck flip @ T=1.0 (Sonnet+Haiku) | 61.1% | [53.0, 68.7] | [41.0, 80.6] | 144 | 12 |
| Verdict-flip @ rep-budget=2 | 0.0% | [0.0, 35.4] | [0.0, 0.0] | 7 | 5 |
| Verdict-flip @ rep-budget=5 | 2.6% | [1.8, 3.9] | [1.1, 4.6] | 946 | 73 |
| Verdict-flip @ rep-budget=20 | 31.5% | [27.4, 35.9] | [21.9, 41.1] | 454 | 12 |
| Verdict-flip @ rep-budget=40 | 25.0% | [12.0, 44.9] | [8.3, 41.7] | 24 | 12 |

Notes: corpus-wide rows have many clusters (broad deck); the matched-deck and temperature rows all share the SAME 12 caseId clusters, so their cluster-bootstrap CIs are wide. The by-rep-budget rows are confounded by *which* cases fall in each bucket (rep=5 is mostly the easy B3 deck; rep=20/40 is the adversarial deck) — they are NOT a clean rep-budget manipulation and should be read as descriptive only.

## Task 2 — Model ordering (Opus < Sonnet < Haiku), matched deck, n=12 clusters

Point flip rates: Opus 4.7 29.5%, Sonnet 4.6 42.7%, Haiku 4.5 55.2%.

| Pairwise difference | Point | Cluster-boot 95% CI | Permutation p (2-sided) | CI excludes 0? |
|---|---|---|---|---|
| Haiku − Opus | 25.7% | [0.1, 47.3] | 0.0001 | YES |
| Sonnet − Opus | 13.2% | [-3.6, 30.9] | 0.0149 | **no** |
| Haiku − Sonnet | 12.5% | [-10.4, 35.4] | 0.0668 | **no** |

- Fraction of bootstrap resamples preserving the **full** ordering Opus<Sonnet<Haiku: **0.778** (7,777/10,000).
- Permutation test for between-model spread larger than chance: observed spread = 25.7%, **p = 0.0002**.

## Task 3 — Temperature effect (T=0.1 vs T=1.0), Sonnet+Haiku matched deck

- Flip @ T=0.1 = 12.5% (n=48); flip @ T=1.0 = 61.1% (n=144).
- Difference (T=1.0 − T=0.1) = **48.6%**, cluster-bootstrap 95% CI [25.7, 69.4] (over 12 caseIds).
- Permutation test (T labels permuted within caseId): **p = 0.0001**.

## Task 4 — Offline aggregation-rule sweep (Monte-Carlo over empirical votes)

Each rule's instability is the per-case probability that its own aggregate decision disagrees with its own modal decision, averaged over the 12 cases. Lower = more stable. `cost` = panel size / #draws.

| Rule | cost | verdict-instability | block-instability |
|---|---|---|---|
| single: Opus 4.7 | 1 | 14.0% | 5.1% |
| single: Sonnet 4.6 | 1 | 20.9% | 13.3% |
| single: Haiku 4.5 | 1 | 18.5% | 11.8% |
| self-ens k=3: Opus 4.7 | 3 | 12.0% | 3.7% |
| self-ens k=5: Opus 4.7 | 5 | 11.3% | 3.1% |
| self-ens k=3: Sonnet 4.6 | 3 | 17.8% | 11.3% |
| self-ens k=5: Sonnet 4.6 | 5 | 15.5% | 10.1% |
| self-ens k=3: Haiku 4.5 | 3 | 13.7% | 7.6% |
| self-ens k=5: Haiku 4.5 | 5 | 12.1% | 6.0% |
| diverse 3-model majority | 3 | 21.8% | 13.7% |
| diverse 3-model fail-closed (block if any) | 3 | 21.5% | 2.6% |
| 3-model k-of-n: block if >=1/3 | 3 | 21.9% | 2.6% |
| 3-model k-of-n: block if >=2/3 | 3 | 21.7% | 13.3% |
| 3-model k-of-n: block if >=3/3 | 3 | 21.8% | 8.7% |

### Pareto frontier (block-instability vs cost)
- cost 1: **single: Opus 4.7** — block-instability 5.1%
- cost 3: **3-model k-of-n: block if >=1/3** — block-instability 2.6%

### Bootstrap CIs on headline reductions (block-instability, resampled over caseIds)

| Comparison | Reduction (pp) | 95% CI (pp) |
|---|---|---|
| Opus 4.7: single->self-ens(k=5), block-instability reduction | +1.9 | [+0.2, +4.4] |
| Sonnet 4.6: single->self-ens(k=5), block-instability reduction | +3.1 | [+0.5, +6.0] |
| Haiku 4.5: single->self-ens(k=5), block-instability reduction | +5.8 | [+3.1, +8.9] |
| single Opus -> diverse 3-model majority, block-instability reduction | -8.7 | [-19.4, -0.3] |

## Task 5 — Inter-model disagreement structure (label-free)

Pairwise agreement on **modal verdict** across the 12 matched-deck caseIds:

| | Opus 4.7 | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|---|
| Opus 4.7 | 100.0% | 66.7% | 41.7% |
| Sonnet 4.6 | 66.7% | 100.0% | 50.0% |
| Haiku 4.5 | 41.7% | 50.0% | 100.0% |

**The error-decorrelation Q-statistic CANNOT be computed here**: Yule's Q measures correlation of *errors*, which requires ground-truth labels to define an error. This corpus has none. Computing Q would require fabricating labels, which we do not do — it needs the live labelled pilot.

## What changed / what holds

- The headline corpus rates (verdict-flip 12.1%, caught-flip 7.3%) are tightly estimated and should now be reported WITH their cluster-bootstrap CIs ([7.0, 17.3] and [3.6, 11.1]).
- The Opus<Sonnet<Haiku ordering is only PARTIALLY supported at n=12 clusters: the full order is preserved in 78% of bootstrap resamples. Pairwise CIs that include 0 (NOT significant): Sonnet−Opus, Haiku−Sonnet. Report the ordering as a trend with CIs, not as three clean significant gaps.
- The temperature effect is large and robust: +48.6% flip from T=0.1 to T=1.0, CI [25.7, 69.4], permutation p=0.0001. This is the strongest single effect and should be reported with its CI.
- Aggregation helps stability: self-ensembling and diverse panels reduce block-decision instability, but the reduction CIs (above) should be quoted because per-case variance is high with only 12 cases.

_All CIs above are stability CIs. Accuracy / error-decorrelation claims remain ungrounded until the labelled live pilot is run._
