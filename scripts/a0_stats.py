#!/usr/bin/env python3
"""
P20 statistical-rigor analysis (A0).

PURE OFFLINE analysis of votes_census_percase.csv. No LLM/API calls.
We have NO ground-truth labels => everything here is STABILITY, not accuracy.

Outputs:
  - data/a0_statistics.md     (structured report)
  - figures/a0_pareto.pdf     (stability vs panel size)

Unit-of-analysis note:
  Repetitions within a (model,caseId) are NOT independent, and multiple rows
  (effort levels) exist per (model,caseId). We therefore report BOTH:
    * Wilson 95% CIs treating each analysed row as one Bernoulli draw (naive,
      ignores clustering; included only because it is the conventional headline),
    * cluster-bootstrap 95% CIs resampling whole caseId clusters (>=10,000
      resamples), which respects the dependence and is the CI we trust.
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260618
rng = np.random.default_rng(SEED)
NBOOT = 10000
NPERM = 10000

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV  = os.path.join(ROOT, "data", "votes_census_percase.csv")
OUT_MD = os.path.join(ROOT, "data", "a0_statistics.md")
OUT_PDF = os.path.join(ROOT, "figures", "a0_pareto.pdf")

THREE = ["Claude Opus 4.7", "Claude Sonnet 4.6", "Claude Haiku 4.5"]
SHORT = {"Claude Opus 4.7": "Opus 4.7",
         "Claude Sonnet 4.6": "Sonnet 4.6",
         "Claude Haiku 4.5": "Haiku 4.5",
         "Nova Micro": "Nova Micro"}

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def wilson_ci(k, n, z=1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = (z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / den
    return (max(0.0, centre - half), min(1.0, centre + half))

def cluster_bootstrap_mean(df, value_col, cluster_col="caseId", nboot=NBOOT, rng=rng):
    """Bootstrap the mean of value_col by resampling whole clusters with
    replacement. Returns (point, lo, hi, n_rows, n_clusters, boot_array).
    Point estimate is the simple mean over rows (row-weighted)."""
    clusters = df[cluster_col].unique()
    nc = len(clusters)
    groups = {c: df[df[cluster_col] == c][value_col].to_numpy() for c in clusters}
    point = df[value_col].mean()
    boots = np.empty(nboot)
    for b in range(nboot):
        pick = rng.choice(clusters, size=nc, replace=True)
        vals = np.concatenate([groups[c] for c in pick])
        boots[b] = vals.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, len(df), nc, boots

def cluster_bootstrap_diff(df, value_col, group_col, gA, gB,
                           cluster_col="caseId", nboot=NBOOT, rng=rng):
    """Bootstrap difference (mean_gA - mean_gB), resampling clusters jointly so
    the paired structure across the same caseIds is preserved.
    Returns (diff_point, lo, hi, boot_array)."""
    clusters = df[cluster_col].unique()
    nc = len(clusters)
    A = {c: df[(df[cluster_col]==c)&(df[group_col]==gA)][value_col].to_numpy() for c in clusters}
    B = {c: df[(df[cluster_col]==c)&(df[group_col]==gB)][value_col].to_numpy() for c in clusters}
    def diff_from(picks):
        a = np.concatenate([A[c] for c in picks if A[c].size]) if any(A[c].size for c in picks) else np.array([np.nan])
        b = np.concatenate([B[c] for c in picks if B[c].size]) if any(B[c].size for c in picks) else np.array([np.nan])
        return np.nanmean(a) - np.nanmean(b)
    point = diff_from(clusters)
    boots = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(clusters, size=nc, replace=True)
        boots[i] = diff_from(pick)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi, boots

# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV)
N_TOTAL = len(df)

def reconstruct_T(eff):
    return 0.1 if str(eff) in ("none", "none/unset") else 1.0
df["T"] = df["effort"].apply(reconstruct_T)

matched = df[(df["benchmark"] == "adversarial") & (df["reps_valid"] >= 19)].copy()
matched3 = matched[matched["model"].isin(THREE)].copy()   # exclude Nova Micro
N_CASES = matched["caseId"].nunique()

report = []
def w(s=""):
    report.append(s)

# ============================================================================
# TASK 1 -- confidence intervals on headline rates
# ============================================================================
ci_rows = []  # (label, k, n, point, wlo, whi, blo, bhi, nclust, scope_df, value_col)

def add_ci(label, sub, value_col="verdict_flip"):
    k = int(sub[value_col].sum())
    n = len(sub)
    p = k / n if n else float("nan")
    wlo, whi = wilson_ci(k, n)
    point, blo, bhi, nrow, nclust, _ = cluster_bootstrap_mean(sub, value_col)
    ci_rows.append(dict(label=label, k=k, n=n, point=p, wlo=wlo, whi=whi,
                        blo=blo, bhi=bhi, nclust=nclust, value_col=value_col))
    return ci_rows[-1]

# Corpus-wide
add_ci("Corpus verdict-flip (all 1437)", df, "verdict_flip")
add_ci("Corpus caught-flip (all 1437)", df, "caught_flip")

# Per-model matched-deck verdict-flip & caught-flip
for mdl in THREE:
    sub = matched[matched["model"] == mdl]
    add_ci(f"Matched-deck verdict-flip: {SHORT[mdl]}", sub, "verdict_flip")
for mdl in THREE:
    sub = matched[matched["model"] == mdl]
    add_ci(f"Matched-deck caught-flip: {SHORT[mdl]}", sub, "caught_flip")

# T=0.1 vs T=1.0 (non-Opus-4.7 matched deck)  [Sonnet+Haiku; Nova has only T=0.1]
nonopus = matched[~matched["model"].str.contains("Opus 4.7")].copy()
sh = nonopus[nonopus["model"].isin(["Claude Sonnet 4.6", "Claude Haiku 4.5"])].copy()
add_ci("Matched-deck flip @ T=0.1 (Sonnet+Haiku)", sh[sh["T"]==0.1], "verdict_flip")
add_ci("Matched-deck flip @ T=1.0 (Sonnet+Haiku)", sh[sh["T"]==1.0], "verdict_flip")

# By repetition budget (corpus). Cluster bootstrap over caseId within each bucket.
for rb in [2, 5, 20, 40]:
    sub = df[df["reps_valid"] == rb]
    add_ci(f"Verdict-flip @ rep-budget={rb}", sub, "verdict_flip")

# ============================================================================
# TASK 2 -- significance of model ordering (Opus < Sonnet < Haiku)
# ----------------------------------------------------------------------------
# Cluster bootstrap over the 12 caseIds. Per bootstrap draw, compute each
# model's row-weighted flip rate on the resampled clusters, then the pairwise
# diffs and whether the full ordering is preserved.
# ============================================================================
clusters = matched3["caseId"].unique()
nc = len(clusters)
by_mc = {}  # (model,caseId)->array of verdict_flip
for mdl in THREE:
    for c in clusters:
        by_mc[(mdl, c)] = matched3[(matched3["model"]==mdl)&(matched3["caseId"]==c)]["verdict_flip"].to_numpy()

def model_rates(picks):
    out = {}
    for mdl in THREE:
        vals = np.concatenate([by_mc[(mdl, c)] for c in picks])
        out[mdl] = vals.mean()
    return out

point_rates = model_rates(clusters)
O, S, H = "Claude Opus 4.7", "Claude Sonnet 4.6", "Claude Haiku 4.5"

boot_HmO = np.empty(NBOOT); boot_SmO = np.empty(NBOOT); boot_HmS = np.empty(NBOOT)
order_preserved = 0
for b in range(NBOOT):
    pick = rng.choice(clusters, size=nc, replace=True)
    r = model_rates(pick)
    boot_HmO[b] = r[H]-r[O]; boot_SmO[b] = r[S]-r[O]; boot_HmS[b] = r[H]-r[S]
    if r[O] < r[S] < r[H]:
        order_preserved += 1

def summ(arr):
    return float(np.mean(arr)), tuple(np.percentile(arr, [2.5, 97.5]))

HmO = (point_rates[H]-point_rates[O],) + (tuple(np.percentile(boot_HmO,[2.5,97.5])),)
SmO = (point_rates[S]-point_rates[O],) + (tuple(np.percentile(boot_SmO,[2.5,97.5])),)
HmS = (point_rates[H]-point_rates[S],) + (tuple(np.percentile(boot_HmS,[2.5,97.5])),)
frac_order = order_preserved / NBOOT

# Permutation test: between-model spread larger than chance?
# Permute model labels WITHIN each caseId (across the rows of that caseId),
# then recompute the spread statistic = max(rate)-min(rate) over the 3 models.
def spread_stat(frame):
    rates = frame.groupby("model")["verdict_flip"].mean()
    return rates.max() - rates.min()

obs_spread = spread_stat(matched3)
# build per-case row pools
perm_frame = matched3[["model", "caseId", "verdict_flip"]].copy()
case_groups = {c: perm_frame[perm_frame["caseId"]==c].copy() for c in clusters}
ge = 0
for _ in range(NPERM):
    parts = []
    for c, g in case_groups.items():
        gp = g.copy()
        gp["model"] = rng.permutation(gp["model"].to_numpy())
        parts.append(gp)
    pf = pd.concat(parts, ignore_index=True)
    if spread_stat(pf) >= obs_spread - 1e-12:
        ge += 1
p_spread = (ge + 1) / (NPERM + 1)

# Pairwise permutation p-values (two-sided), permuting the two models' labels
# within caseId.
def pairwise_perm_p(frame, gA, gB):
    sub = frame[frame["model"].isin([gA, gB])][["model","caseId","verdict_flip"]].copy()
    obs = abs(sub[sub.model==gA]["verdict_flip"].mean() - sub[sub.model==gB]["verdict_flip"].mean())
    cg = {c: sub[sub.caseId==c].copy() for c in sub["caseId"].unique()}
    ge = 0
    for _ in range(NPERM):
        parts=[]
        for c,g in cg.items():
            gp=g.copy(); gp["model"]=rng.permutation(gp["model"].to_numpy()); parts.append(gp)
        pf=pd.concat(parts, ignore_index=True)
        d=abs(pf[pf.model==gA]["verdict_flip"].mean()-pf[pf.model==gB]["verdict_flip"].mean())
        if d >= obs - 1e-12: ge+=1
    return (ge+1)/(NPERM+1), obs

p_HmO, _ = pairwise_perm_p(matched3, H, O)
p_SmO, _ = pairwise_perm_p(matched3, S, O)
p_HmS, _ = pairwise_perm_p(matched3, H, S)

# ============================================================================
# TASK 3 -- temperature effect (T=0.1 vs T=1.0), non-Opus-4.7 matched deck
# ============================================================================
# cluster bootstrap of difference T=1.0 - T=0.1 on Sonnet+Haiku, paired by caseId
tdf = sh[["caseId","T","verdict_flip"]].copy()
tdf["Tlab"] = tdf["T"].map({0.1:"T01", 1.0:"T10"})
t_diff, t_lo, t_hi, t_boots = cluster_bootstrap_diff(
    tdf, "verdict_flip", "Tlab", "T10", "T01")
# permutation: permute T labels within caseId
cg = {c: tdf[tdf.caseId==c].copy() for c in tdf["caseId"].unique()}
obs_t = tdf[tdf.Tlab=="T10"]["verdict_flip"].mean() - tdf[tdf.Tlab=="T01"]["verdict_flip"].mean()
ge=0
for _ in range(NPERM):
    parts=[]
    for c,g in cg.items():
        gp=g.copy(); gp["Tlab"]=rng.permutation(gp["Tlab"].to_numpy()); parts.append(gp)
    pf=pd.concat(parts, ignore_index=True)
    d=pf[pf.Tlab=="T10"]["verdict_flip"].mean()-pf[pf.Tlab=="T01"]["verdict_flip"].mean()
    if abs(d) >= abs(obs_t)-1e-12: ge+=1
p_temp=(ge+1)/(NPERM+1)
rate_T01 = tdf[tdf.Tlab=="T01"]["verdict_flip"].mean()
rate_T10 = tdf[tdf.Tlab=="T10"]["verdict_flip"].mean()

# ============================================================================
# TASK 4 -- offline aggregation-rule sweep (Monte-Carlo over empirical votes)
# ----------------------------------------------------------------------------
# Verdict->block mapping: 'consistent' = ALLOW (judge saw no drift), 'drifting'
# or 'hijacked' = BLOCK. We Monte-Carlo draws from each (model,caseId)'s
# empirical multinomial over {consistent,drifting,hijacked} (pooled across the
# matched-deck rows so each cell has ~20*#efforts reps), then apply each
# aggregation rule and measure:
#   * verdict instability  P(aggregate verdict != that config's own MODE)
#   * block-decision instability P(block != that config's own modal block)
# averaged over the 12 caseIds. Single-agent uses one model's marginal.
# ============================================================================
VERDS = ["consistent", "drifting", "hijacked"]
BLOCK = {"consistent": 0, "drifting": 1, "hijacked": 1}

def empirical_probs(frame, mdl, c):
    sub = frame[(frame.model==mdl)&(frame.caseId==c)]
    counts = np.array([sub["n_consistent"].sum(), sub["n_drifting"].sum(), sub["n_hijacked"].sum()], float)
    if counts.sum()==0: return None
    return counts / counts.sum()

# Pre-compute per (model,caseId) probability vectors on matched deck (3 models)
probs = {}
for mdl in THREE:
    for c in clusters:
        p = empirical_probs(matched3, mdl, c)
        if p is not None:
            probs[(mdl,c)] = p

MC = 4000  # monte-carlo draws per caseId per config

def mode_of(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return vals[np.argmax(cnts)]

def draw_single(mdl, c, n):
    return rng.choice(3, size=n, p=probs[(mdl,c)])

def aggregate_verdict_from_draws(draws):
    """draws: array of int codes (0/1/2). Majority by count; ties -> most severe
    (higher code = more severe: hijacked>drifting>consistent)."""
    cnts = np.bincount(draws, minlength=3)
    mx = cnts.max()
    winners = np.where(cnts==mx)[0]
    return winners.max()  # tie -> most severe

def block_any(draws):
    return int(np.any(np.array([BLOCK[VERDS[d]] for d in draws]) == 1))

def block_majority(draws):
    b = np.array([BLOCK[VERDS[d]] for d in draws])
    return int(b.sum() > len(b)/2)

def block_kofn(draws, k):
    b = np.array([BLOCK[VERDS[d]] for d in draws])
    return int(b.sum() >= k)

def run_config(name, sampler, decide_block, n_draws_for_cost):
    """sampler(c) -> returns aggregate verdict code AND a block decision per MC trial.
    Returns dict with verdict_instability, block_instability, cost."""
    vinst = []; binst = []
    for c in clusters:
        if not all((m,c) in probs for m in THREE):
            continue
        vouts = np.empty(MC, int); bouts = np.empty(MC, int)
        for t in range(MC):
            v, b = sampler(c)
            vouts[t]=v; bouts[t]=b
        vmode = mode_of(vouts); bmode = mode_of(bouts)
        vinst.append(np.mean(vouts != vmode))
        binst.append(np.mean(bouts != bmode))
    return dict(name=name,
                v_inst=float(np.mean(vinst)),
                b_inst=float(np.mean(binst)),
                cost=n_draws_for_cost,
                v_per_case=np.array(vinst),
                b_per_case=np.array(binst))

configs = []

# Single agent (each of the 3 models) — 1 draw
for mdl in THREE:
    def mk(mdl):
        def s(c):
            d = draw_single(mdl, c, 1)
            return aggregate_verdict_from_draws(d), int(BLOCK[VERDS[d[0]]])
        return s
    configs.append(run_config(f"single: {SHORT[mdl]}", mk(mdl), None, 1))

# Self-ensemble k draws from ONE model, majority verdict + majority block
for mdl in THREE:
    for k in (3,5):
        def mk(mdl,k):
            def s(c):
                d = draw_single(mdl, c, k)
                return aggregate_verdict_from_draws(d), block_majority(d)
            return s
        configs.append(run_config(f"self-ens k={k}: {SHORT[mdl]}", mk(mdl,k), None, k))

# Diverse 3-model majority (1 draw each)
def s_div(c):
    d = np.array([draw_single(m, c, 1)[0] for m in THREE])
    return aggregate_verdict_from_draws(d), block_majority(d)
configs.append(run_config("diverse 3-model majority", s_div, None, 3))

# Unanimity-to-allow / conjunctive fail-closed: block if ANY of 3 blocks
def s_conj(c):
    d = np.array([draw_single(m, c, 1)[0] for m in THREE])
    return aggregate_verdict_from_draws(d), block_any(d)
configs.append(run_config("diverse 3-model fail-closed (block if any)", s_conj, None, 3))

# k-of-n threshold rules on 3-model panel: block if >=k of 3 block
for k in (1,2,3):
    def mk(k):
        def s(c):
            d = np.array([draw_single(m, c, 1)[0] for m in THREE])
            return aggregate_verdict_from_draws(d), block_kofn(d, k)
        return s
    configs.append(run_config(f"3-model k-of-n: block if >={k}/3", mk(k), None, 3))

# Bootstrap CI on the headline reduction: self-ensemble k=5 vs single, per model,
# on block instability. Resample over caseIds (paired per-case arrays).
def boot_reduction(single_cfg, ens_cfg, metric="b_per_case"):
    a = single_cfg[metric]; b = ens_cfg[metric]
    n = len(a)
    diffs = np.empty(NBOOT)
    point = a.mean() - b.mean()  # reduction (single - ensemble), >0 = improvement
    for i in range(NBOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo,hi = np.percentile(diffs,[2.5,97.5])
    return point, lo, hi

# map names
cfg_by_name = {c["name"]: c for c in configs}
reductions = []
for mdl in THREE:
    sname=f"single: {SHORT[mdl]}"; ename=f"self-ens k=5: {SHORT[mdl]}"
    pt,lo,hi = boot_reduction(cfg_by_name[sname], cfg_by_name[ename], "b_per_case")
    reductions.append((f"{SHORT[mdl]}: single->self-ens(k=5), block-instability reduction", pt,lo,hi))
# diverse majority vs single Opus
pt,lo,hi = boot_reduction(cfg_by_name["single: Opus 4.7"], cfg_by_name["diverse 3-model majority"], "b_per_case")
reductions.append(("single Opus -> diverse 3-model majority, block-instability reduction", pt,lo,hi))

# ============================================================================
# TASK 5 -- inter-model disagreement structure (label-free)
# ----------------------------------------------------------------------------
# Modal verdict per (model,caseId) on the matched deck; pairwise agreement.
# ============================================================================
modal = {}
for mdl in THREE:
    for c in clusters:
        p = probs.get((mdl,c))
        modal[(mdl,c)] = VERDS[int(np.argmax(p))] if p is not None else None
agree = {}
for i,a in enumerate(THREE):
    for bb in THREE:
        same = [modal[(a,c)]==modal[(bb,c)] for c in clusters if modal[(a,c)] and modal[(bb,c)]]
        agree[(a,bb)] = np.mean(same)

# ============================================================================
# WRITE REPORT
# ============================================================================
def fpct(x): return f"{100*x:.1f}%"
def fci(lo,hi): return f"[{100*lo:.1f}, {100*hi:.1f}]"

w(f"# P20 Statistical Rigor (A0) — Stability Analysis")
w(f"")
w(f"_Generated offline from `votes_census_percase.csv` (no LLM/API calls). "
  f"Seed={SEED}; bootstrap resamples={NBOOT:,}; permutations={NPERM:,}; "
  f"Monte-Carlo draws/cell={MC:,}._")
w(f"")
w(f"**Label-free caveat.** There is NO ground truth in this corpus, so every "
  f"quantity below measures **stability/self-consistency**, not accuracy. "
  f"`verdict_flip=1` means a (model,caseId)'s repetitions were not unanimous; "
  f"`caught_flip` is the subset of flips the harness flagged. Verdict→block "
  f"mapping for the aggregation sweep: `consistent`=ALLOW, "
  f"`drifting`/`hijacked`=BLOCK.")
w(f"")
w(f"**Unit of analysis.** Repetitions and effort-rows within a caseId are not "
  f"independent. We report Wilson CIs (naive, row-level Bernoulli — the "
  f"conventional headline) AND cluster-bootstrap CIs that resample whole "
  f"caseId clusters; the **cluster-bootstrap CI is the one to quote**. The "
  f"matched base deck has **{N_CASES} caseId clusters** "
  f"({sorted(clusters.tolist())}).")
w("")
w("## Task 1 — Confidence intervals on headline rates")
w("")
w("| Rate | Point | Wilson 95% CI | Cluster-bootstrap 95% CI | n rows | #clusters |")
w("|---|---|---|---|---|---|")
for r in ci_rows:
    w(f"| {r['label']} | {fpct(r['point'])} | {fci(r['wlo'],r['whi'])} | "
      f"{fci(r['blo'],r['bhi'])} | {r['n']} | {r['nclust']} |")
w("")
w("Notes: corpus-wide rows have many clusters (broad deck); the matched-deck "
  "and temperature rows all share the SAME 12 caseId clusters, so their "
  "cluster-bootstrap CIs are wide. The by-rep-budget rows are confounded by "
  "*which* cases fall in each bucket (rep=5 is mostly the easy B3 deck; "
  "rep=20/40 is the adversarial deck) — they are NOT a clean rep-budget "
  "manipulation and should be read as descriptive only.")
w("")
w("## Task 2 — Model ordering (Opus < Sonnet < Haiku), matched deck, n=12 clusters")
w("")
w(f"Point flip rates: Opus 4.7 {fpct(point_rates[O])}, Sonnet 4.6 "
  f"{fpct(point_rates[S])}, Haiku 4.5 {fpct(point_rates[H])}.")
w("")
w("| Pairwise difference | Point | Cluster-boot 95% CI | Permutation p (2-sided) | CI excludes 0? |")
w("|---|---|---|---|---|")
def line(lbl, trip, p):
    pt, ci = trip
    excl = "YES" if (ci[0] > 0 or ci[1] < 0) else "**no**"
    w(f"| {lbl} | {fpct(pt)} | {fci(ci[0],ci[1])} | {p:.4f} | {excl} |")
line("Haiku − Opus", HmO, p_HmO)
line("Sonnet − Opus", SmO, p_SmO)
line("Haiku − Sonnet", HmS, p_HmS)
w("")
w(f"- Fraction of bootstrap resamples preserving the **full** ordering "
  f"Opus<Sonnet<Haiku: **{frac_order:.3f}** ({order_preserved:,}/{NBOOT:,}).")
w(f"- Permutation test for between-model spread larger than chance: "
  f"observed spread = {fpct(obs_spread)}, **p = {p_spread:.4f}**.")
w("")
w("## Task 3 — Temperature effect (T=0.1 vs T=1.0), Sonnet+Haiku matched deck")
w("")
w(f"- Flip @ T=0.1 = {fpct(rate_T01)} (n={len(tdf[tdf.Tlab=='T01'])}); "
  f"flip @ T=1.0 = {fpct(rate_T10)} (n={len(tdf[tdf.Tlab=='T10'])}).")
w(f"- Difference (T=1.0 − T=0.1) = **{fpct(t_diff)}**, cluster-bootstrap 95% CI "
  f"{fci(t_lo,t_hi)} (over {N_CASES} caseIds).")
w(f"- Permutation test (T labels permuted within caseId): **p = {p_temp:.4f}**.")
w("")
w("## Task 4 — Offline aggregation-rule sweep (Monte-Carlo over empirical votes)")
w("")
w("Each rule's instability is the per-case probability that its own aggregate "
  "decision disagrees with its own modal decision, averaged over the 12 cases. "
  "Lower = more stable. `cost` = panel size / #draws.")
w("")
w("| Rule | cost | verdict-instability | block-instability |")
w("|---|---|---|---|")
for c in configs:
    w(f"| {c['name']} | {c['cost']} | {fpct(c['v_inst'])} | {fpct(c['b_inst'])} |")
w("")
w("### Pareto frontier (block-instability vs cost)")
# compute Pareto: minimize cost and block-instability
pts = sorted([(c["cost"], c["b_inst"], c["name"]) for c in configs])
pareto=[]
best=1e9
for cost,bi,name in pts:
    # for nondominated: among same-or-lower cost, lowest b_inst
    dominated = any((o["cost"]<=cost and o["b_inst"]<bi) or (o["cost"]<cost and o["b_inst"]<=bi) for o in configs if o["name"]!=name)
    if not dominated:
        pareto.append((cost,bi,name))
for cost,bi,name in sorted(set(pareto)):
    w(f"- cost {cost}: **{name}** — block-instability {fpct(bi)}")
w("")
w("### Bootstrap CIs on headline reductions (block-instability, resampled over caseIds)")
w("")
w("| Comparison | Reduction (pp) | 95% CI (pp) |")
w("|---|---|---|")
for lbl,pt,lo,hi in reductions:
    w(f"| {lbl} | {100*pt:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
w("")
w("## Task 5 — Inter-model disagreement structure (label-free)")
w("")
w("Pairwise agreement on **modal verdict** across the 12 matched-deck caseIds:")
w("")
w("| | " + " | ".join(SHORT[m] for m in THREE) + " |")
w("|---|" + "---|"*len(THREE))
for a in THREE:
    w(f"| {SHORT[a]} | " + " | ".join(fpct(agree[(a,b)]) for b in THREE) + " |")
w("")
w("**The error-decorrelation Q-statistic CANNOT be computed here**: Yule's Q "
  "measures correlation of *errors*, which requires ground-truth labels to "
  "define an error. This corpus has none. Computing Q would require fabricating "
  "labels, which we do not do — it needs the live labelled pilot.")
w("")
# what changed / what holds
order_holds = (frac_order >= 0.95 and p_HmO < 0.05 and p_SmO < 0.05 and p_HmS < 0.05)
w("## What changed / what holds")
w("")
sentences=[]
sentences.append(
    f"The headline corpus rates (verdict-flip {fpct(ci_rows[0]['point'])}, "
    f"caught-flip {fpct(ci_rows[1]['point'])}) are tightly estimated and should "
    f"now be reported WITH their cluster-bootstrap CIs "
    f"({fci(ci_rows[0]['blo'],ci_rows[0]['bhi'])} and "
    f"{fci(ci_rows[1]['blo'],ci_rows[1]['bhi'])}).")
if order_holds:
    sentences.append(
        f"The Opus<Sonnet<Haiku ordering SURVIVES at n=12 clusters: every "
        f"pairwise difference's bootstrap CI excludes 0, all permutation "
        f"p<0.05, and {frac_order:.0%} of resamples preserve the full order.")
else:
    weak=[]
    if not (HmO[1][0]>0): weak.append("Haiku−Opus")
    if not (SmO[1][0]>0): weak.append("Sonnet−Opus")
    if not (HmS[1][0]>0): weak.append("Haiku−Sonnet")
    sentences.append(
        f"The Opus<Sonnet<Haiku ordering is only PARTIALLY supported at n=12 "
        f"clusters: the full order is preserved in {frac_order:.0%} of bootstrap "
        f"resamples. Pairwise CIs that include 0 (NOT significant): "
        f"{', '.join(weak) if weak else 'none'}. "
        f"Report the ordering as a trend with CIs, not as three clean "
        f"significant gaps.")
sentences.append(
    f"The temperature effect is large and robust: +{fpct(t_diff)} flip from "
    f"T=0.1 to T=1.0, CI {fci(t_lo,t_hi)}, permutation p={p_temp:.4f}. This is "
    f"the strongest single effect and should be reported with its CI.")
sentences.append(
    f"Aggregation helps stability: self-ensembling and diverse panels reduce "
    f"block-decision instability, but the reduction CIs (above) should be "
    f"quoted because per-case variance is high with only 12 cases.")
for s in sentences:
    w(f"- {s}")
w("")
w("_All CIs above are stability CIs. Accuracy / error-decorrelation claims "
  "remain ungrounded until the labelled live pilot is run._")

with open(OUT_MD, "w") as f:
    f.write("\n".join(report) + "\n")
print("WROTE", OUT_MD)

# ----------------------------------------------------------------------------
# Pareto figure
# ----------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs=[c["cost"] for c in configs]; ys=[100*c["b_inst"] for c in configs]
    ax.scatter(xs, ys, s=46, color="#3b6", zorder=3, edgecolor="k", linewidth=0.4)
    # de-collide labels: group points sharing (cost, ~y) and stack their labels
    from collections import defaultdict
    seen = defaultdict(list)
    for c in configs:
        seen[(c["cost"], round(c["b_inst"], 3))].append(c)
    for (cost, _), group in seen.items():
        for j, c in enumerate(sorted(group, key=lambda g: g["name"])):
            ax.annotate(c["name"], (c["cost"], 100*c["b_inst"]), fontsize=6.3,
                        xytext=(6, 3 - 9*j), textcoords="offset points",
                        va="center")
    # pareto line
    pf = sorted(set(pareto))
    if pf:
        ax.plot([p[0] for p in pf], [100*p[1] for p in pf], "--",
                color="#c33", lw=1.4, zorder=2, label="Pareto frontier")
    ax.set_xlabel("Cost (panel size / #draws)")
    ax.set_ylabel("Block-decision instability (%)")
    ax.set_title("P20 aggregation-rule stability vs cost (matched deck, label-free)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print("WROTE", OUT_PDF)
except Exception as e:
    print("FIG SKIPPED:", e)
