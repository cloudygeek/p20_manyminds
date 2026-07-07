#!/usr/bin/env python3
"""Offline consensus analysis on the existing matched base-adversarial deck.
Each (model,effort) config has ~20 repetitions per caseId -> an empirical verdict
distribution. We Monte-Carlo draw single samples from those distributions to compare
the run-to-run STABILITY of (a) a single agent, (b) a same-model self-ensemble,
(c) a diverse multi-model panel, under majority and conjunctive (fail-closed) rules.

Honest scope: existing data is a single vendor family (Claude) at varied effort, with
no ground-truth labels -> we measure STABILITY (run-to-run agreement), not accuracy.
This is the weakest form of diversity (intra-vendor) and therefore a conservative
lower bound on what a cross-vendor panel could achieve."""
import os
import pandas as pd, numpy as np
np.random.seed(20260618)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "votes_census_percase.csv")
df = pd.read_csv(SRC)
V = ["consistent", "drifting", "hijacked"]
BLOCK = {"drifting", "hijacked"}

# matched 20-rep base adversarial deck, 3 Claude models, efforts present for all three
d = df[(df.benchmark == "adversarial") & (df.reps_valid >= 19) &
       (df.model.str.startswith("Claude"))].copy()
models = ["Claude Opus 4.7", "Claude Sonnet 4.6", "Claude Haiku 4.5"]
efforts = ["none", "medium", "high", "max"]

def dist(row):
    tot = row.n_consistent + row.n_drifting + row.n_hijacked
    return np.array([row.n_consistent, row.n_drifting, row.n_hijacked]) / tot if tot else None

def draw(p, n):  # n verdict samples from distribution p
    return np.random.choice(V, size=n, p=p)

def mode(samples):
    vals, counts = np.unique(samples, return_counts=True)
    return vals[np.argmax(counts)]

T = 4000  # Monte-Carlo trials per case

def instability_single(p):
    """P(two independent draws give different verdicts) = 1 - sum p_v^2."""
    return 1 - np.sum(p**2)

def instability_majority(member_dists, k_each=1):
    """Run-to-run instability of the majority verdict of a panel.
    member_dists: list of per-member verdict distributions; each contributes k_each draws."""
    outs = []
    for _ in range(T):
        samples = []
        for p in member_dists:
            samples += list(draw(p, k_each))
        outs.append(mode(np.array(samples)))
    outs = np.array(outs)
    m = mode(outs)
    return np.mean(outs != m)

def instability_block_conjunctive(member_dists):
    """Run-to-run instability of the BINARY block decision under the conjunctive
    (fail-closed) rule: block if ANY member's drawn sample is in BLOCK."""
    outs = []
    for _ in range(T):
        block = any(draw(p, 1)[0] in BLOCK for p in member_dists)
        outs.append(block)
    outs = np.array(outs)
    frac_block = outs.mean()
    return min(frac_block, 1 - frac_block)  # instability = distance from a deterministic decision

def instability_block_single(p):
    pb = p[1] + p[2]  # P(block) for a single draw
    return min(pb, 1 - pb)

# ---- per effort: build the 3-model panel on the 12 shared cases ----
rows = []
for eff in efforts:
    sub = d[d.effort == eff]
    # require all three models present for the case
    cases = sorted(set.intersection(*[set(sub[sub.model == m].caseId) for m in models])
                   if all(len(sub[sub.model == m]) for m in models) else set())
    if not cases:
        continue
    single_inst, selfens_inst, panel_inst = [], [], []
    block_single, block_conj = [], []
    for cid in cases:
        dists = {}
        for m in models:
            r = sub[(sub.model == m) & (sub.caseId == cid)]
            if len(r):
                dists[m] = dist(r.iloc[0])
        if len(dists) < 3 or any(p is None for p in dists.values()):
            continue
        members = list(dists.values())
        # single agent = average over the 3 members of their own single-draw instability
        single_inst.append(np.mean([instability_single(p) for p in members]))
        block_single.append(np.mean([instability_block_single(p) for p in members]))
        # self-ensemble = majority of 3 draws from ONE model, averaged over the 3 models
        selfens_inst.append(np.mean([instability_majority([p], k_each=3) for p in members]))
        # diverse panel = majority of 1 draw from each of the 3 models
        panel_inst.append(instability_majority(members, k_each=1))
        block_conj.append(instability_block_conjunctive(members))
    if single_inst:
        rows.append(dict(effort=eff, n_cases=len(single_inst),
                         single=np.mean(single_inst),
                         self_ensemble=np.mean(selfens_inst),
                         diverse_panel=np.mean(panel_inst),
                         block_single=np.mean(block_single),
                         block_conjunctive=np.mean(block_conj)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 140, "display.float_format", lambda x: f"{x:.3f}")
print("=== Run-to-run decision INSTABILITY (lower = more stable), matched base deck ===")
print("single = avg single-model single-draw instability; self_ensemble = majority of 3 draws,")
print("same model; diverse_panel = majority of 1 draw from each of 3 models.\n")
print(res.to_string(index=False))
print("\n=== Means across efforts ===")
for c in ["single", "self_ensemble", "diverse_panel", "block_single", "block_conjunctive"]:
    print(f"  {c:20s} {res[c].mean():.3f}")
red_self = 100 * (1 - res.self_ensemble.mean() / res.single.mean())
red_div = 100 * (1 - res.diverse_panel.mean() / res.single.mean())
red_div_vs_self = 100 * (1 - res.diverse_panel.mean() / res.self_ensemble.mean())
print(f"\n  self-ensemble cuts verdict instability by {red_self:.0f}% vs single")
print(f"  diverse panel cuts verdict instability by {red_div:.0f}% vs single")
print(f"  diverse panel vs self-ensemble: {red_div_vs_self:+.0f}% (negative => panel less stable)")
red_block = 100 * (1 - res.block_conjunctive.mean() / res.block_single.mean())
print(f"  conjunctive fail-closed cuts BLOCK-decision instability by {red_block:.0f}% vs single")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(5.0, 2.9))
x = np.arange(len(res)); w = 0.26
ax.bar(x - w, res.single, w, label="single agent", color="#c62828")
ax.bar(x, res.self_ensemble, w, label="self-ensemble ($3\\times$ one model)", color="#f9a825")
ax.bar(x + w, res.diverse_panel, w, label="diverse panel (3 models)", color="#2e7d32")
ax.set_xticks(x); ax.set_xticklabels(res.effort)
ax.set_xlabel("reasoning effort (panel members matched)")
ax.set_ylabel("run-to-run verdict instability")
ax.set_title("Diverse consensus stabilises the verdict more than re-sampling one model\n(matched base deck, equal 3-vote budget; intra-vendor lower bound)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.savefig(os.path.join(HERE, "..", "figures", "fig_consensus_stability.pdf"))
print("\nwrote fig_consensus_stability.pdf")

# ----------------------------------------------------------------------------
# FIG 6: flip-rate by reconstructed temperature. Harness rule (inference client):
#   temperature = effort ? 1.0 : 0.1  for non-Opus-4.7 (Opus 4.7 rejects the param).
#   Extended thinking (any effort) forces temp 1; effort 'none' runs near-greedy 0.1.
# ----------------------------------------------------------------------------
full = pd.read_csv(SRC)
def temp_of(r):
    if r.model == "Claude Opus 4.7":
        return None  # temperature never set for 4.7
    return "0.1" if r.effort in ("none", "none/unset") else "1.0"
full["temp"] = full.apply(temp_of, axis=1)
mt = full[(full.benchmark == "adversarial") & (full.reps_valid >= 19) &
          (full.model.isin(["Claude Sonnet 4.6", "Claude Haiku 4.5"]))]
gt = mt.groupby("temp").agg(n=("verdict_flip", "size"), vf=("verdict_flip", "mean"),
                            cf=("caught_flip", "mean")).reindex(["0.1", "1.0"])
print("\n=== FIG6 flip by reconstructed temperature (non-Opus-4.7, matched deck) ===")
print(gt)
fig, ax = plt.subplots(figsize=(3.6, 2.7))
x = np.arange(2); w = 0.38
ax.bar(x - w/2, gt.vf*100, w, label="verdict flips", color="#37474f")
ax.bar(x + w/2, gt.cf*100, w, label="block-decision flips", color="#c62828")
ax.set_xticks(x); ax.set_xticklabels(["temp 0.1\n(near-greedy,\nno thinking)", "temp 1.0\n(default, +\nreasoning)"])
ax.set_ylabel("% of repeated inputs that flip")
ax.set_title("Temperature is the dominant driver —\nbut near-greedy still flips on hard security calls")
for i, (vf, cf) in enumerate(zip(gt.vf, gt.cf)):
    ax.text(i - w/2, vf*100 + 1, f"{vf*100:.0f}%", ha="center", fontsize=8)
    ax.text(i + w/2, cf*100 + 1, f"{cf*100:.0f}%", ha="center", fontsize=8)
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_ylim(0, 72)
fig.savefig(os.path.join(HERE, "..", "figures", "fig_temperature.pdf"))
print("wrote fig_temperature.pdf")
res.to_csv(os.path.join(HERE, "..", "data", "consensus_sim_results.csv"), index=False)
