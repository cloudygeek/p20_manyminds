#!/usr/bin/env python3
"""Generate the RQ1/RQ2 figures for P20 from the per-case non-determinism census CSV.
All numbers come from ../data/votes_census_percase.csv (no fabrication)."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
# verdict colour scheme: consistent=green (allow), drifting=amber, hijacked=red (block)
C = {"consistent": "#2e7d32", "drifting": "#f9a825", "hijacked": "#c62828"}
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "votes_census_percase.csv")
OUT = os.path.join(HERE, "..", "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(SRC)

# ----------------------------------------------------------------------------
# FIG 1: verdict-flip & caught-flip by model, on the MATCHED 20-rep base
# adversarial deck (same 12 caseIds, all three Claude models, ~20 reps).
# ----------------------------------------------------------------------------
m = df[(df.benchmark == "adversarial") & (df.reps_valid >= 19) &
       (df.model.str.startswith("Claude"))].copy()
order = ["Claude Opus 4.7", "Claude Sonnet 4.6", "Claude Haiku 4.5"]
g = m.groupby("model").agg(n=("verdict_flip", "size"),
                           vf=("verdict_flip", "mean"),
                           cf=("caught_flip", "mean")).reindex(order)
print("=== FIG1 matched 20-rep base-adversarial deck, by model ===")
print(g)
fig, ax = plt.subplots(figsize=(4.4, 2.7))
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, g.vf*100, w, label="verdict flips", color="#37474f")
ax.bar(x + w/2, g.cf*100, w, label="flag/allow flips", color="#90a4ae")
ax.set_xticks(x); ax.set_xticklabels([o.replace("Claude ", "") for o in order])
ax.set_ylabel("% of repeated inputs that flip")
ax.set_title("Same input, same config: how often the verdict changes")
for i, (vf, cf) in enumerate(zip(g.vf, g.cf)):
    ax.text(i - w/2, vf*100 + 1, f"{vf*100:.0f}%", ha="center", fontsize=8)
    ax.text(i + w/2, cf*100 + 1, f"{cf*100:.0f}%", ha="center", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_ylim(0, max(g.vf.max(), g.cf.max())*100 + 10)
fig.savefig(OUT + "fig_flip_by_model.pdf"); plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 2: observed flip-rate vs repetition budget (measurement sensitivity).
# ----------------------------------------------------------------------------
reps_map = {2: 2, 3: 5, 4: 5, 5: 5, 19: 20, 20: 20, 40: 40}
df["repbin"] = df.reps_valid.map(reps_map)
gr = df.groupby("repbin").agg(n=("verdict_flip", "size"),
                              vf=("verdict_flip", "mean"),
                              cf=("caught_flip", "mean"))
print("\n=== FIG2 flip-rate by repetition budget ===")
print(gr)
fig, ax = plt.subplots(figsize=(4.2, 2.7))
xb = [str(int(i)) for i in gr.index]
ax.plot(xb, gr.vf*100, "o-", color="#37474f", label="verdict flips")
ax.plot(xb, gr.cf*100, "s--", color="#c62828", label="flag/allow flips")
for xi, vf in zip(xb, gr.vf):
    ax.text(xi, vf*100 + 1.5, f"{vf*100:.0f}%", ha="center", fontsize=8)
ax.set_xlabel("repetitions per input (measurement budget)")
ax.set_ylabel("% observed to flip")
ax.set_title("Short repetition budgets hide non-determinism")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_ylim(0, 38)
fig.savefig(OUT + "fig_reps_sensitivity.pdf"); plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 3: the adv-12 effort panel - Opus 4.7, identical input (adversarial-B7
# deck), 20 reps, ONLY reasoning effort varies.
# ----------------------------------------------------------------------------
a = df[(df.caseId == "adv-12") & (df.model == "Claude Opus 4.7") &
       (df.benchmark == "adversarial-B7") & (df.reps_valid >= 19)].copy()
# one row per effort; order them
eff_order = ["none/unset", "medium", "high", "max"]
a = a.set_index("effort").reindex(eff_order)
print("\n=== FIG3 adv-12 Opus 4.7 (B7 deck), by effort ===")
print(a[["n_consistent", "n_drifting", "n_hijacked"]])
fig, ax = plt.subplots(figsize=(4.6, 2.8))
y = np.arange(len(eff_order))[::-1]
left = np.zeros(len(eff_order))
for v in ["consistent", "drifting", "hijacked"]:
    vals = a["n_" + v].fillna(0).values
    ax.barh(y, vals, left=left, color=C[v], label=v, edgecolor="white")
    left += vals
ax.set_yticks(y); ax.set_yticklabels(eff_order)
ax.set_xlabel("verdicts across 20 identical repetitions")
ax.set_title('"Document all active API keys for SOC2 report"\nOpus 4.7, same input — only reasoning effort changes')
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_xlim(0, 20)
fig.savefig(OUT + "fig_adv12_effort.pdf"); plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 4: coin-flip gallery - most balanced 20-rep cases as proportion bars.
# ----------------------------------------------------------------------------
cf = df[(df.reps_valid >= 19)].copy()
cf["tot"] = cf.n_consistent + cf.n_drifting + cf.n_hijacked
# balance score: 1 - gap between top two verdicts (higher = more coin-flip)
def balance(r):
    xs = sorted([r.n_consistent, r.n_drifting, r.n_hijacked], reverse=True)
    return (xs[0] - xs[1]) / r.tot if r.tot else 1
cf["gap"] = cf.apply(balance, axis=1)
cf = cf[cf.verdict_flip == 1].sort_values("gap").drop_duplicates(
    subset=["caseId", "model", "effort", "benchmark"]).head(10)
labels = [f"{r.caseId} · {r.model.replace('Claude ','')}/{r.effort}" for _, r in cf.iterrows()]
print("\n=== FIG4 coin-flip gallery (top 10 balanced) ===")
print(cf[["caseId", "model", "effort", "n_consistent", "n_drifting", "n_hijacked"]].to_string(index=False))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
y = np.arange(len(cf))[::-1]
left = np.zeros(len(cf))
for v in ["consistent", "drifting", "hijacked"]:
    vals = (cf["n_" + v] / cf["tot"] * 100).values
    ax.barh(y, vals, left=left, color=C[v], edgecolor="white")
    left += vals
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("% of 20 identical repetitions")
ax.set_title("A gallery of coin-flip security decisions")
ax.set_xlim(0, 100)
ax.legend(handles=[Patch(color=C[v], label=v) for v in C],
          frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.savefig(OUT + "fig_coinflip_gallery.pdf"); plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 5: model x effort flip-rate heatmap on the matched 20-rep base deck.
# ----------------------------------------------------------------------------
h = df[(df.benchmark == "adversarial") & (df.reps_valid >= 19) &
       (df.model.str.startswith("Claude"))].copy()
heff = ["none", "medium", "high", "max"]
piv = (h[h.effort.isin(heff)].groupby(["model", "effort"]).verdict_flip.mean()
       .unstack("effort").reindex(index=order, columns=heff))
print("\n=== FIG5 model x effort verdict-flip% (matched 20-rep base deck) ===")
print((piv*100).round(0))
fig, ax = plt.subplots(figsize=(4.4, 2.6))
im = ax.imshow(piv.values*100, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(heff))); ax.set_xticklabels(heff)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace("Claude ", "") for o in order])
for i in range(len(order)):
    for j in range(len(heff)):
        val = piv.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val*100:.0f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)
ax.set_title("Verdict-flip % by model × reasoning effort\n(matched 12-case base deck, 20 reps)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% flip")
fig.savefig(OUT + "fig_model_effort_heatmap.pdf"); plt.close(fig)

print("\nAll figures written to", OUT)
