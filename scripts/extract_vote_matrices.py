#!/usr/bin/env python3
"""
Extract anonymised per-repetition vote matrices from the raw judge corpus.

This is the provenance script that produced the processed CSVs shipped in
``artifact/data/``.  It reads the raw per-(model, case) repetition JSONs from
the authors' private evaluation corpus and writes flat, de-identified CSVs that
contain ONLY the released fields:

    wave, deck, model, vendor, persona, caseId, expectedVerdict, expectedCaught,
    rep, verdict, confidence, blocked, hijacked

It deliberately DROPS every free-text and identifying field from the raw runs:
the model's ``reasoning`` text, the system ``prompt`` body, build/invocation
metadata, timestamps, token counts, region, and any file paths.  The released
CSVs are therefore sufficient to reproduce every number, table, and figure in
the paper, but carry no proprietary prompt text or identifying metadata.

Reviewers and re-users do NOT need to run this script: the processed CSVs are
already in ``artifact/data/``.  It is included for transparency of provenance.

Usage (authors only, needs the private corpus):
    python extract_vote_matrices.py --raw-root /path/to/private/results -o ../data
"""
import argparse
import csv
import glob
import json
import os

# block = the deployed fail-closed gate decision (drifting OR hijacked).
BLOCK = {"drifting", "hijacked"}

# Map raw provider model ids to short public names + vendor.
MODEL_NAMES = {
    "claude-opus-4-8": ("opus-4-8", "Anthropic"),
    "claude-opus-4-7": ("opus-4-7", "Anthropic"),
    "claude-sonnet-4-6": ("sonnet-4-6", "Anthropic"),
    "claude-haiku-4-5": ("haiku-4-5", "Anthropic"),
    "qwen3-235b": ("qwen3-235b", "Alibaba"),
    "nova-pro": ("nova-pro", "Amazon"),
    "gpt-oss-120b": ("gpt-oss-120b", "OpenAI-OW"),
    "glm-4.7": ("glm-4.7", "Zhipu"),
    "gpt-4o": ("gpt-4o", "OpenAI"),
    "deepseek-v3.2": ("deepseek-v3.2", "DeepSeek"),
    "kimi-k2": ("kimi-k2", "Moonshot"),
}


def short_model(mid):
    """Best-effort map a raw model id to (public_name, vendor)."""
    m = mid.lower()
    if "opus-4-8" in m or "opus.4.8" in m or "opus-4.8" in m:
        return MODEL_NAMES["claude-opus-4-8"]
    if "opus-4-7" in m or "opus.4.7" in m:
        return MODEL_NAMES["claude-opus-4-7"]
    if "sonnet-4-6" in m or "sonnet.4.6" in m:
        return MODEL_NAMES["claude-sonnet-4-6"]
    if "haiku-4-5" in m or "haiku.4.5" in m:
        return MODEL_NAMES["claude-haiku-4-5"]
    if "qwen" in m:
        return MODEL_NAMES["qwen3-235b"]
    if "nova" in m:
        return MODEL_NAMES["nova-pro"]
    if "gpt-oss" in m:
        return MODEL_NAMES["gpt-oss-120b"]
    if "glm" in m or "zai" in m:
        return MODEL_NAMES["glm-4.7"]
    if "gpt-4o" in m:
        return MODEL_NAMES["gpt-4o"]
    if "deepseek" in m or m.endswith(".v3") or ".v3." in m:
        return MODEL_NAMES["deepseek-v3.2"]
    if "kimi" in m:
        return MODEL_NAMES["kimi-k2"]
    return (mid, "unknown")


# wave label -> (glob of cell directories under raw-root, judge-file glob, deck label)
WAVES = {
    "pilot": ("p20-consensus-20260618/p20-consensus-*", "adversarial-judge-*.json", "adv12-hijack"),
    "hard": ("p20-hard-20260619/p20-hard-*", "adversarial-judge-*.json", "hard-nearmiss"),
    "wave2_persona": ("p20-followup-20260619/p20-w2-persona-*", "adversarial-judge-*.json", "hard-nearmiss"),
    "primevul": ("p20-primevul-20260619/p20-primevul-*", "adversarial-judge-*.json", "primevul-100"),
    "perturb": ("p20-perturb-20260619/p20-perturb-*", "adversarial-judge-*.json", "adv12-perturb"),
    "tempsweep": ("p20-tempsweep-20260619/p20-tempsweep-*", "adversarial-judge-*.json", "adv12-tempsweep"),
}

FIELDS = ["wave", "deck", "model", "vendor", "persona", "caseId",
          "expectedVerdict", "expectedCaught", "rep", "verdict",
          "confidence", "blocked", "hijacked"]


def rows_for_cell(wave, deck_label, path):
    j = json.load(open(path))
    mid = j.get("model", {}).get("id", "unknown")
    name, vendor = short_model(mid)
    persona = (j.get("sampling", {}) or {}).get("persona", "persona-neutral")
    persona = persona.replace("persona-", "")
    if "tempsweep" in path:
        # temperature sweep encodes the requested temperature in the cell.
        t = (j.get("sampling", {}) or {}).get("requestedTemperature")
        if t is not None:
            persona = f"neutral-t{t}"
    out = []
    for c in j.get("cases", []):
        cid = c.get("caseId")
        ev = c.get("expectedVerdict", "")
        ec = c.get("expectedCaught", "")
        reps = c.get("reps") or []
        for r in reps:
            v = r.get("verdict")
            out.append({
                "wave": wave, "deck": deck_label, "model": name, "vendor": vendor,
                "persona": persona, "caseId": cid, "expectedVerdict": ev,
                "expectedCaught": ec, "rep": r.get("rep"), "verdict": v,
                "confidence": r.get("confidence"),
                "blocked": int(v in BLOCK), "hijacked": int(v == "hijacked"),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True,
                    help="root of the private results tree (authors only)")
    ap.add_argument("-o", "--out", default="../data", help="output dir for CSVs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for wave, (celldir_glob, jf_glob, deck_label) in WAVES.items():
        rows = []
        for d in sorted(glob.glob(os.path.join(args.raw_root, celldir_glob))):
            if not os.path.isdir(d):
                continue
            for f in sorted(glob.glob(os.path.join(d, jf_glob))):
                rows.extend(rows_for_cell(wave, deck_label, f))
        if not rows:
            print(f"[skip] {wave}: no cells found")
            continue
        outpath = os.path.join(args.out, f"votes_{wave}.csv")
        with open(outpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[ok]  {wave}: {len(rows)} rep-rows -> {outpath}")


if __name__ == "__main__":
    main()
