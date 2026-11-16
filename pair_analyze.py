"""Analyse the paired argument experiment (run from flipcurve/ after pair_exp.py).

Adoption = flipcurve.parse(reply, n_options) equals the target; an unparseable reply counts as not adopted.
Delta = (only-B - only-A) / pairs, in percentage points; dW on correct suggestions, dC on wrong ones.
95% CI: paired bootstrap stratified by phase, resampling questions within each phase (all models and both
branches of a sampled question move together). Writes pair/results.md and pair/results.json.
"""
import json
from collections import defaultdict
import numpy as np
import flipcurve as fc

B_N, SEED = 5000, 20261115
pairs = fc.load("pair/pairs.jsonl")
raw = {}
for r in fc.load("pair/raw_responses.jsonl"):
    raw.setdefault((r["pair_id"], r["branch"]), r)  # first stored reply per (pair, branch)

rows = []
for p in pairs:
    a, b = raw.get((p["pair_id"], "A")), raw.get((p["pair_id"], "B"))
    if not (a and b): continue
    pa, pb = (fc.parse(x["raw"], p["n_options"])[0] for x in (a, b))
    rows.append(dict(p, A=pa == p["target"], B=pb == p["target"], invA=pa is None, invB=pb is None))
complete = {r["pair_id"] for r in rows}
missing = defaultdict(int)
for p in pairs:
    if p["pair_id"] not in complete: missing[(p["phase"], p["model"], p["kind"])] += 1


def delta(rs):
    n = len(rs); ob = sum(r["B"] and not r["A"] for r in rs); oa = sum(r["A"] and not r["B"] for r in rs)
    return 100 * (ob - oa) / n if n else float("nan")


def boot(rs):
    rng = np.random.default_rng(SEED)
    strata = defaultdict(lambda: defaultdict(list))
    for r in rs: strata[r["phase"]][r["qid"]].append(r)
    out = []
    for _ in range(B_N):
        s = []
        for ph, qmap in strata.items():
            qs = list(qmap)
            for q in rng.choice(len(qs), len(qs)): s += qmap[qs[q]]
        out.append(delta(s))
    return np.percentile(out, [2.5, 97.5])


res, lines = {}, []
hdr = "| Suggestion | Pairs | A adopted | B adopted | Only B | Only A | Delta (pp) [95% CI] | Invalid A / B |"
lines += [hdr, "|" + "---|" * 8]
for kind, label, dname in (("W", "Correct (initially wrong)", "dW"), ("C", "Wrong (initially right)", "dC")):
    rs = [r for r in rows if r["kind"] == kind]; n = len(rs)
    if not n: continue
    nA, nB = sum(r["A"] for r in rs), sum(r["B"] for r in rs)
    ob, oa = sum(r["B"] and not r["A"] for r in rs), sum(r["A"] and not r["B"] for r in rs)
    d = delta(rs); lo, hi = boot(rs)
    iA, iB = sum(r["invA"] for r in rs), sum(r["invB"] for r in rs)
    res[dname] = dict(pairs=n, A=nA, B=nB, only_B=ob, only_A=oa, delta_pp=d, ci=[lo, hi], invalid_A=iA, invalid_B=iB,
                      by_phase={ph: dict(pairs=len(g), A=sum(r["A"] for r in g), B=sum(r["B"] for r in g), delta_pp=delta(g))
                                for ph in (1, 2) for g in [[r for r in rs if r["phase"] == ph]] if g})
    lines.append(f"| {label} | {n} | {nA} ({100*nA/n:.1f}%) | {nB} ({100*nB/n:.1f}%) | {ob} | {oa} | "
                 f"{dname[1]}: {d:+.1f} [{lo:+.1f}, {hi:+.1f}] | {iA} / {iB} |")
res["complete_pairs"] = len(rows); res["total_pairs"] = len(pairs)
res["missing"] = {f"phase{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(missing.items())}
lines += ["", f"Complete pairs: {len(rows)}/{len(pairs)}. Bootstrap: {B_N} resamples, stratified by phase, clustered by question."]
if missing: lines += ["Missing pairs (phase, model, kind): " + ", ".join(f"{k}: {v}" for k, v in res["missing"].items())]
lines += ["", "By model (A% / B% adopted, delta pp):", "", "| Model | W pairs | W: A / B / delta | C pairs | C: A / B / delta |", "|---|---|---|---|---|"]
for m in sorted({r["model"] for r in rows}):
    cells = []
    for kind in ("W", "C"):
        g = [r for r in rows if r["model"] == m and r["kind"] == kind]
        cells += [str(len(g)), f"{100*np.mean([r['A'] for r in g]):.0f} / {100*np.mean([r['B'] for r in g]):.0f} / {delta(g):+.1f}" if g else "-"]
    lines.append(f"| {m} | " + " | ".join(cells) + " |")
open("pair/results.md", "w").write("\n".join(lines) + "\n")
json.dump(res, open("pair/results.json", "w"), indent=1, default=float)
print("\n".join(lines))
