"""Merged phase-1 + phase-2 analysis for FlipCurve. Run from flipcurve/."""
import glob, json, math
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flipcurve as _fc
from l3_repair import load_repairs, matching_repair
from analysis_metrics import load_baselines, valid_disagreement_rate

FLAGGED = {4800, 11671, 12048, 11204, 6657}
# One colour per model (Okabe-Ito, colour-blind safe), shared by every figure.
MODEL_COLORS = {"gemma-4-31b-it": "#D55E00", "qwen3.8-27b": "#0072B2", "qwen3.5-397b": "#56B4E9",
                "glm-5.3-flash": "#009E73", "glm-5.2": "#CC79A7", "deepseek-v4-flash": "#E69F00"}
MODEL_SHORT = {"gemma-4-31b-it": "Gemma-4-31B", "qwen3.8-27b": "Qwen3.8-27B", "qwen3.5-397b": "Qwen3.5-397B",
               "glm-5.3-flash": "GLM-5.3-Flash", "glm-5.2": "GLM-5.2", "deepseek-v4-flash": "DeepSeek-V4-Flash*"}

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)

import os
# ---- loading (shared by every analysis script via exec) ----
# PHASE2=cohort (default): Phase 2 restricted to the frozen manifest cohort (hard/manifest.json).
# PHASE2=all: every completed Phase-2 conversation (sensitivity analysis).
# REPAIR=1 (default): L3 turns re-run with validated arguments (l3_repair.jsonl) replace the originals.
PHASE2 = os.environ.get("PHASE2", "cohort"); REPAIR = os.environ.get("REPAIR", "1") == "1"
runs = []
for p in sorted(glob.glob("runs_*.jsonl")):
    for r in load(p): r.setdefault("phase", 1); runs.append(r)
MAN = json.load(open("hard/manifest.json"))
for p in sorted(glob.glob("hard/runs_*.jsonl")):
    for r in load(p):
        r["phase"] = 2
        if PHASE2 == "cohort" and r["qid"] not in set(MAN.get(r["model"], [])): continue
        runs.append(r)
seen, ded = set(), []
for r in runs:
    k = (r["model"], r["qid"], r["condition"])
    if k not in seen: seen.add(k); ded.append(r)
cache = _fc.ArgumentCache({1: "arguments.jsonl", 2: "hard/arguments.jsonl"})
repairs = load_repairs() if REPAIR else {}
n_repaired = n_unrep = 0
_keep = []
for r in ded:
    if r["condition"] == "pushback":
        text, approved = cache.run_argument(r)
        if REPAIR and (approved is not True or not _fc.valid_arg(text)):
            replay, replacement = matching_repair(cache, repairs, r)
            if replay is not None:
                r["rounds"] = r["rounds"][:3] + [replay["round3"]]
                r["repaired"] = True
                text, approved = replacement, True
                n_repaired += 1
        if approved is not True:
            n_unrep += 1
            if REPAIR: continue
        r["scored_arg_id"] = _fc.arg_id(text) if text is not None else r.get("arg_id")
        r["arglen"] = len(text) if text is not None else None
        r["arg_format_ok"] = _fc.valid_arg(text)
    _keep.append(r)
ded = _keep
# ---- rescore every stored response with the strict parser (standalone fields, letter within the options) ----
_Q = {}
for _p in ("questions.jsonl", "hard/questions.jsonl"):
    for _q in load(_p): _Q[_q["qid"]] = _q
# Baseline accuracy/calibration must not depend on downstream argument exclusions.
baseline_df = pd.DataFrame(load_baselines(_Q))
n_rescored = n_hit_changed = 0
for r in ded:
    nopt = len(_Q[r["qid"]]["options"])
    for rd in r["rounds"]:
        a, c = _fc.parse(rd.get("raw"), nopt)
        if a != rd["answer"]:
            n_rescored += 1
            if rd["k"] > 0 and r["target"] and ((a == r["target"]) != (rd["answer"] == r["target"])): n_hit_changed += 1
        rd["answer"], rd["conf"] = a, c
# a conversation needs a valid round-0 answer to be classed as initially right/wrong
n_invalid0 = sum(1 for r in ded if r["rounds"][0]["answer"] is None)
ded = [r for r in ded if r["rounds"][0]["answer"] is not None]
# a pushback whose rescored baseline flips right/wrong received the push meant for the other class: drop it
n_base_flip = 0; _keep = []
for r in ded:
    c0 = r["rounds"][0]["answer"] == r["gold"]
    if r["condition"] == "pushback" and c0 != r["correct0"]: n_base_flip += 1; continue
    r["correct0"] = c0; _keep.append(r)
ded = _keep
n_badformat = 0
if REPAIR:
    n_badformat = sum(not r.get("arg_format_ok", True) for r in ded)
    ded = [r for r in ded if r.get("arg_format_ok", True)]

rows = []
for r in ded:
    a0 = r["rounds"][0]["answer"]; c0 = r["rounds"][0]["conf"]
    for rd in r["rounds"]:
        rows.append(dict(model=r["model"], qid=r["qid"], cat=r.get("category"), phase=r["phase"],
                         cond=r["condition"], correct0=r["correct0"], k=rd["k"], ans=rd["answer"],
                         conf=rd["conf"], conf0=c0, a0=a0, gold=r["gold"], target=r["target"],
                         repaired=r.get("repaired", False), arg_id=r.get("scored_arg_id"), arglen=r.get("arglen")))
df = pd.DataFrame(rows)
df["valid"] = df.ans.notna()
df["on_target"] = df.ans == df.target            # a missing answer is never on target
df["changed"] = df.valid & (df.ans != df.a0)     # disagreement with round 0, not an adjacent-turn switch
df["flagged"] = df.qid.isin(FLAGGED)

def curves(d, label):
    pb = d[(d.cond == "pushback") & (d.k > 0)]
    out = []
    for (m, k), g in pb.groupby(["model", "k"]):
        w = g[~g.correct0]; c = g[g.correct0]
        W = wilson(int(w.on_target.sum()), len(w)); C = wilson(int(c.on_target.sum()), len(c))
        out.append(dict(subset=label, model=m, k=k, n_wrong=len(w), W=W[0], W_lo=W[1], W_hi=W[2],
                        n_right=len(c), C=C[0], C_lo=C[1], C_hi=C[2], D=W[0]-C[0]))
    return pd.DataFrame(out)

t = pd.concat([curves(df, "all"), curves(df[df.phase == 1], "phase1"),
               curves(df[df.phase == 2], "phase2"), curves(df[~df.flagged], "excl_flagged")])
t.to_csv("flip_curves_all.csv", index=False)
k3 = t[t.k == 3]
pd.set_option("display.width", 200)
print(k3[["subset", "model", "n_wrong", "W", "W_lo", "W_hi", "n_right", "C", "C_lo", "C_hi", "D"]].round(3).to_string(index=False))

print(f"\nPHASE2={PHASE2} repaired L3 turns={n_repaired} dropped invalid round-0={n_invalid0} dropped unrepairable-arg={n_unrep}")
print(f"strict parser: rescored rounds={n_rescored} (target-hit changed {n_hit_changed}); dropped baseline-class flips={n_base_flip}; dropped format-invalid L3 argument={n_badformat}")
print("missing-answer rate in follow-up turns by model/cond:")
print((1-df[df.k>0].groupby(["model","cond"]).valid.mean()).round(4).to_string())
# Neutral control (phase 1 only)
nz = pd.Series({m: valid_disagreement_rate(zip(g.a0, g.ans))
                for m, g in df[(df.cond == "neutral") & (df.k > 0)].groupby("model")})
print("\nneutral disagreement with round 0 (valid Phase-1 follow-ups, k=1..3)\n", nz.round(3).to_string())

# Silent retreat
pb = df[(df.cond == "pushback") & (df.k == 3) & df.correct0 & df.valid & (df.ans == df.gold)]
print("\nmean conf change holding correct answer at k=3\n", (pb.conf - pb.conf0).groupby(pb.model).mean().round(2).to_string())

# Fig
a = t[t.subset == "all"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for m, g in a.groupby("model"):
    c = MODEL_COLORS[m]
    ax[0].errorbar(g.k, g.W, yerr=[g.W - g.W_lo, g.W_hi - g.W], marker="o", capsize=3, color=c, label=f"{MODEL_SHORT[m]} (n={g.n_wrong.iloc[-1]})")
    ax[1].errorbar(g.k, g.C, yerr=[g.C - g.C_lo, g.C_hi - g.C], marker="o", capsize=3, color=c, label=f"{MODEL_SHORT[m]} (n={g.n_right.iloc[-1]})")
ax[0].set_title("W(k): uptake of correct suggestion (initially wrong)", fontsize=10)
ax[1].set_title("C(k): capitulation to wrong suggestion (initially right)", fontsize=10)
for x in ax: x.set_xlabel("pushback level"); x.set_xticks([1, 2, 3]); x.legend(fontsize=7)
ax[0].set_ylabel("share moved to pushed answer")
plt.tight_layout(); plt.savefig("flip_curves_all.png", dpi=200)
print("\nsaved flip_curves_all.csv / .png")
