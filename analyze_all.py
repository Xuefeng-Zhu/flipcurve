"""Merged phase-1 + phase-2 analysis for FlipCurve. Run from flipcurve/."""
import glob, json, math
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLAGGED = {4800, 11671, 12048, 11204, 6657}

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)

runs = []
for p in sorted(glob.glob("runs_*.jsonl")):
    for r in load(p): r.setdefault("phase", 1); runs.append(r)
for p in sorted(glob.glob("hard/runs_*.jsonl")):
    for r in load(p): r["phase"] = 2; runs.append(r)
seen, ded = set(), []
for r in runs:
    k = (r["model"], r["qid"], r["condition"])
    if k not in seen: seen.add(k); ded.append(r)

rows = []
for r in ded:
    a0 = r["rounds"][0]["answer"]; c0 = r["rounds"][0]["conf"]
    for rd in r["rounds"]:
        rows.append(dict(model=r["model"], qid=r["qid"], cat=r.get("category"), phase=r["phase"],
                         cond=r["condition"], correct0=r["correct0"], k=rd["k"], ans=rd["answer"],
                         conf=rd["conf"], conf0=c0, a0=a0, gold=r["gold"], target=r["target"]))
df = pd.DataFrame(rows)
df["on_target"] = df.ans == df.target
df["changed"] = df.ans != df.a0
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

# Neutral control (phase 1 only)
nz = df[(df.cond == "neutral") & (df.k > 0)].groupby("model").changed.mean()
print("\nneutral change rate (phase1, k=1..3)\n", nz.round(3).to_string())

# Silent retreat
pb = df[(df.cond == "pushback") & (df.k == 3) & df.correct0 & ~df.changed]
print("\nmean conf change holding correct answer at k=3\n", (pb.conf - pb.conf0).groupby(pb.model).mean().round(2).to_string())

# Fig
a = t[t.subset == "all"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for m, g in a.groupby("model"):
    ax[0].errorbar(g.k, g.W, yerr=[g.W - g.W_lo, g.W_hi - g.W], marker="o", capsize=3, label=f"{m} (n={g.n_wrong.iloc[-1]})")
    ax[1].errorbar(g.k, g.C, yerr=[g.C - g.C_lo, g.C_hi - g.C], marker="o", capsize=3, label=f"{m} (n={g.n_right.iloc[-1]})")
ax[0].set_title("Warranted update (initially wrong) - higher is better")
ax[1].set_title("Capitulation (initially right) - lower is better")
for x in ax: x.set_xlabel("pushback level"); x.set_xticks([1, 2, 3]); x.legend(fontsize=7)
ax[0].set_ylabel("share moved to pushed answer")
plt.tight_layout(); plt.savefig("flip_curves_all.png", dpi=200)
print("\nsaved flip_curves_all.csv / .png")
