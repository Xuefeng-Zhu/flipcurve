"""L3 outcome by round-0 stated confidence (Table 2). Pooled across models, so intervals are a
question-cluster bootstrap (2,000 resamples, seed 0): the same question recurs across models."""
import numpy as np
exec(open("analyze_all.py").read().split("def curves")[0])
p = df[(df.cond == "pushback") & (df.k == 3)].copy()
p["bin"] = pd.cut(p.conf0, [-1, 79, 89, 94, 100], labels=["<80", "80-89", "90-94", "95-100"])
p["key"] = p.phase.astype(str) + "_" + p.qid.astype(str)
def cboot(g, B=2000, seed=0):
    s = g.groupby("key").on_target.agg(["sum", "count"]); a, n = s["sum"].values, s["count"].values
    idx = np.random.default_rng(seed).integers(0, len(s), (B, len(s)))
    r = a[idx].sum(1) / n[idx].sum(1)
    return a.sum() / n.sum(), np.quantile(r, .025), np.quantile(r, .975), len(s)
rows = {}
for wrong, g in p.groupby(~p.correct0):
    print("initially", "WRONG -> W" if wrong else "RIGHT -> C")
    for b, gb in g.groupby("bin", observed=True):
        r = cboot(gb); rows[(wrong, b)] = (r, len(gb))
        print(f"  conf0 {b}: n={len(gb)} questions={r[3]} rate={100*r[0]:.0f}% [{100*r[1]:.0f},{100*r[2]:.0f}]")
w = p[~p.correct0].bin.value_counts(normalize=True); c = p[p.correct0].groupby("bin", observed=True).on_target.mean()
print("C reweighted to W conf distribution:", round(100 * (w * c).sum(), 1))
L = ["| Round-0 confidence | W(3): initially wrong → correct | C(3): initially right → wrong |", "|---|---|---|"]
fmt = lambda x: f"{100*x[0][0]:.0f}% [{100*x[0][1]:.0f}, {100*x[0][2]:.0f}] (n = {x[1]:,}; {x[0][3]} q)"
for b in ["<80", "80-89", "90-94", "95-100"]:
    L.append(f"| {b.replace('-', '–').replace('<', '< ')} | {fmt(rows[(True, b)])} | {fmt(rows[(False, b)])} |")
open("tab_conf.md", "w").write("\n".join(L) + "\n")
