"""Question-cluster bootstrap for pooled W(k), C(k), D(k) (same questions recur across models). Run from flipcurve/."""
import numpy as np
exec(open("analyze_all.py").read().split("def curves")[0])
rng = np.random.default_rng(0); B = 2000
pb = df[(df.cond == "pushback") & (df.k > 0)].copy()
pb["key"] = pb.phase.astype(str) + "_" + pb.qid.astype(str)
for k in (1, 2, 3):
    g = pb[pb.k == k]
    # per-question sums so one resample is a vector op
    s = g.groupby(["key", "correct0"]).on_target.agg(["sum", "count"]).unstack(fill_value=0)
    ws, wn = s[("sum", False)].values, s[("count", False)].values
    cs, cn = s[("sum", True)].values, s[("count", True)].values
    idx = np.random.default_rng(0).integers(0, len(s), (B, len(s)))  # seed 0 per k, as report_numbers.py
    W = ws[idx].sum(1) / wn[idx].sum(1); C = cs[idx].sum(1) / cn[idx].sum(1); D = W - C
    pt = lambda a, b: a.sum() / b.sum()
    q = lambda x: f"[{100*np.quantile(x,.025):.1f}, {100*np.quantile(x,.975):.1f}]"
    print(f"k={k} questions={len(s)}  W={100*pt(ws,wn):.1f} {q(W)}  C={100*pt(cs,cn):.1f} {q(C)}  D={100*(pt(ws,wn)-pt(cs,cn)):.1f} {q(D)}  n_wrong={wn.sum()} n_right={cn.sum()}")
