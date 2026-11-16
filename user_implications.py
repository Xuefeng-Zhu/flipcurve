"""User-facing implications computed from the main analysis population. Run from flipcurve/.
Writes user_numbers.json (merged into report placeholders by report/build.sh)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
exec(open("analyze_all.py").read().split("def curves")[0])
from analysis_metrics import valid_disagreement_rate
N = {}
b = baseline_df[baseline_df.conf.notna() & baseline_df.answer.notna()].copy()
b["bin"] = pd.cut(b.conf, [-1, 89.999, 94.999, 100], labels=["lt90", "90_94", "95_100"])
for k, g in b.groupby("bin", observed=True):
    N[f"acc_{k}"] = f"{100*g.ok.mean():.0f}"; N[f"acc_{k}_n"] = f"{len(g):,}"
N["share_95"] = f"{100*(b.bin=='95_100').mean():.0f}"
nz = df[(df.cond == "neutral") & (df.k > 0)]
N["neut_pool"] = f"{100*valid_disagreement_rate(zip(nz.a0, nz.ans)):.1f}"
pb = df[(df.cond == "pushback") & (df.k > 0)]
qs = np.linspace(0, 1, 10001)
def team(g):
    """Illustrative point estimates: exactly one initial answer is correct.

    User correctness q and model correctness 1-q are complementary. The
    observed correction/retention rates are assumed to transfer to these
    disagreements; both-wrong cases and sampling uncertainty are excluded.
    """
    R = g[g.correct0]; W = g[~g.correct0]
    fr = (R.ans == R.gold).mean(); fw = (W.ans == W.gold).mean()
    f = qs*fw + (1-qs)*fr; ok = qs[(f > qs) & (f > 1-qs)]
    return fr, fw, ok.min(), ok.max()
for k in (1, 3):
    fr, fw, lo, hi = team(pb[pb.k == k])
    N[f"keep_r{k}"] = f"{100*fr:.0f}"; N[f"fix_w{k}"] = f"{100*fw:.0f}"
    N[f"band_lo{k}"] = f"{100*lo:.0f}"; N[f"band_hi{k}"] = f"{100*hi:.0f}"
    N[f"team50_{k}"] = f"{100*(fr+fw)/2:.0f}"
his = [team(g)[3] for _, g in pb[pb.k == 3].groupby("model")]
N["band_hi_min"] = f"{100*min(his):.0f}"; N["band_hi_max"] = f"{100*max(his):.0f}"
json.dump(N, open("user_numbers.json", "w"), indent=1)
for k, v in N.items(): print(k, v)
