"""Robustness: category split (phase confound) and argument-length control. Run from flipcurve/."""
import json, math, glob
import pandas as pd
from analysis_metrics import fit_pressure_model
exec(open("analyze_all.py").read().split("def curves")[0])  # reuse loading -> df, wilson
SHARED = {"history","philosophy","engineering","law","business","health"}
pb3 = df[(df.cond=="pushback") & (df.k==3)].copy()
pd.set_option("display.width", 220)
def rate(g, col="on_target"):
    p,lo,hi = wilson(int(g[col].sum()), len(g)); return pd.Series(dict(n=len(g), rate=p, lo=lo, hi=hi))
print("== Arg length (chars) by phase & target correctness ==\n")
print(pb3.drop_duplicates(["phase","qid","target","arg_id"]).assign(tc=lambda d: d.target==d.gold).groupby(["phase","tc"]).arglen.describe()[["count","mean","50%"]].round(0))
print("\n== C(k=3) by phase, restricted to the 6 shared categories ==")
c = pb3[pb3.correct0 & pb3.cat.isin(SHARED)]
print(c.groupby(["model","phase"]).apply(rate).round(3).to_string())
print("\n== C(k=3) by category and phase (all models) ==")
print(pb3[pb3.correct0].groupby(["cat","phase"]).apply(rate).round(3).to_string())
print("\n== C(k=3) by category, pooled phases (counts and point estimates) ==")
print(pb3[pb3.correct0].groupby("cat").on_target.agg(["sum", "count", "mean"]).round(3).to_string())
print("\n== W(k=3) by category (pooled) ==")
print(pb3[~pb3.correct0].groupby("cat").apply(rate).round(3).to_string())
print("\n== Length control: on_target at k=3 by within-phase arg-length tercile ==")
pb3["tc"] = pb3.target==pb3.gold
pb3["lt"] = pb3.groupby("phase").arglen.transform(lambda s: pd.qcut(s, 3, labels=["short","mid","long"]))
print(pb3.groupby(["tc","lt"]).apply(rate).round(3).to_string())
# logistic regression on_target ~ model + correct0 + phase + z(arglen)
try:
    m = fit_pressure_model(pb3)
    print("\n== logit y~model+right0*phase+zlen (SEs clustered by question) ==\n", m.summary2().tables[1][["Coef.","P>|z|"]].round(3).to_string())
except ImportError as e: raise SystemExit("statsmodels is required: pip install statsmodels")
