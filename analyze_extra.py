"""Robustness: category split (phase confound) and argument-length control. Run from flipcurve/."""
import json, math, glob
import pandas as pd
exec(open("analyze_all.py").read().split("def curves")[0])  # reuse loading -> df, wilson
SHARED = {"history","philosophy","engineering","law","business","health"}
alen = {}
for p, ph in [("arguments.jsonl",1), ("hard/arguments.jsonl",2)]:
    for r in load(p): alen[(ph, r["qid"], r["target"])] = len(r["arg"])
df["arglen"] = [alen.get((p,q,t)) for p,q,t in zip(df.phase, df.qid, df.target)]
pb3 = df[(df.cond=="pushback") & (df.k==3)].copy()
pd.set_option("display.width", 220)
def rate(g, col="on_target"):
    p,lo,hi = wilson(int(g[col].sum()), len(g)); return pd.Series(dict(n=len(g), rate=p, lo=lo, hi=hi))
print("== Arg length (chars) by phase & target correctness ==")
print(pd.DataFrame([dict(phase=k[0], target_correct=v) for k,v in []]) if False else "")
for ph in (1,2):
    L = [(v, k) for k,v in alen.items() if k[0]==ph]
print(df.drop_duplicates(["phase","qid","target"]).assign(tc=lambda d: d.target==d.gold).groupby(["phase","tc"]).arglen.describe()[["count","mean","50%"]].round(0))
print("\n== C(k=3) by phase, restricted to the 6 shared categories ==")
c = pb3[pb3.correct0 & pb3.cat.isin(SHARED)]
print(c.groupby(["model","phase"]).apply(rate).round(3).to_string())
print("\n== C(k=3) by category (pooled phases, all models) ==")
print(pb3[pb3.correct0].groupby(["cat","phase"]).apply(rate).round(3).to_string())
print("\n== W(k=3) by category (pooled) ==")
print(pb3[~pb3.correct0].groupby("cat").apply(rate).round(3).to_string())
print("\n== Length control: on_target at k=3 by within-phase arg-length tercile ==")
pb3["tc"] = pb3.target==pb3.gold
pb3["lt"] = pb3.groupby("phase").arglen.transform(lambda s: pd.qcut(s, 3, labels=["short","mid","long"]))
print(pb3.groupby(["tc","lt"]).apply(rate).round(3).to_string())
# logistic regression on_target ~ model + correct0 + phase + z(arglen)
try:
    import statsmodels.formula.api as smf
    pb3["zlen"] = pb3.groupby("phase").arglen.transform(lambda s:(s-s.mean())/s.std())
    pb3["y"]=pb3.on_target.astype(int); pb3["right0"]=pb3.correct0.astype(int)
    m = smf.logit("y ~ C(model) + right0*C(phase) + zlen", data=pb3.dropna(subset=["zlen"])).fit(disp=0)
    print("\n== logit y~model+right0*phase+zlen ==\n", m.summary2().tables[1][["Coef.","P>|z|"]].round(3).to_string())
except Exception as e: print("statsmodels:", e)
