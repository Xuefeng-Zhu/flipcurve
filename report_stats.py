"""Numbers + figures for the report. Run from flipcurve/ after analyze_all.py."""
import json, glob, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("analyze_all.py").read().split("def curves")[0])
SHORT = {"gemma-4-31b-it":"Gemma-4-31B","qwen3.8-27b":"Qwen3.8-27B","qwen3.5-397b":"Qwen3.5-397B",
         "glm-5.3-flash":"GLM-5.3-Flash","glm-5.2":"GLM-5.2","deepseek-v4-flash":"DeepSeek-V4-Flash*"}
out = {}
# round-0: phase1 runs (one per model,qid) + phase2 screens
r0 = []
for r in ded:
    if r["phase"]==1: r0.append(dict(model=r["model"],qid=r["qid"],phase=1,ok=r["correct0"],conf=r["rounds"][0]["conf"]))
for p in glob.glob("hard/screen_*.jsonl"):
    for s in load(p):
        a, c = _fc.parse(s.get("raw"), len(_Q[s["qid"]]["options"]))  # strict parser, as in analyze_all
        if a: r0.append(dict(model=s["model"],qid=s["qid"],phase=2,ok=a==s["gold"],conf=c))
r0 = pd.DataFrame(r0).drop_duplicates(["model","qid"]).dropna(subset=["conf"])
def ece(g, bins=10):
    c = g.conf.clip(0,100)/100; b = np.minimum((c*bins).astype(int), bins-1); e=0
    for i in range(bins):
        m=b==i
        if m.any(): e += m.mean()*abs(g.ok[m].mean()-c[m].mean())
    return e
pb = df[(df.cond=="pushback")&(df.k>0)]
rows=[]
for m,g in pb.groupby("model"):
    row=dict(model=SHORT[m])
    for k in (1,2,3):
        gk=g[g.k==k]; w=gk[~gk.correct0]; c=gk[gk.correct0]
        row[f"W{k}"]=wilson(int(w.on_target.sum()),len(w)); row[f"C{k}"]=wilson(int(c.on_target.sum()),len(c))
        row["nW"]=len(w); row["nC"]=len(c)
    rz=r0[r0.model==m]; row["n0"]=len(rz); row["acc0"]=rz.ok.mean(); row["ece"]=ece(rz); row["conf0"]=rz.conf.mean()
    nz=df[(df.model==m)&(df.cond=="neutral")&(df.k>0)]; row["neutral"]=nz.changed.mean() if len(nz) else float("nan")
    h=g[(g.k==3)&g.correct0&~g.changed]; row["dconf"]=(h.conf-h.conf0).mean()
    # flip back: initially-wrong, moved to target at some k then left by k3? skip
    rows.append(row)
T=pd.DataFrame(rows).sort_values("model")
def f(x): return f"{100*x[0]:.0f} [{100*x[1]:.0f}–{100*x[2]:.0f}]"
lines=["| Model | n wrong/right | W(3) % | C(3) % | D(3) | Acc0 % (n) | ECE | Neutral % | Δconf |","|---|---|---|---|---|---|---|---|---|"]
for _,r in T.iterrows():
    lines.append(f"| {r.model} | {r.nW}/{r.nC} | {f(r.W3)} | {f(r.C3)} | {r.W3[0]-r.C3[0]:.2f} | {100*r.acc0:.0f} ({r.n0}) | {r.ece:.3f} | "+("—" if math.isnan(r.neutral) else f"{100*r.neutral:.1f}")+f" | {r.dconf:+.1f} |")
open("tab_main.md","w").write("\n".join(lines)+"\n"); print("\n".join(lines))
print("\nper-k:")
for _,r in T.iterrows(): print(r.model, "W", [round(100*r[f"W{k}"][0]) for k in (1,2,3)], "C", [round(100*r[f"C{k}"][0],1) for k in (1,2,3)])
# pooled
for k in (1,2,3):
    gk=pb[pb.k==k]; w=gk[~gk.correct0]; c=gk[gk.correct0]
    print("pooled k",k,"W",f(wilson(int(w.on_target.sum()),len(w))),"C",f(wilson(int(c.on_target.sum()),len(c))))
print("total convs", len(ded[:]), "pushback convs", sum(1 for r in ded if r["condition"]=="pushback"), "neutral", sum(1 for r in ded if r["condition"]=="neutral"))
print("round0 items", len(r0))
# Fig 2: W vs C at k=3 scatter with CIs
fig,ax=plt.subplots(figsize=(5,4))
_COL={MODEL_SHORT[m]:c for m,c in MODEL_COLORS.items()}
for _,r in T.iterrows():
    ax.errorbar(100*r.C3[0],100*r.W3[0],color=_COL[r.model],xerr=[[100*(r.C3[0]-r.C3[1])],[100*(r.C3[2]-r.C3[0])]],yerr=[[100*(r.W3[0]-r.W3[1])],[100*(r.W3[2]-r.W3[0])]],fmt="o",capsize=3)
    # labels placed in empty space with leader lines so none sits on an error bar
    pos={"Gemma-4-31B":(15,84),"DeepSeek-V4-Flash*":(8,90),"Qwen3.8-27B":(15,72),"Qwen3.5-397B":(15,46),
         "GLM-5.2":(15,60),"GLM-5.3-Flash":(4,28)}[r.model]
    ax.annotate(r.model,(100*r.C3[0],100*r.W3[0]),xytext=pos,fontsize=8,color=_COL[r.model],
                arrowprops=dict(arrowstyle="-",color=_COL[r.model],lw=0.6,alpha=0.7))
ax.set_xlabel("Capitulation C(3), % (lower is better)"); ax.set_ylabel("Uptake of correct suggestion W(3), %")
ax.set_xlim(0,25); ax.set_ylim(20,100); ax.grid(alpha=.3); ax.set_title("W(3) vs. C(3) at full pressure (L3)")
plt.tight_layout(); plt.savefig("fig_wc.png",dpi=200)
