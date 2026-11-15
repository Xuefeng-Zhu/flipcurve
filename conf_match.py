exec(open("analyze_all.py").read().split("def curves")[0])
p=df[(df.cond=="pushback")&(df.k==3)].copy()
p["bin"]=pd.cut(p.conf0,[-1,79,89,94,100],labels=["<80","80-89","90-94","95-100"])
for wrong,g in p.groupby(~p.correct0):
    print("initially", "WRONG -> W" if wrong else "RIGHT -> C")
    for b,gb in g.groupby("bin"):
        r=wilson(int(gb.on_target.sum()),len(gb)); print(f"  conf0 {b}: n={len(gb)} rate={100*r[0]:.0f}% [{100*r[1]:.0f},{100*r[2]:.0f}]")
# mixed-confidence reweighting: C reweighted to W's conf distribution
w=p[~p.correct0].bin.value_counts(normalize=True); c=p[p.correct0].groupby("bin").on_target.mean()
print("C reweighted to W conf distribution:", round(100*(w*c).sum(),1))
