"""Every number quoted in the report prose, computed from the same loading/scoring as the tables.
Writes report_numbers.json; report/build.sh substitutes {{key}} placeholders in report.md.
Respects PHASE2 / REPAIR so a sensitivity build cannot show prose that disagrees with its tables."""
import json, numpy as np
exec(open("analyze_all.py").read().split("def curves")[0])
N = {}
d0 = df[df.k == 0]
N["n_conv"] = f"{len(d0):,}"; N["n_push"] = f"{(d0.cond=='pushback').sum():,}"; N["n_neut"] = f"{(d0.cond=='neutral').sum():,}"
N["n_q"] = f"{(d0.phase.astype(str)+'_'+d0.qid.astype(str)).nunique():,}"
N["n_repaired"] = str(n_repaired); N["n_unrep"] = str(n_unrep); N["n_invalid0"] = str(n_invalid0)
N["n_rescored"] = str(n_rescored); N["n_hit_changed"] = str(n_hit_changed); N["n_badformat"] = str(n_badformat)
pb = df[(df.cond == "pushback") & (df.k > 0)].copy(); pb["key"] = pb.phase.astype(str) + "_" + pb.qid.astype(str)
def boot(k, B=2000, seed=0):
    g = pb[pb.k == k]; keys = g.key.unique(); rng = np.random.default_rng(seed)
    S = g.groupby(["key", "correct0"]).on_target.agg(["sum", "count"]).unstack(fill_value=0)
    ws, wn, cs, cn = S["sum"][False].values, S["count"][False].values, S["sum"][True].values, S["count"][True].values
    idx = rng.integers(0, len(keys), (B, len(keys)))
    W = ws[idx].sum(1) / wn[idx].sum(1); C = cs[idx].sum(1) / cn[idx].sum(1); D = W - C
    q = lambda x: (np.quantile(x, .025), np.quantile(x, .975))
    return ws.sum()/wn.sum(), q(W), cs.sum()/cn.sum(), q(C), ws.sum()/wn.sum()-cs.sum()/cn.sum(), q(D), int(ws.sum()), int(wn.sum()), int(cs.sum()), int(cn.sum())
for k in (1, 2, 3):
    w, wq, c, cq, d, dq, wk, wn, ck, cn = boot(k)
    N[f"W{k}"] = f"{100*w:.0f}"; N[f"C{k}"] = f"{100*c:.0f}"; N[f"D{k}"] = f"{100*d:.1f}"
    N[f"W{k}x"] = f"{100*w:.1f}"; N[f"C{k}x"] = f"{100*c:.1f}"
    N[f"W{k}ci"] = f"{100*wq[0]:.0f}–{100*wq[1]:.0f}"; N[f"C{k}ci"] = f"{100*cq[0]:.0f}–{100*cq[1]:.0f}"
    N[f"D{k}ci"] = f"{100*dq[0]:.1f}, {100*dq[1]:.1f}"; N[f"D{k}f"] = f"{d:.2f}"; N[f"D{k}fci"] = f"{dq[0]:.2f}, {dq[1]:.2f}"
    N[f"W{k}frac"] = f"{wk}/{wn}"; N[f"C{k}frac"] = f"{ck}/{cn}"
N["ratio3"] = f"{float(N['W3x'])/float(N['C3x']):.0f}"
g3 = pb[pb.k == 3]
for ph in (1, 2):
    g = g3[(g3.phase == ph) & g3.correct0]; N[f"C3_p{ph}"] = f"{100*g.on_target.mean():.1f}"
ex = g3[~g3.qid.isin(FLAGGED)]; N["W3_excl"] = f"{100*ex[~ex.correct0].on_target.mean():.1f}"
N["W3_excl_diff"] = f"{float(N['W3_excl'])-float(N['W3x']):+.1f}"
# paired neutral control: same model, same Phase-1 item, identical round-0 answer in both conversations
P = df[(df.phase == 1) & (df.k == 3)]
j = P[P.cond == "pushback"].merge(P[P.cond == "neutral"], on=["model", "qid"], suffixes=("_p", "_n"))
j = j[j.a0_p == j.a0_n]
N["pair_n"] = str(len(j))
for c0, lab in ((False, "w"), (True, "c")):
    g = j[j.correct0_p == c0]
    N[f"pair_{lab}_n"] = str(len(g)); N[f"pair_{lab}_push"] = str(int(g.on_target_p.sum()))
    N[f"pair_{lab}_neut"] = str(int((g.ans_n == g.target_p).sum()))
json.dump(N, open("report_numbers.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(N, ensure_ascii=False))
# Table A1: per-model L3 estimates by phase (sample composition differs across phases)
SH = {"gemma-4-31b-it":"Gemma-4-31B","qwen3.8-27b":"Qwen3.8-27B","qwen3.5-397b":"Qwen3.5-397B",
      "glm-5.3-flash":"GLM-5.3-Flash","glm-5.2":"GLM-5.2","deepseek-v4-flash":"DeepSeek-V4-Flash*"}
def cell(g):
    if len(g) == 0: return "—"
    r = wilson(int(g.on_target.sum()), len(g)); return f"{100*r[0]:.0f} [{100*r[1]:.0f}–{100*r[2]:.0f}] ({len(g)})"
L = ["| Model | W(3) Phase 1 | W(3) Phase 2 | C(3) Phase 1 | C(3) Phase 2 |", "|---|---|---|---|---|"]
for m in sorted(g3.model.unique(), key=lambda m: SH[m]):
    g = g3[g3.model == m]
    L.append(f"| {SH[m]} | " + " | ".join(cell(g[(g.phase == ph) & (g.correct0 == c0)]) for c0 in (False, True) for ph in (1, 2)) + " |")
open("tab_phase.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
