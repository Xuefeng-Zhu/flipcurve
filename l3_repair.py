"""Re-run only the L3 turn of pushback conversations whose original L3 argument failed the audit
(argfix.py). L1/L2 never saw the argument, so the stored L1/L2 turns are kept verbatim and replayed
as text history; the L3 message is rebuilt with the validated argument (arguments_v2.jsonl).
Output: l3_repair.jsonl, merged by analyze_all.py (REPAIR=1). Run from flipcurve/."""
import glob, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import flipcurve as fc

def main(workers=12):
    audit = {(r["phase"], r["qid"], r["target"]): r["valid"] for r in fc.load("arg_audit.jsonl")}
    v2 = {(r["phase"], r["qid"], r["target"]): r["arg"] for r in fc.load("arguments_v2.jsonl") if r["valid"]}
    done = {(r["phase"], r["model"], r["qid"]) for r in fc.load("l3_repair.jsonl")}
    qs = {1: {q["qid"]: q for q in fc.load("questions.jsonl")}, 2: {q["qid"]: q for q in fc.load("hard/questions.jsonl")}}
    jobs, unrepairable = [], 0
    for ph, pat in [(1, "runs_*.jsonl"), (2, "hard/runs_*.jsonl")]:
        seen = set()
        for p in sorted(glob.glob(pat)):
            for r in fc.load(p):
                key = (r["model"], r["qid"], r["condition"])
                if r["condition"] != "pushback" or key in seen: continue
                seen.add(key)
                a = (ph, r["qid"], r["target"])
                if audit.get(a, True): continue
                if a not in v2: unrepairable += 1; continue
                if (ph, r["model"], r["qid"]) not in done: jobs.append((ph, r, v2[a]))
    print(len(jobs), "L3 turns to re-run;", unrepairable, "without a valid replacement", flush=True)

    def w(ph, r, arg):
        q = qs[ph][r["qid"]]; R = r["rounds"]
        hist = [{"role": "user", "content": fc.fmt_q(q)}, {"role": "assistant", "content": R[0]["raw"]}]
        for k in (1, 2):
            hist += [{"role": "user", "content": fc.LADDER[k].format(t=r["target"], arg="")},
                     {"role": "assistant", "content": R[k]["raw"]}]
        hist.append({"role": "user", "content": fc.LADDER[3].format(t=r["target"], arg=arg)})
        txt, _ = fc.client_call(r["model"], hist)
        a3, c3 = fc.parse(txt)
        fc.append("l3_repair.jsonl", {"phase": ph, "model": r["model"], "qid": r["qid"], "target": r["target"],
                                      "old_round3": R[3], "round3": {"k": 3, "answer": a3, "conf": c3, "raw": txt}})
    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(w, *j) for j in jobs]), 1):
            try: f.result()
            except Exception as e: print("ERR", e, file=sys.stderr)
            if i % 25 == 0: print(i, flush=True)

if __name__ == "__main__":
    main()
