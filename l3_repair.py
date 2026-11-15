"""Re-run only the L3 turn of pushback conversations whose argument failed audit or format checks.
L1/L2 never saw the argument, so the stored L1/L2 turns are kept verbatim and replayed
as text history; the L3 message is rebuilt with the validated argument (arguments_v2.jsonl).
Output: l3_repair.jsonl, merged by analyze_all.py (REPAIR=1). Run from flipcurve/."""
import glob, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import flipcurve as fc


def history_id(run):
    history = json.dumps([rd["raw"] for rd in run["rounds"][:3]], ensure_ascii=False)
    return fc.arg_id(history)


def load_repairs():
    repairs = {}
    for row in fc.load("l3_repair.jsonl"):
        key = fc.argument_key(row), row["model"]
        repairs.setdefault(key, []).append(row)
    return repairs


def matching_repair(cache, repairs, run):
    """Find a replay of this exact run and resolve the replacement argument it used."""
    key = fc.argument_key(run)
    run_arg_id = run.get("arg_id")
    old_id = run_arg_id
    if old_id is None and key in cache.originals:
        old_id = fc.arg_id(cache.originals[key])
    for replay in reversed(repairs.get((key, run["model"]), ())):
        if replay.get("old_round3") != run["rounds"][3]:
            continue
        replay_old_id = replay.get("old_arg_id")
        if replay_old_id != old_id and (replay_old_id is not None or run_arg_id is not None):
            continue
        if replay.get("history_id") not in (None, history_id(run)):
            continue
        fingerprint = replay.get("arg_id")
        if fingerprint is not None:
            text = cache.replacement_versions.get((key, fingerprint))
        else:
            # Pre-fingerprint replays used the last approved legacy replacement for the key.
            text = cache.legacy_replacements.get(key) if run_arg_id is None else None
        if fc.valid_arg(text):
            return replay, text
    return None, None


def main(workers=12):
    cache = fc.ArgumentCache({1: "arguments.jsonl", 2: "hard/arguments.jsonl"})
    repairs = load_repairs()
    qs = {1: {q["qid"]: q for q in fc.load("questions.jsonl")}, 2: {q["qid"]: q for q in fc.load("hard/questions.jsonl")}}
    jobs, unrepairable = [], 0
    for ph, pat in [(1, "runs_*.jsonl"), (2, "hard/runs_*.jsonl")]:
        seen = set()
        for p in sorted(glob.glob(pat)):
            for r in fc.load(p):
                key = (r["model"], r["qid"], r["condition"])
                if r["condition"] != "pushback" or key in seen: continue
                seen.add(key)
                r["phase"] = ph
                text, audit_ok = cache.run_argument(r)
                if audit_ok is True and fc.valid_arg(text): continue
                replay, _ = matching_repair(cache, repairs, r)
                if replay is not None: continue
                replacement = cache.replacements.get(fc.argument_key(r))
                if replacement is None: unrepairable += 1; continue
                jobs.append((ph, r, replacement))
    print(len(jobs), "L3 turns to re-run;", unrepairable, "without a valid replacement", flush=True)

    def w(ph, r, arg):
        q = qs[ph][r["qid"]]; R = r["rounds"]
        hist = [{"role": "user", "content": fc.fmt_q(q)}, {"role": "assistant", "content": R[0]["raw"]}]
        for k in (1, 2):
            hist += [{"role": "user", "content": fc.LADDER[k].format(t=r["target"], arg="")},
                     {"role": "assistant", "content": R[k]["raw"]}]
        hist.append({"role": "user", "content": fc.LADDER[3].format(t=r["target"], arg=arg)})
        txt, _ = fc.client_call(r["model"], hist)
        a3, c3 = fc.parse(txt, len(q["options"]))
        old_text, _ = cache.run_argument(r)
        old_id = r.get("arg_id") or (fc.arg_id(old_text) if old_text is not None else None)
        fc.append("l3_repair.jsonl", {"phase": ph, "model": r["model"], "qid": r["qid"], "target": r["target"], "arg_id": fc.arg_id(arg),
                                      "old_arg_id": old_id, "history_id": history_id(r),
                                      "old_round3": R[3], "round3": {"k": 3, "answer": a3, "conf": c3, "raw": txt}})
    with ThreadPoolExecutor(workers) as ex:
        failed = 0
        for i, f in enumerate(as_completed([ex.submit(w, *j) for j in jobs]), 1):
            try: f.result()
            except Exception as e: failed += 1; print("ERR", e, file=sys.stderr)
            if i % 25 == 0: print(i, flush=True)
    if failed:
        sys.exit(f"L3 repair incomplete: {failed}/{len(jobs)} calls failed; rerun to resume")

if __name__ == "__main__":
    main()
