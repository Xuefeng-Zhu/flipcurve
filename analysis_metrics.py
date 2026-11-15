"""Shared baseline and neutral metrics, independent of pushback/audit exclusions."""
import glob
import json
import math

from flipcurve import parse


def load_baselines(questions, phase1_paths=None, phase2_paths=None):
    """First valid raw answer per (model, qid), in sorted-file then record order.

    Phase 1 uses raw conversation round 0; Phase 2 uses raw screen records.
    Missing confidence does not exclude a valid answer from baseline accuracy.
    Argument approval, follow-up completion, and the frozen pushback manifest
    deliberately do not enter this baseline population.
    """
    if phase1_paths is None:
        phase1_paths = glob.glob("runs_*.jsonl")
    if phase2_paths is None:
        phase2_paths = glob.glob("hard/screen_*.jsonl")
    seen, rows = set(), []
    for phase, paths in ((1, phase1_paths), (2, phase2_paths)):
        for path in sorted(paths):
            with open(path) as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = record["model"], record["qid"]
                    if key in seen:
                        continue
                    raw = record["rounds"][0].get("raw") if phase == 1 else record.get("raw")
                    answer, confidence = parse(raw, len(questions[record["qid"]]["options"]))
                    if answer is None:
                        continue
                    seen.add(key)
                    rows.append(dict(model=record["model"], qid=record["qid"], phase=phase,
                                     answer=answer, ok=answer == record["gold"], conf=confidence))
    return rows


def baseline_ece(rows, bins=10):
    """ECE over confidence-bearing baselines only; accuracy uses all baselines."""
    eligible = [r for r in rows if r["conf"] is not None and math.isfinite(r["conf"])]
    if not eligible:
        return float("nan")
    groups = [[] for _ in range(bins)]
    for row in eligible:
        confidence = min(100, max(0, row["conf"])) / 100
        groups[min(int(confidence * bins), bins - 1)].append((row["ok"], confidence))
    return sum(abs(sum(ok - confidence for ok, confidence in group)) for group in groups) / len(eligible)


def valid_disagreement_rate(answer_pairs):
    """Fraction of valid follow-up answers that differ from a valid round 0."""
    valid = [(initial, followup) for initial, followup in answer_pairs
             if isinstance(initial, str) and isinstance(followup, str)]
    return sum(initial != followup for initial, followup in valid) / len(valid) if valid else float("nan")


def fit_pressure_model(followups):
    """Shared L3 robustness model with within-phase standardized argument length."""
    import statsmodels.formula.api as smf
    data = followups.copy()
    data["zlen"] = data.groupby("phase").arglen.transform(lambda s: (s - s.mean()) / s.std())
    data["y"] = data.on_target.astype(int)
    data["right0"] = data.correct0.astype(int)
    data = data.dropna(subset=["zlen"])
    return smf.logit("y ~ C(model) + right0*C(phase) + zlen", data=data).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": data.qid})
