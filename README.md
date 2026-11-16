# FlipCurve

How often do LLMs adopt correct and incorrect suggestions under the same escalating pushback?

Submission for the Apart Research AI x Epistemics Research Sprint (Nov 13-15, 2026), Track 1: Model Epistemics.

All code in this repository was written during the sprint (Nov 13-15, 2026); there is no preexisting code.

## Design

- MMLU-Pro multiple-choice items. Round 0: answer + confidence (0-100).
- Initially correct -> user pushes toward a wrong option; adopting that target is **capitulation**, C(k).
- Initially wrong -> user pushes toward the gold answer; adopting it is **uptake of a correct suggestion**, W(k).
- Identical cumulative 3-step ladder in both directions: L1 doubt, L2 claimed expertise, L3 the previous context plus an argument. Argument persuasiveness was not independently measured or matched. The judge audit checks target alignment; a separate check rejects forbidden answer/confidence fields.
- Neutral control ("please re-check") measures disagreement with the conversation's round-0 answer among valid follow-ups, pooling L1-L3. It is not an adjacent-turn switching rate. A separate matched-pair analysis compares target uptake at L3.
- Metrics: W(k), C(k), uptake gap D(k)=W-C, round-0 ECE, and confidence change when holding a correct answer. D describes asymmetric uptake under this protocol; it does not measure evidence sensitivity.

## Phases

- Phase 1 (`questions.jsonl`, `runs_<model>.jsonl`): 196 sampled items stratified across 14 categories; 193 had the gold and at least one decoy argument available. Three items were omitted for incomplete argument coverage, not benchmark invalidity. Historical generation attempted the gold answer plus three decoys using `deepseek-v4-flash`; the experiment selected the first available decoy from the completed cache. Pushback and neutral conversations had separate round-0 calls.
- Phase 2 (`hard/`): 300 new items from six higher-error categories (history, philosophy, engineering, law, business, health) to enlarge the initially-wrong sample. A frozen manifest selects all initially wrong screens and an equal-size seeded sample of initially correct screens per model. Original L3 arguments are recorded as `gen: hark`. The author confirms that Hark was based on Claude Opus 5.5, a base model distinct from the evaluated set. The records do not independently verify that attribution, and no immutable model revision, historical prompt or generation settings were saved, so the original execution cannot be independently reconstructed.
- The released runner implements a revised future-run protocol: a seeded decoy is chosen before generation, only the gold and that decoy are generated, arguments are audited before use, and Phase 2 stages run sequentially. The Phase 2 generator now defaults to `gemma-4-26b-a4b-it`; it does not reconstruct the historical Hark process. Rebuilding the report uses the fixed released data and makes no model-provider calls.

## Usage

To rebuild only the released English report, install the pinned analysis dependencies in your chosen Python environment, plus Pandoc 3.9 and LibreOffice:

```bash
pip install -r requirements-report.txt
bash report/build.sh
```

The build accepts `PYTHON`, `PANDOC` and `SOFFICE` executable overrides and makes no model-provider calls. It regenerates the figures, tables and substituted Markdown, formats the DOCX using the [official template](report/template/README.md), and exports that same document to PDF. Old Standard TT fonts are bundled under their OFL license; the PDF export checks font embedding and the eight-page main-text limit. When run with Codex's bundled Python, the exporter uses the matching bundled LibreOffice. The Chinese reading edition is maintained separately.

The data-collection commands below call the provider and can append to data files. Use a separate copy for new experiments; the final two commands analyse the released records locally.

```
pip install adaption datasets pandas matplotlib numpy statsmodels
export ADAPTION_API_KEY=...
# Phase 1 (from flipcurve/). Default --models is the reported Phase 1 lineup:
#   gemma-4-31b-it qwen3.8-27b qwen3.5-397b glm-5.3-flash glm-5.2
# (DeepSeek-V4-Flash wrote the Phase 1 arguments and is refused as an evaluated model.)
# --mock does an offline dry run inside ./mock_run/ and never touches the tracked data files.
python flipcurve.py prep
python flipcurve.py args                        # dedicated argument-writing prompt, format-checked
python argfix.py judge --all && python argfix.py regen --all   # audit; `run` uses ONLY audited arguments
python flipcurve.py run                         # writes runs_<model>.jsonl; exits non-zero on any failure
                                                # or if a question lacks audited arguments (--allow-partial)
# Phase 2 (strictly sequential; from flipcurve/hard/)
./run_hard.sh screen   # rerun to retry failed calls and invalid answers
./run_hard.sh freeze   # writes manifest.json; refuses incomplete/empty screens (hard.py freeze --allow-incomplete)
./run_hard.sh args     # cohort arguments + audit
./run_hard.sh push     # refuses to run without manifest.json
# Analysis (from flipcurve/)
bash report/build.sh   # canonical report: repaired L3 turns and frozen Phase 2 cohort
python bootstrap.py    # question-cluster bootstrap for pooled W/C/D
```
`flipcurve.py analyze` is an alias for `analyze_all.py`. The report build requires `PHASE2=cohort` and `REPAIR=1`; it rejects alternative modes to avoid mixing main-analysis text with sensitivity estimates. Individual analysis scripts accept `PHASE2=all` or `REPAIR=0` for exploratory sensitivity runs. Headline and selected supporting estimates are substituted from `report_numbers.json` through `{{key}}` placeholders; some fixed protocol counts and descriptive text remain in the report source. `report/build.sh` regenerates the canonical tables, figures and report exports.

## Scoring

`flipcurve.parse` reads the final answer field and checks the letter against the item's options. `analyze_all.py` rescores stored responses with it. New pushback runs record `arg_id`, a fingerprint of the argument text, which the analysis uses to find the corresponding audit or replacement. For older runs, the analysis reconstructs argument provenance from the released caches and repair records. Legacy approvals without fingerprints assume the original cache text. Run `argfix.py judge --all` after changing cached arguments. Arguments that fail the format check are excluded even if the judge passed them.

Baseline accuracy and ECE use the first strictly valid round-0 answer per model/question from the raw Phase 1 conversation logs and Phase 2 screens, before L3 argument exclusions. Records are read in deterministic file and record order. Accuracy uses all selected valid answers; ECE requires reported confidence. Main-table W/C/D and supplemental baseline metrics use their explicitly stated populations. Ladder-analysis question-cluster bootstraps use 2,000 resamples; the branched argument test uses 5,000 resamples stratified by phase, retaining both branches and all models within each sampled question. Illustrative break-even intervals condition on fixed observed baseline accuracy.

## Data provenance and the argument audit

An external review found that the original Phase 1 arguments were generated with the answering system prompt, so 68 of 222 used arguments did not argue for their target. `arg_audit.jsonl` holds a judge label for all 583 used arguments, `arguments_v2.jsonl` the regenerated replacements, and `l3_repair.jsonl` the re-run L3 turns (original turns kept under `old_round3`). Phase 1 decoy targets in the original runs were the first wrong-option argument in `arguments.jsonl`; the recorded `target` is authoritative for scoring. Phase 2 screening is incomplete (1,632/1,800 records); `hard/manifest.json` is the frozen cohort.

Models (open-weight, via Adaption Responses API): gemma-4-31b-it, qwen3.8-27b, qwen3.5-397b, glm-5.3-flash, glm-5.2, deepseek-v4-flash (phase 2 only).

See [methods and reproducibility notes](report/methods-notes.md) for known settings, missing provider metadata and historical/current protocol differences. See [answer-key review notes](report/answer-key-notes.md) for the historical five-item exploratory exclusion; the five flags are not five established key errors, and the original benchmark keys remain the primary analysis.
## Paired argument experiment (suggestion vs. suggestion + argument)
Each of the 1,212 analysed pushback conversations is forked from its stored round 0 into two independent
single-reply branches: A (suggestion only) and B (same suggestion + the audited argument). Same SYSTEM, temperature 0.
```
python pair_exp.py build                 # pair/pairs.jsonl from the analysis cohort
python pair_exp.py run --workers 4       # resumable; dedup on (pair_id, branch); --stream for gateway 504s
python pair_exp.py status
python pair_analyze.py                   # pair/results.md, pair/results.json
```
Raw replies: `pair/raw_responses.jsonl` (first reply per pair/branch is analysed); errors: `pair/errors.jsonl`;
model-ID map: `pair/model_map.json`. The A/B contrast changes both the argument content and the prompt length.
