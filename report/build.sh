#!/bin/bash
# Rebuild figures, tables and the report PDF/DOCX. Run from flipcurve/.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"
PANDOC_BIN="${PANDOC:-pandoc}"
# Sensitivity analyses can be run separately, but their populations must not be mixed
# with the fixed protocol and sampling descriptions in the main report.
if [[ "${PHASE2:-cohort}" != cohort || "${REPAIR:-1}" != 1 ]]; then
    echo "The report build requires PHASE2=cohort and REPAIR=1. Run analysis scripts separately for sensitivity analyses." >&2
    exit 1
fi
"$PYTHON_BIN" analyze_all.py > analysis_all.txt
"$PYTHON_BIN" analyze_extra.py > analysis_extra.txt
"$PYTHON_BIN" conf_match.py > analysis_conf.txt
"$PYTHON_BIN" report_stats.py > analysis_stats.txt
"$PYTHON_BIN" pair_analyze.py > /dev/null
"$PYTHON_BIN" report_numbers.py > /dev/null
"$PYTHON_BIN" user_implications.py > analysis_user.txt
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
for name in ('analysis_all.txt', 'analysis_extra.txt', 'analysis_conf.txt', 'analysis_stats.txt'):
    path = Path(name)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
PY
cd report
"$PYTHON_BIN" -c "
import json, re
s=open('report.md').read()
for marker, filename in [('TABLE_MAIN', 'tab_main.md'), ('TABLE_BASELINE', 'tab_baseline.md'), ('TABLE_CONF', 'tab_conf.md'), ('TABLE_PHASE', 'tab_phase.md')]:
    s=s.replace(marker,open('../'+filename).read())
N=json.load(open('../report_numbers.json')); N.update(json.load(open('../user_numbers.json')))
s=re.sub(r'\{\{(\w+)\}\}', lambda m: N[m.group(1)], s)
assert '{{' not in s and not re.search(r'TABLE_[A-Z]+', s)
open('_build.md','w').write(s)"
"$PANDOC_BIN" _build.md -o FlipCurve_report.docx --resource-path=.:..
"$PYTHON_BIN" format_docx.py FlipCurve_report.docx
"$PYTHON_BIN" export_pdf.py FlipCurve_report.docx
echo built
