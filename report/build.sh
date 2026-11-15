#!/bin/bash
# Rebuild figures, tables and the report PDF/DOCX. Run from flipcurve/.
set -e
python analyze_all.py > analysis_all.txt
python analyze_extra.py > analysis_extra.txt
python conf_match.py > analysis_conf.txt
python report_stats.py > analysis_stats.txt
python report_numbers.py > /dev/null
cd report
python -c "
import json, re
s=open('report.md').read().replace('TABLE_MAIN',open('../tab_main.md').read()).replace('TABLE_CONF',open('../tab_conf.md').read()).replace('TABLE_PHASE',open('../tab_phase.md').read())
N=json.load(open('../report_numbers.json'))
s=re.sub(r'\{\{(\w+)\}\}', lambda m: N[m.group(1)], s)
assert '{{' not in s
open('_build.md','w').write(s)"
pandoc _build.md -o FlipCurve_report.pdf --pdf-engine=pdflatex --resource-path=.:..
pandoc _build.md -o FlipCurve_report.docx --resource-path=.:..
echo built
