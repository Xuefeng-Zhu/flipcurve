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
# Keep table rows together when Word paginates the report.
python - <<'PY'
from pathlib import Path
from xml.dom import minidom
from zipfile import ZipFile

docx = Path('FlipCurve_report.docx')
temp = docx.with_suffix('.tmp')
with ZipFile(docx) as source, ZipFile(temp, 'w') as output:
    for entry in source.infolist():
        content = source.read(entry)
        if entry.filename == 'word/document.xml':
            xml = minidom.parseString(content)
            for row in xml.getElementsByTagName('w:tr'):
                props = next((child for child in row.childNodes if child.nodeName == 'w:trPr'), None)
                if props is None:
                    props = row.insertBefore(xml.createElement('w:trPr'), row.firstChild)
                if not props.getElementsByTagName('w:cantSplit'):
                    props.insertBefore(xml.createElement('w:cantSplit'), props.firstChild)
            content = xml.toxml(encoding='utf-8')
        output.writestr(entry, content)
temp.replace(docx)
PY
echo built
