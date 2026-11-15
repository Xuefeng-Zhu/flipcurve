"""Apply reproducible page, table and caption formatting to the Pandoc Word export."""
import sys
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def child(parent, tag):
    element = parent.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        parent.append(element)
    return element


def format_document(path):
    doc = Document(path)
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin = section.bottom_margin = Inches(.9)
        section.left_margin = section.right_margin = Inches(.9)
    for name in ('Normal', 'Body Text', 'First Paragraph', 'Compact'):
        style = doc.styles[name]
        style.font.name, style.font.size = 'Times New Roman', Pt(10.5)
        style.paragraph_format.line_spacing = 1.05
        style.paragraph_format.widow_control = True
    for style in doc.styles:
        if style.name in ('Title', 'Subtitle', 'Author') or style.name.startswith('Heading'):
            style.font.name = 'Times New Roman'
            style.font.color.rgb = RGBColor(0, 0, 0)
    doc.styles['Title'].font.size = Pt(19)
    for style in doc.styles:
        if style.name.lower() in ('caption', 'table caption', 'image caption'):
            style.font.name = 'Times New Roman'
            style.font.size = Pt(9.5)
    figure_number = table_number = 0
    for paragraph in doc.paragraphs:
        if paragraph.style.name == 'Captioned Figure':
            paragraph.paragraph_format.keep_with_next = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif paragraph.style.name in ('Table Caption', 'Image Caption'):
            if paragraph.style.name == 'Image Caption':
                figure_number += 1
                prefix = f'Figure {figure_number}: '
            else:
                table_number += 1
                prefix = f'Table {table_number}: '
                paragraph.paragraph_format.keep_with_next = True
            first = paragraph.runs[0]._r if paragraph.runs else None
            run = paragraph.add_run(prefix)
            if first is not None:
                first.addprevious(run._r)
        elif paragraph.text.startswith(('Table A1.', 'Table B1.')):
            paragraph.paragraph_format.keep_with_next = True
    for table in doc.tables:
        table.autofit = False
        n = len(table.columns)
        widths = ([1.45, 2.625, 2.625] if n == 3 else [1.85] + [4.85/(n-1)]*(n-1))
        for col, width in zip(table.columns, widths):
            col.width = Inches(width)
        props = table._tbl.tblPr
        borders = child(props, 'w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            element = child(borders, f'w:{edge}')
            for key, value in {'val': 'single', 'sz': '4', 'color': 'D9D9D9'}.items():
                element.set(qn(f'w:{key}'), value)
        for i, row in enumerate(table.rows):
            child(row._tr.get_or_add_trPr(), 'w:cantSplit')
            if i == 0:
                child(row._tr.get_or_add_trPr(), 'w:tblHeader')
            for j, cell in enumerate(row.cells):
                cell.width = Inches(widths[j])
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                props = cell._tc.get_or_add_tcPr()
                margins = child(props, 'w:tcMar')
                for edge in ('top', 'bottom', 'left', 'right'):
                    element = child(margins, f'w:{edge}')
                    element.set(qn('w:w'), '70')
                    element.set(qn('w:type'), 'dxa')
                if i == 0:
                    child(props, 'w:shd').set(qn('w:fill'), 'EEEEEE')
                for paragraph in cell.paragraphs:
                    # All report tables are compact: keep their rows together, with
                    # the preceding caption, instead of leaving an orphan final row.
                    paragraph.paragraph_format.keep_with_next = i < len(table.rows) - 1
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.name, run.font.size = 'Times New Roman', Pt(9.5)
                        if i == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(0, 0, 0)
    doc.save(path)


if __name__ == '__main__':
    format_document(sys.argv[1])
