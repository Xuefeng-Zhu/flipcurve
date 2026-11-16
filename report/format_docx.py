"""Apply reproducible page, table and caption formatting to the Pandoc Word export."""
import sys
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = 'Old Standard TT'  # official sprint template font (Letter, 1in margins)


def child(parent, tag):
    element = parent.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        parent.append(element)
    return element


def embed_template_fonts(path):
    """Carry the official template's embeddable fonts into the Word export."""
    with ZipFile(path) as archive:
        members = {info.filename: (info, archive.read(info)) for info in archive.infolist()}
    template = Path(__file__).resolve().parent / 'template' / 'official-template.docx'
    with ZipFile(template) as archive:
        for info in archive.infolist():
            if info.filename.startswith('word/fonts/') or info.filename in ('word/fontTable.xml', 'word/_rels/fontTable.xml.rels'):
                members[info.filename] = (info, archive.read(info))
    key = '[Content_Types].xml'
    root = etree.fromstring(members[key][1])
    ns = 'http://schemas.openxmlformats.org/package/2006/content-types'
    existing = {node.get('Extension') for node in root}
    for ext, content_type in [('ttf', 'application/x-font-ttf'),
                              ('odttf', 'application/vnd.openxmlformats-officedocument.obfuscatedFont')]:
        if ext not in existing and any(name.endswith('.' + ext) for name in members):
            etree.SubElement(root, '{'+ns+'}Default', Extension=ext, ContentType=content_type)
    members[key] = (members[key][0], etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True))
    temporary = Path(path).with_suffix('.tmp.docx')
    with ZipFile(temporary, 'w') as archive:
        for info, data in members.values():
            archive.writestr(info, data)
    temporary.replace(path)


def set_font(properties):
    fonts = child(properties, 'w:rFonts')
    for attribute in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        fonts.set(qn('w:'+attribute), FONT)
    for attribute in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'csTheme'):
        fonts.attrib.pop(qn('w:'+attribute), None)


def format_document(path):
    embed_template_fonts(path)
    doc = Document(path)
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin = section.bottom_margin = Inches(1.0)
        section.left_margin = section.right_margin = Inches(1.0)
        section.header_distance = section.footer_distance = Inches(0.5)
    for style in doc.styles:
        if style.type in (1, 2):
            set_font(style.element.get_or_add_rPr())
            style.font.color.rgb = RGBColor(0, 0, 0)
        if (style.name or '').lower() in ('normal', 'body text', 'first paragraph', 'compact'):
            style.font.size = Pt(11)
            style.paragraph_format.line_spacing = 1.15
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(0)
            style.paragraph_format.widow_control = True
    doc.styles['Title'].font.size = Pt(20)
    doc.styles['Title'].font.bold = True
    doc.styles['Title'].paragraph_format.line_spacing = 1
    doc.styles['Title'].paragraph_format.space_after = Pt(10)
    for style in doc.styles:
        if style.name and style.name.lower() == 'heading 1':
            style.font.size = Pt(14)
            style.font.bold = True
            style.paragraph_format.line_spacing = 1.5
            style.paragraph_format.space_before = Pt(10)
            style.paragraph_format.space_after = Pt(0)
            style.paragraph_format.keep_with_next = True
        elif style.name and style.name.lower() == 'heading 2':
            style.font.size = Pt(13)
            style.font.bold = True
            style.paragraph_format.space_before = Pt(10)
            style.paragraph_format.space_after = Pt(4)
        elif style.name == 'Author':
            style.font.size = Pt(11)
            style.font.bold = False
            style.paragraph_format.line_spacing = 1
            style.paragraph_format.space_after = Pt(10)
    for style in doc.styles:
        if style.name and style.name.lower() in ('caption', 'table caption', 'image caption'):
            style.font.name = FONT
            style.font.size = Pt(9.5)
    note = 'Research conducted at the AI x Epistemics Research Sprint, November 2026.'
    footnote_text = ' '.join(
        node.text or ''
        for part in doc.part.package.parts if str(part.partname) == '/word/footnotes.xml'
        for node in parse_xml(part.blob).iter(qn('w:t'))
    )
    if note not in footnote_text and not any(note in p.text for p in doc.paragraphs):
        for paragraph in doc.paragraphs:
            if paragraph.style.name == 'Author':
                new = OxmlElement('w:p'); paragraph._p.addnext(new)
                from docx.text.paragraph import Paragraph
                para = Paragraph(new, paragraph._parent)
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = para.add_run(note); run.italic = True; run.font.size = Pt(9); run.font.name = FONT
                break
    figure_number = table_number = 0
    in_abstract = False
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            set_font(run._r.get_or_add_rPr())
        if paragraph.style.name == 'Title':
            # The official template's two title rules are retained deliberately.
            borders = child(paragraph._p.get_or_add_pPr(), 'w:pBdr')
            for side, size in [('top', '24'), ('bottom', '8')]:
                border = child(borders, 'w:'+side)
                for key, value in {'val':'single', 'sz':size, 'space':'12', 'color':'000000'}.items():
                    border.set(qn('w:'+key), value)
            paragraph.paragraph_format.space_before = Pt(12)
            paragraph.paragraph_format.space_after = Pt(24)
        elif paragraph.style.name == 'Author':
            author, affiliation, collaboration = paragraph.text.split(' — ', 2)
            partner = collaboration.removeprefix('with ')
            paragraph.text = f'{author}\n{affiliation}\n\n'
            paragraph.add_run('With').bold = True
            paragraph.add_run(f'\n{partner}')
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if paragraph.text == 'Abstract':
            in_abstract = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.space_before = Pt(12)
            paragraph.paragraph_format.space_after = Pt(10)
        elif paragraph.style.name == 'Heading 1':
            in_abstract = False
        elif in_abstract:
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.left_indent = Inches(0.75)
            paragraph.paragraph_format.right_indent = Inches(0.75)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if paragraph.text == 'Our main contributions are:':
            paragraph.paragraph_format.keep_with_next = True
        elif paragraph.style.name == 'Captioned Figure':
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
        widths = ([1.4, 2.55, 2.55] if n == 3 else [1.8] + [4.7/(n-1)]*(n-1))
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
                        run.font.name, run.font.size = FONT, Pt(9.5)
                        if i == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(0, 0, 0)
    for part in doc.part.package.parts:
        if str(part.partname) == '/word/footnotes.xml':
            root = parse_xml(part.blob)
            for paragraph in root.iter(qn('w:p')):
                spacing = child(child(paragraph, 'w:pPr'), 'w:spacing')
                for key, value in [('line','240'), ('lineRule','auto'), ('after','200')]:
                    spacing.set(qn('w:'+key), value)
            for run in root.iter(qn('w:r')):
                properties = child(run, 'w:rPr')
                set_font(properties)
                child(properties, 'w:sz').set(qn('w:val'), '20')
            part._blob = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    doc.save(path)


if __name__ == '__main__':
    format_document(sys.argv[1])
