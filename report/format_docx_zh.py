"""Chinese typography and layout for the Pandoc translation export."""
import sys
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor


def child(parent, tag):
    item = parent.find(qn(tag))
    if item is None:
        item = OxmlElement(tag); parent.append(item)
    return item


def font(properties, east_asia='Songti SC'):
    fonts = child(properties, 'w:rFonts')
    for key, value in [('ascii','Times New Roman'), ('hAnsi','Times New Roman'),
                       ('eastAsia',east_asia), ('cs','Times New Roman')]:
        fonts.set(qn('w:'+key), value)
    child(properties,'w:lang').set(qn('w:eastAsia'),'zh-CN')


def format_document(path):
    doc = Document(path)
    for section in doc.sections:
        section.page_width, section.page_height = Mm(210), Mm(297)
        section.left_margin = section.right_margin = Mm(20)
        section.top_margin = section.bottom_margin = Mm(19)
        section.footer_distance = Mm(9)
        p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(); field = OxmlElement('w:fldSimple')
        field.set(qn('w:instr'),'PAGE'); run._r.addnext(field)
    for style in doc.styles:
        if style.type != 1 and style.type != 2:
            continue
        heading = style.name.startswith('Heading') or style.name == 'Title'
        font(style.element.get_or_add_rPr(), 'Heiti SC' if heading else 'Songti SC')
        style.font.name = 'Times New Roman'
        style.font.color.rgb = RGBColor(0,0,0)
        if style.type == 1:
            style.paragraph_format.widow_control = True
    for name in ('Normal','Body Text','First Paragraph','Compact'):
        if name in doc.styles:
            st=doc.styles[name]; st.font.size=Pt(10.5)
            st.paragraph_format.line_spacing=1.2
            st.paragraph_format.space_after=Pt(5)
    doc.styles['Title'].font.size=Pt(20)
    doc.styles['Title'].paragraph_format.space_after=Pt(8)
    for name,size in [('Heading 1',14),('Heading 2',12),('Author',11)]:
        if name in doc.styles:
            doc.styles[name].font.size=Pt(size)
    for name in ('Caption','Table Caption','Image Caption','Footnote Text'):
        if name in doc.styles:
            doc.styles[name].font.size=Pt(9)
    fig_num=table_num=0
    for p in doc.paragraphs:
        for run in p.runs:
            font(run._r.get_or_add_rPr(), 'Heiti SC' if p.style.name.startswith('Heading') or p.style.name=='Title' else 'Songti SC')
        if p.style.name == 'Captioned Figure':
            p.paragraph_format.keep_with_next=True
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif p.style.name in ('Image Caption','Table Caption'):
            if p.style.name=='Image Caption':
                fig_num+=1; prefix=f'图 {fig_num}：'
            else:
                table_num+=1; prefix=f'表 {table_num}：'; p.paragraph_format.keep_with_next=True
            first=p.runs[0]._r if p.runs else None
            run=p.add_run(prefix)
            if first is not None: first.addprevious(run._r)
        if p.text.startswith(('表 A1','表 B1','我们的主要贡献','本研究的主要贡献')):
            p.paragraph_format.keep_with_next=True
    for table in doc.tables:
        table.autofit=False
        count=len(table.columns)
        widths=[1.42,2.62,2.62] if count==3 else [1.5]+[5.16/(count-1)]*(count-1)
        for column,width in zip(table.columns,widths): column.width=Inches(width)
        borders=child(table._tbl.tblPr,'w:tblBorders')
        for edge in ('top','left','bottom','right','insideH','insideV'):
            border=child(borders,'w:'+edge)
            for key,value in [('val','single'),('sz','4'),('color','D9D9D9')]: border.set(qn('w:'+key),value)
        for i,row in enumerate(table.rows):
            props=row._tr.get_or_add_trPr(); child(props,'w:cantSplit')
            if i==0: child(props,'w:tblHeader')
            for j,cell in enumerate(row.cells):
                cell.width=Inches(widths[j]); cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER
                props=cell._tc.get_or_add_tcPr()
                if i==0: child(props,'w:shd').set(qn('w:fill'),'EEEEEE')
                margins=child(props,'w:tcMar')
                for edge in ('top','bottom','left','right'):
                    element=child(margins,'w:'+edge); element.set(qn('w:w'),'65');element.set(qn('w:type'),'dxa')
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next=i<len(table.rows)-1
                    p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
                    p.paragraph_format.line_spacing=1.05
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        font(run._r.get_or_add_rPr());run.font.size=Pt(9)
                        if i==0:run.bold=True
    for part in doc.part.package.parts:
        if str(part.partname)=='/word/footnotes.xml':
            root=parse_xml(part.blob)
            for run in root.iter(qn('w:r')): font(child(run,'w:rPr'))
            from lxml import etree
            part._blob=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    doc.save(path)


if __name__=='__main__':
    format_document(sys.argv[1])
