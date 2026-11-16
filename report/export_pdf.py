"""Export the formatted DOCX with its bundled fonts and enforce the page limit."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape

from docx import Document
from pypdf import PdfReader


def soffice_path():
    configured = os.environ.get('SOFFICE')
    if configured:
        path = shutil.which(configured)
        if not path:
            raise RuntimeError(f'SOFFICE executable not found: {configured}')
        return Path(path)
    # Codex's Python and LibreOffice belong to the same bundled runtime.
    dependencies = Path(sys.executable).absolute().parents[2]
    bundled = dependencies / 'bin' / 'override' / 'soffice'
    if dependencies.name == 'dependencies':
        if not bundled.exists():
            raise RuntimeError(f'Bundled LibreOffice is missing: {bundled}')
        return bundled
    path = shutil.which('soffice')
    if not path:
        raise RuntimeError('Install LibreOffice or set SOFFICE to its executable.')
    return Path(path)


def main(docx_path):
    docx_path = Path(docx_path).resolve()
    executable = soffice_path()
    font_directory = Path(__file__).resolve().parent / 'fonts'
    for name in ('Regular', 'Bold', 'Italic'):
        if not (font_directory / f'OldStandard-{name}.ttf').is_file():
            raise RuntimeError('The bundled Old Standard TT fonts are missing.')
    # A private profile/cache avoids touching the user's LibreOffice settings.
    temporary_base = '/private/tmp' if sys.platform == 'darwin' else None
    with tempfile.TemporaryDirectory(prefix='flipcurve-pdf-', dir=temporary_base) as temporary:
        root = Path(temporary)
        font_cache = root / 'font-cache'
        font_cache.mkdir()
        config = root / 'fonts.conf'
        directories = [font_directory]
        dependencies = Path(sys.executable).absolute().parents[2]
        bundled_fonts = dependencies / 'native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype'
        if bundled_fonts.is_dir():
            directories.append(bundled_fonts)
        # Old Standard TT does not cover every mathematical symbol. Allow the
        # platform's installed fallback fonts when outside the bundled runtime.
        if not bundled_fonts.is_dir():
            for directory in ('/usr/share/fonts', '/usr/local/share/fonts',
                              '/System/Library/Fonts', '/Library/Fonts'):
                if Path(directory).is_dir():
                    directories.append(Path(directory))
        config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
                          '<fontconfig>' + ''.join(f'<dir>{escape(str(p))}</dir>' for p in directories)
                          + f'<cachedir>{escape(str(font_cache))}</cachedir></fontconfig>')
        env = os.environ.copy()
        env['FONTCONFIG_FILE'] = str(config)
        env['XDG_CACHE_HOME'] = str(root / 'cache')
        env['XDG_CONFIG_HOME'] = str(root / 'config')
        if temporary_base:
            env['TMPDIR'] = temporary_base
        subprocess.run([str(executable), f'-env:UserInstallation={(root / "profile").as_uri()}',
                        '--headless', '--invisible', '--norestore', '--convert-to', 'pdf',
                        '--outdir', str(root), str(docx_path)], env=env, check=True, timeout=180)
        pdf_path = root / docx_path.with_suffix('.pdf').name
        reader = PdfReader(pdf_path)
        extracted = '\n'.join(page.extract_text() for page in reader.pages)
        for page_number, page in enumerate(reader.pages, 1):
            letters = [0, 0]

            def count_font(text, _cm, _tm, font, _size):
                count = sum(character.isascii() and character.isalpha() for character in text)
                letters[0] += count
                if font and 'OldStandard' in str(font.get('/BaseFont', '')):
                    letters[1] += count

            page.extract_text(visitor_text=count_font)
            if letters[0] and letters[1] / letters[0] < 0.98:
                raise RuntimeError(f'Body font substitution detected on page {page_number}.')
            for reference in page['/Resources']['/Font'].values():
                font = reference.get_object()
                if 'OldStandard' not in str(font.get('/BaseFont', '')):
                    continue
                descendants = font.get('/DescendantFonts', [reference])
                for descendant in descendants:
                    descriptor = descendant.get_object().get('/FontDescriptor')
                    if not descriptor or not any(key in descriptor.get_object() for key in ('/FontFile', '/FontFile2', '/FontFile3')):
                        raise RuntimeError(f'Old Standard TT is not embedded on page {page_number}.')
        document = Document(docx_path)
        source_text = '\n'.join(p.text for p in document.paragraphs)
        source_text += '\n'.join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
        for symbol in 'Δ∈≥≤×→−':
            if symbol in source_text and symbol not in extracted:
                raise RuntimeError(f'Missing mathematical symbol in PDF: {symbol}')
        main_pages = None
        for number, page in enumerate(reader.pages, 1):
            match = re.search(r'(?m)^References\s*$', page.extract_text())
            if match:
                preceding = page.extract_text()[:match.start()].strip()
                main_pages = number - (not preceding or preceding == str(number))
                break
        if main_pages is None or main_pages > 8:
            raise RuntimeError(f'Main report exceeds the 8-page limit or has no References heading: {main_pages}')
        final_path = docx_path.with_suffix('.pdf')
        shutil.copyfile(pdf_path, final_path)
        print(f'Exported {final_path.name}: {len(reader.pages)} total pages, main text within {main_pages} pages; Old Standard TT embedded.')


if __name__ == '__main__':
    main(sys.argv[1])
