# Official report template

`official-template.docx` was exported on November 16, 2026 from the
[template linked by the sprint organizers](https://docs.google.com/document/d/17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs/edit).
It is the formatting reference and source of the embedded Word fonts, not
inserted as report content. Pandoc's own style container is retained because
the Google Docs export lacks several styles that Pandoc requires for captions,
tables, and the author block.

The visible template uses US Letter paper, one-inch margins, Old Standard TT
at 11 points for body text, 20-point bold title text, and 14-point bold main
headings. Several of these settings are direct formatting in the template
rather than named styles, so `format_docx.py` applies them explicitly. It also
retains the title rules, centered author block, indented abstract, and sprint
footnote. The PDF is exported from that formatted DOCX, so both deliverables
share the same layout.

Old Standard TT Regular, Bold, and Italic are bundled in `../fonts/` under
the SIL Open Font License 1.1 (`../fonts/OFL.txt`). They were downloaded from
[Google Fonts](https://github.com/google/fonts/tree/main/ofl/oldstandardtt).
The isolated PDF export registers those fonts without changing system fonts
or the user's LibreOffice profile. The export fails if the font is substituted
or the main text exceeds eight pages.
