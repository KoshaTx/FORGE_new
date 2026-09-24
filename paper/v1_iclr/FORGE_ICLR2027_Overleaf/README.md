# FORGE, ICLR 2027

Build `main.tex` with pdfLaTeX and BibTeX, or run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This standalone export uses the official ICLR 2027 template in anonymous review mode.
Its source is `paper/v1_iclr/FORGE_ICLR2027_paper.tex`. Figures, tables, experimental
images, bibliography databases and required styles are included. Historical numerical
tables are unchanged. Mathematical corrections and implementation-provenance limits
are documented in `REVISION_NOTES.md` and `mathematical_review.json`.

The main experimental figure includes the original mouse image, the three supplied
lipid structures and the four reported ROI signals. This experimental extension
remains a working draft pending the protocol, replication and units information
listed in `experimental/in_vivo/evidence_record.json`. No new experiments were run.

Page-budget editing is deferred. The AI-use statement records author-declared assistance.
The ethics statement records approval by Children's Hospital of Philadelphia; the protocol
number is omitted at the author's request. Appendix A.8 describes historical source provenance
and access limitations of the standalone manuscript archive.

Molecular drawings are vector PDFs with editable SVG masters in `figures/chemical_drawings/`.
The saved structure receipts and `render.py` reproduce them with RDKit and rsvg-convert.
The atlas pairs generated lipids and canonical SMILES with exact L1 building blocks.
