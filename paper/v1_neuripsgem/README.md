# FORGE, GEM workshop (NeurIPS 2026) short paper

`FORGE_GEM2026_paper.tex` is a condensation of `paper/v1` for the GEM workshop short-paper track.
Build it in place:

```bash
cd paper/v1_neuripsgem && latexmk -pdf FORGE_GEM2026_paper.tex
```

## Venue contract

Taken from <https://www.gembio.ai/> on 26 August 2026.

| Requirement | Status |
| --- | --- |
| Up to 5 pages, excluding references and appendix | Body ends on page 5; references begin on page 6 |
| NeurIPS 2026 main-conference template, GEM style file substituted | `\usepackage{GEM_workshop_2026}` |
| Anonymous, one round of double-blind review | Style file's default branch; `\iclrfinalcopy` is not called |
| Maximum 50 MB | 1.5 MB |
| Deadline 30 August 2026, 11:59 PM AoE, via OpenReview | Not submitted by this repository |

`GEM_workshop_2026.sty` is the workshop's file, downloaded unmodified from the link on the call for
papers. Note that its running head reads "Under review at the GEM workshop, ICLR 2026": the workshop
appears to have carried the string over from the previous edition. It is left as shipped rather than
corrected, because every submission built from their file will carry the same head.

`NEURIPS_2026_reference_template.tex` is the unmodified NeurIPS 2026 skeleton, kept only so the
substitution the call for papers asks for can be checked against the original.

## Where the numbers come from

This build carries no numbers of its own. Every quantitative statement is either typed verbatim from
the `paper/v1` body or expanded from `../v1/generated/completed_evidence_macros.tex`, and every table
row is the same generated artifact `paper/v1` inputs. Figures resolve through
`\graphicspath{{./}{../v1/}}`. Nothing is copied into this directory, so this paper and the full
manuscript cannot drift apart: reissue an artifact and both builds move together.

Editing a numeric value in this file without reissuing its artifact breaks that correspondence and
must not be done.

## What moved where

The body keeps the abstract, a condensed introduction, the FORGE formulation including the
program-masked training objective, four result paragraphs and a short discussion, with two tables and
one figure. Everything else is appendix, which the venue does not count:

| Appendix | Source in `paper/v1` |
| --- | --- |
| A Preliminaries | `\section{Preliminaries}` |
| B FORGE in full | `\section{FORGE}` |
| C Extended experimental protocol and results | `\section{Results}` |
| D Extended computational reporting tables | appendix section of the same name |
| E Theoretical properties | appendix section of the same name |
| F Generated-structure atlas | appendix section of the same name |

The appendix was assembled by slicing those section ranges out of
`paper/v1/FORGE_ICLR2027_paper.tex` rather than by retyping, so appendix prose is byte-identical to
the manuscript apart from three mechanical changes:

1. `\input{generated/...}` and `\input{figures/...}` were retargeted to `../v1/...`.
2. The two floats the body now owns (`tab:ugi-common-benchmark`, `fig:forge-generated-samples`) were
   dropped from the appendix so their labels are not multiply defined. A comment marks each site.
3. Three appendix tables and the overview diagram were widened or stepped down one type size. They
   were sized against the v1 build, which loads `times`; neither the NeurIPS template nor the GEM
   style file does, and the wider default face pushed them past the text block.

To re-derive the appendix after the v1 manuscript changes, redo those slices; the section boundaries
are the `\section` lines listed above.
