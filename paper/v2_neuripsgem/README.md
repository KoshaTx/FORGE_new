# FORGE, GEM workshop (NeurIPS 2026) short paper, v2

`FORGE_GEM2026_paper.tex` is a condensation of `paper/v1` for the GEM workshop short-paper track.
Build it in place:

```bash
cd paper/v2_neuripsgem && latexmk -pdf FORGE_GEM2026_paper.tex
```

The leading `v2` numbers **this workshop paper**, not the manuscript. The body is still a
condensation of `paper/v1`; there is no `paper/v2`.

## What changed from `v1_neuripsgem`

| Change | Detail |
| --- | --- |
| Abstract | Replaced in full, author-supplied, reframed around reaction chemistry as a coordinate system |
| Introduction | Replaced in full, author-supplied, three paragraphs ending in a threefold contributions statement |
| FORGE expansion | First mention in both abstract and introduction now reads FORGE (**F**low-matched, **O**pen-ended, **R**oute-resolved **G**eneration and **E**xploration), initials bolded |
| Bibliography | One new entry, `maganti2026synthesis`, added to `../v1/references.bib` |
| Everything else | Byte-identical to `v1_neuripsgem` |

Two substantive differences follow from the new text and are deliberate, not oversights:

1. The abstract no longer quantifies the improvement over post-hoc filtering. Where v1 gave
   `\ForgeNullUgiDifferencePP`, `\ForgeNullBLDifferencePP` and `\ForgeNullLXDifferencePP`
   percentage points, v2 says "substantially outperforming". The quantities remain in the body and
   in Appendix C.
2. The route sentence no longer carries the "bounded, route-blinded Ugi shortlist" scoping. It still
   says "admitted", and the fail-closed reading ("unresolved or search-censored") is unchanged.

The introduction now carries 22 citations against v1's 13, adding `qin2025defog`, `lee2025genmol`,
`ou2024deep` and `maganti2026synthesis`, and restoring `gao2020synthesizability`. The bibliography
is 27 entries. Every key resolves; the final pdflatex pass reports no undefined citation or
reference.

### Open items

- **`maganti2026synthesis` is not fully verified.** The entry was reconstructed from
  `docs/provenance/REFERENCE_AUDIT.md` row 10 and the corresponding row of
  `docs/provenance/NATURE_BIOTECH_DRAFTING_ANALYSIS.md`. The OpenReview record at
  <https://openreview.net/forum?id=6RFQqfjD06> could not be retrieved anonymously, so the exact
  title string and the full author list are unconfirmed. A comment in `../v1/references.bib` marks
  this. Confirm both against the published version before submission.
- **`rekesh2025syncogen` renders as 2025.** The prose this introduction was drafted from cites
  "Rekesh et al., 2026". The bib entry carries `year = {2025}` (arXiv:2507.11818), so the rendered
  citation reads 2025. Update the entry if the intended reference is a later published version.

## Venue contract

Taken from <https://www.gembio.ai/> on 26 August 2026.

| Requirement | Status |
| --- | --- |
| Up to 5 pages, excluding references and appendix | **NOT MET.** Body runs to page 6; Figure 1, its caption and the Discussion sit past the limit, and references begin partway down page 6 |
| NeurIPS 2026 main-conference template, GEM style file substituted | `\usepackage{GEM_workshop_2026}` |
| Anonymous, one round of double-blind review | Style file's default branch; `\iclrfinalcopy` is not called |
| Maximum 50 MB | 1.5 MB |
| Deadline 30 August 2026, 11:59 PM AoE, via OpenReview | Not submitted by this repository |

The longer abstract and introduction added roughly twenty-three lines over `v1_neuripsgem`, which
built to exactly five body pages. Recovering them is an open editorial decision, deliberately
deferred rather than made here: the candidates are moving Figure 1 to the appendix, shrinking it
below 0.52 textwidth, or condensing the Results paragraphs. Shrinking Figure 1 to 0.52 textwidth
was measured and does not on its own bring the body back to five pages.

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

The abstract's route counts are written as `\ForgeRouteCompleteProducts` and `\ForgeRouteAdmitted`
rather than as literals, so they track the artifact. Its three exact-L1 yields are typed verbatim
from the v1 body, matching `../v1/generated/production_comparison_rows.tex`.

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
