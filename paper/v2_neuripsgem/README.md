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
| Figure 1 | New framework overview added as Figure 1; all later figures shift by one. The sample-atlas figure moved from the body to Appendix F |
| Routing de-emphasis | The sparse-evidence route diagnostics are removed from the abstract, the main Results and Appendix C.7; Table 7 and the route-disposition figure are deleted |
| AGILE dependence | The "AGILE-type" qualifier is dropped in all four places it appeared; the Ugi chemistry itself is unchanged |
| Everything else | Byte-identical to `v1_neuripsgem` |

Two substantive differences follow from the new text and are deliberate, not oversights:

1. The abstract no longer quantifies the improvement over post-hoc filtering. Where v1 gave
   `\ForgeNullUgiDifferencePP`, `\ForgeNullBLDifferencePP` and `\ForgeNullLXDifferencePP`
   percentage points, v2 says "substantially outperforming". The quantities remain in the body and
   in Appendix C.
2. No route-completeness number appears anywhere in the manuscript. Recursive routing is presented
   as a downstream capability with an abstention contract, not as a scored result.

### Why the routing numbers were removed

The bounded cascade closed 19 of 255 admitted Ugi products. That figure measures coverage of one
deliberately sparse, frozen, method-blind evidence contract, and the manuscript itself recorded that
most unresolved components were absent from the bounded evidence index and were never expanded. Read
without that context it invites the conclusion that FORGE finds routes for about seven percent of its
designs, which is not what the experiment establishes.

An earlier route audit reported 2,515 of 2,590 prediction-supported Ugi designs, 97.1%, resolving to
purchasable starting material. The two are not comparable, and `docs/FORGE_IMPLEMENTATION_REFERENCE.md`
already warns that the 97.1% denominator is prediction-supported designs, "roughly 8.6% of admitted
ones". Reconciling them is a separate audit; until it runs, the manuscript reports no route
percentage. All thirteen route macros are now unused by this build.

The introduction now carries 22 citations against v1's 13, adding `qin2025defog`, `lee2025genmol`,
`ou2024deep` and `maganti2026synthesis`, and restoring `gao2020synthesizability`. The bibliography
is 27 entries. Every key resolves; the final pdflatex pass reports no undefined citation or
reference.

### Reference audit, 26 August 2026

All 27 cited references were checked against authoritative sources, not against memory.

- 19 entries carrying DOIs were resolved through the CrossRef API. Title, year and journal match
  the deposited record in every case.
- 7 entries without DOIs were resolved on arXiv. Titles match exactly and author counts match the
  bib entries: SynFlowNet 9, SynNet 3, RGFN 9, GenMol 9, Ou 5, DeFoG 4, SynCoGen 9.
- Venues were confirmed independently through DBLP and PMLR: SynFlowNet at ICLR 2025, SynNet at
  ICLR 2022, RGFN at NeurIPS 2024. PMLR volume 267 is confirmed as the 42nd ICML, 2025.
- The two page ranges most at risk of fabrication were checked digit by digit against PMLR and are
  exact: GenMol `lee25o` 33205--33226, DeFoG `qin25d` 50269--50326.

Two of the three fields that were initially unverified have since been resolved at the source:

- `koziarski2024rgfn` pages `46908--46955` are confirmed exactly against the official NeurIPS
  proceedings bibtex for hash `53704142f230054140418ecd8857f391`, together with volume 37, the
  publisher and the nine-author list. That record also supplied `doi = 10.52202/079017-1488`, now
  added to the entry.
- `ou2024deep` booktitle is confirmed by arXiv's own comment field on 2412.00928, which reads
  "NeurIPS 2024 Workshop on AI for New Drug Modalities" verbatim.

- `maganti2026synthesis` is now verified by reading the OpenReview record directly. It could not be
  retrieved programmatically: OpenReview answers the API with `ChallengeRequiredError` 403, a bot
  challenge, and neither Semantic Scholar nor DBLP indexes it. Accepted to the ICLR 2026 GEM
  workshop, submission 111, published 01 March 2026. The reconstruction had been wrong in two ways
  and both are corrected: the title is "Synthesis-constrained discrete diffusion for ionizable lipid
  generation", not "Ionizable-Lipid", and the authors are Rohin Maganti, Rahul Maganti and
  Mohamad-Gabriel Alameh, not "Maganti et al."

**All 27 references are verified. No field in this bibliography now rests on recall.**

One editorial point this raises. `maganti2026synthesis` is a self-citation, and the cited paper is
close to this one in subject: scaffold-conditioned Ugi lipid generation by an overlapping author
group. It is cited in the third person, which is the correct handling under double-blind review, but
a reviewer may still infer authorship from it. That is a normal and accepted cost of citing relevant
prior work, and it is noted here only so the choice is a deliberate one.

`rekesh2025syncogen` is correctly dated 2025: arXiv 2507.11818 was posted in 2025, so the entry is
right and the "Rekesh et al., 2026" reading in the drafted prose was not.

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
| Up to 5 pages, excluding references and appendix | **NOT MET.** The new Figure 1 costs about twenty-seven lines; the body runs eight lines onto page 6 |
| NeurIPS 2026 main-conference template, GEM style file substituted | `\usepackage{GEM_workshop_2026}` |
| Anonymous, one round of double-blind review | Style file's default branch; `\iclrfinalcopy` is not called |
| Maximum 50 MB | 1.5 MB |
| Deadline 30 August 2026, 11:59 PM AoE, via OpenReview | Not submitted by this repository |

The longer abstract and introduction had pushed the body twenty-three lines past five pages. Cutting
the routing material recovered seven. The remaining sixteen came from layout alone, with no further
content cut and no reduction in figure size: the three body floats were carrying `[H]` placement,
which pins a float exactly where it is declared and ships the page out early when it does not fit,
stranding whitespace at the foot of pages 3 through 5. Freeing them to `[tb]` and tightening
LaTeX's default caption and float separation recovered the rest.

Figure 1 remains at its original `0.72\textwidth`. Shrinking it was tested at 0.66, 0.62, 0.56 and
0.52 and is not needed; do not shrink it to buy space that layout already provides. Note that these
floats now move relative to the paragraph that introduces them, which is normal for this template
but means float order should be re-checked after any future body edit.

## Figure 1 and the page budget

The framework overview was added as Figure 1 at `0.92\textwidth`, and every later figure shifts by
one: the sample atlas is now Figure 5 and lives in Appendix F, where its own caption already pointed
for the selection rule and the twelve further examples.

The body had been exactly full at five pages, so this figure costs real space. Measured, from a
clean build each time:

| Configuration | Lines past page 5 |
| --- | --- |
| Figure 1 full width, atlas still in body | 17 |
| Figure 1 full width, atlas moved to Appendix F | 10 |
| Figure 1 at 0.88 or 0.80, atlas in Appendix F | 7 |
| Figure 1 at 0.92, atlas in Appendix F | 8 |

Shrinking Figure 1 saturates: 0.88 and 0.80 give the same result, because the remaining overflow is
Discussion prose, not figure height. Floats move as indivisible blocks, so width tuning cannot
recover a partial float. Closing the last eight lines needs one of: trimming about eight lines of
body prose, moving Table 2 to the appendix, or accepting six pages.

## Build requirement

tcolorbox 6.9.0 and later call `\NewStructureName`, which needs the LaTeX kernel's tagging sockets
from the 2024-11 release. On a TeX Live 2024 format those commands are undefined and roughly
eighteen lines of literal tagging markup are typeset above the title. If that appears, install
tcolorbox 6.3.0 (2024/07/10) under `TEXMFHOME`, or update the kernel. The document source is not
modified to work around it.

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
