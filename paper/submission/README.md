# Submitted FORGE manuscript

**Start with [FORGE.pdf](FORGE.pdf).** This is the unmodified PDF supplied for the reviewer cleanup,
originally named `FORGE__ICLR_ (2).pdf`. It contains 39 pages and is titled *FORGE: Reaction-Guided
Generative Design of Ionizable Lipids*. The PDF itself says “Under review as a conference paper
at ICLR 2027”; this repository does not assert acceptance or assign a different venue.

SHA-256: `96e1a21b3be3d59431ef1f9c256171a8fcba984305a03999f32f1d35dc66fa35`.

## Suggested review order

1. Read the method and main comparisons in the PDF, pages 1–9.
2. Use [EVIDENCE.md](EVIDENCE.md) to navigate from each table, figure and method to its files.
3. Run `make review-check` from the repository root. It needs only Python 3.10+ and no network.
4. Read [ARTIFACTS.md](ARTIFACTS.md) before attempting generation or numerical reproduction.
5. Read Appendix A.8 and the [historical implementation record](../v1_iclr/mathematical_review.json)
   for the boundary between present-day code and the code used for historical runs.

## What is included

- The exact submitted PDF, unchanged.
- Related generated tables, molecular drawings, experimental assets and provenance in
  [`../v1_iclr/`](../v1_iclr/), indexed and SHA-256-pinned by [manifest.json](manifest.json).
- A paper-to-code map and a deterministic offline integrity/arithmetic check.
- Current source and retained historical evidence elsewhere in this repository.

The check proves file identity and a stated arithmetic cross-check. It does not establish that
all reported experiments can be rerun, that a rendered table is an independent measurement,
or that present-day implementations equal missing historical source snapshots.

## Manuscript source status

The existing `paper/v1_iclr/FORGE_ICLR2027_paper.tex`, compiled PDF and Overleaf exports are an
**earlier related revision**, not the source of this exact submitted PDF. In particular, the supplied
PDF changes the abstract/introduction/discussion and places a two-panel imaging figure on page 8;
the older canonical source includes an additional panel of supplied lipid structures.

The supplied PDF is authoritative for this review branch. Its exact LaTeX/Overleaf source has not
been supplied. Building the older source is useful for inspecting its assets but will not recreate
the submitted PDF. Numerical table assets matching the submission remain in their original paths;
older source provenance is not rewritten to claim a newer source identity.

## Maintainer review before external release

- Supply the exact submitted LaTeX export if editable manuscript reproduction is required.
- Publish/access-enable the specific checkpoint archives and evaluation inputs listed in
  [ARTIFACTS.md](ARTIFACTS.md), with their recorded hashes and applicable data permissions.
- Supply the remaining experimental details listed in [EVIDENCE.md](EVIDENCE.md#experimental-record).
  The administration route is intramuscular, clarified by the user to match the submitted paper.
- Choose the appropriate reviewer distribution. This is the existing organization repository with
  retained Git history; the PDF's anonymous author block does not make this repository anonymous.

No new scientific runs, thresholds, results or experimental claims are introduced by this cleanup.
