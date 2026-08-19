# Migration notes — forge → forge_new

Date: 18 August 2026

## Source

Everything here came from the old `forge/` repository at:

```
branch  agent/active-design-20260815
commit  531d5a90694bb1dc3767861a07616f160fa0810e
date    2026-08-18 10:18:56 -0400
```

That branch was verified to be the single authoritative line of work. The old repo had 16
branches; 14 of them are strict ancestors of `active-design-20260815`, so they contributed
nothing it does not already contain. The two that were not ancestors —
`m0-06-defog-feasibility` and `m0-06-sparse-topology-probe` — introduce **no files** that
`active-design` lacks, and every file they share with it is older and shorter there
(for example `src/forge/product/sparse_topology_feasibility.py` is 1,709 lines on the probe
branch against 1,875 lines on `active-design`). Nothing was lost by taking one branch.

`forge/` was left in place and untouched. It still holds all 238 commits of history and the
`origin` remote; this migration copied files out of it rather than moving them, because the
working tree of `forge/` sits on `main`, which contains only a README — the content exists
solely as git objects.

## What moved

| Destination | Contents |
|---|---|
| `src/` | the `forge` package — 295 modules |
| `scripts/` | 343 experiment and figure runners |
| `configs/` | 356 frozen run configurations |
| `data/` | split manifests, AGILE reference data, vendor manifest |
| `results/` | 110 tracked result artifacts (see the gap below) |
| `tests/` | 275 test modules |
| `docs/` | the 64 design, audit and decision documents |
| `docs/provenance/` | evidence matrix (JSON + Markdown), evidence ledger, figure visual system, reference audit |
| `paper/figures/framework/` | Figure 1 source and rendered output |
| `paper/figures/chemist_packet/` | `FORGE_candidate_smiles.csv`, the 34-candidate chemist packet |
| `paper/latex/` | ICLR and Nature Biotech templates, pandoc filters, `iclr2027` class files |
| root | `Makefile`, `pyproject.toml`, `AGENTS.md`, `CLAUDE.md`, `README.md`, `.gitignore` |

`paper/` itself was **not** overwritten. The copy of `FORGE_ICLR2027_paper.tex` already in
`forge_new/paper/` is newer than the one on the branch (175,224 bytes against 170,641), as are
`forge.bib`, `generated_product_structures.tex` and several `panel_v6` figures. Files were merged
into `paper/` only where `paper/` did not already have them.

## What was deliberately left behind

All of it superseded, none of it code:

- `manuscript/FORGE_ICLR2027_paper.tex` — older than the copy in `paper/`.
- `manuscript/FORGE_ICLR2027_submission.md` — the paper is now authored in LaTeX directly, not
  generated from Markdown.
- `manuscript/FORGE_Nature_Biotechnology_working_draft.{md,docx}`,
  `NATURE_BIOTECH_DRAFTING_ANALYSIS.md`, `Perspective.docx` — a different paper direction.
- `manuscript/figures/_archive_2026-08-06/` — archived concept art and superseded Figures 1–2.
- `manuscript/figures/panel_v4/` — superseded by `panel_v6`.

## The results gap — read this before trying to rebuild

`results/*` is gitignored behind a whitelist. The whitelist covers the M0 milestone thoroughly
but almost none of phase 1. **82 of the artifacts the paper's own ledgers cite are not in the
repository and do not exist anywhere on this machine.** They were written on the machine that ran
the training and never tracked. The missing set includes:

- every trained checkpoint (`*.pt` is globally ignored) — the production generator
  `ugi_joint_sparse_balanced_v2_full`, the size-only ablation arm, the oracle checkpoint;
- the balanced training corpus and its cache (`ugi_balanced_chemistry_corpus_v2/`,
  `ugi_balanced_training_cache_v2/`);
- the expanded enumeration products, the component registry, the dossier and readiness ledgers;
- the prospective panel (`ugi_prospective_panel_v6/prospective_panel.jsonl.gz`), which is what the
  four `panel_40_structures` figures are drawn from;
- the semantics pilot, novelty audit, actionability funnel and rank-sensitivity results that the
  abstract's numbers come from.

What survived is the *rendered* output: the figure PNGs in `paper/figures/`, the generated LaTeX
tables, and the numbers themselves, which are hardcoded as `\newcommand` macros at the top of
`FORGE_ICLR2027_paper.tex` and transcribed in `docs/FORGE_MANUSCRIPT_NUMBERS_CURRENT.md` and
`docs/provenance/`.

Consequence: **the paper can be rebuilt, but its results cannot be regenerated here.** Recovering
them means retrieving `results/phase1/` from the workstation that produced it. Until then the
provenance documents are the only record linking each published number to the artifact and SHA-256
it came from.

## Why the engine was moved whole rather than pruned

A static reachability pass was run first — root the analysis at the artifacts the paper cites, find
the scripts that name them, close over `forge` imports. It nominated 73 scripts and 49 modules as
unreachable, but spot-checking showed the list was wrong: `forge.route.graph2edits_single_step_recovery`
appeared on it while producing the paper's 29/36 single-step recovery figure, and
`ugi_balanced_chemistry_corpus` appeared on it while producing the 112,386-product training corpus
named in the abstract. Scripts assemble their output paths from configs and arguments rather than
string literals, so literal-match producer detection under-reports badly.

Rather than prune on an analysis known to be unsound, the whole engine was moved. Pruning it later
is cheap and safe; recovering a module deleted because a heuristic missed it is neither. If you do
want to prune, drive it from `docs/provenance/COMPUTATIONAL_RESULTS_EVIDENCE_MATRIX.json`, which
maps each claim to its artifact, and trace producers through the `Makefile` targets rather than
through import edges.

## Known rough edge

The figure and table generators — `phase1_render_prospective_panel_v6_figure.py`,
`phase1_build_manuscript_supplement_v1.py`, `phase1_render_appendix_tables_v1.py` and the others —
write into `manuscript/`, which no longer exists here; the paper now lives in `paper/`. Those output
paths need redirecting before any regeneration run. This is moot until the missing `results/phase1/`
artifacts are recovered, since none of those scripts can run without their inputs.

## Repository initialization

`forge_new/` was initialized as a fresh git repository on 18 August 2026 with a single root commit.
History was not carried over — `forge/` remains the archive of the original 238 commits and its
`origin` remote.

Two adjustments were made as part of that:

- `.gitignore` was rewritten for the new layout. The `manuscript/*` rules were dead once the paper
  moved to `paper/`, which would have left latexmk intermediates tracked. `paper/` now ignores
  `.aux`, `.blg`, `.fls`, `.fdb_latexmk`, `.log`, `.out` and `.toc`. The PDF and `.bbl` are tracked
  on purpose — they are the submission deliverables, and Overleaf bundles need the `.bbl`. The
  `results/` whitelist was left exactly as it was.
- `manuscript/NATURE_BIOTECH_DRAFTING_ANALYSIS.md` was recovered into `docs/provenance/`. It had
  been left behind with the other Nature Biotech material, but `CLAUDE.md` names it as required
  reading for manuscript prose, so dropping it would have broken the bootstrap. The pointer in
  `CLAUDE.md` was updated to the new path.

Pointers in `CLAUDE.md` and the docs that referred to `manuscript/` paths were checked. Everything
still referenced resolves, except for links to the four superseded drafts listed above, which were
intentionally not migrated.
