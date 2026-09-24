# COMPOSE-Lipid v8 integration

FORGE can import and query the complete 2026-09-17 COMPOSE-Lipid release through a
versioned corpus interface. The source contains 23 reaction-program families plus
three reference-only groups. These definitions are distinct from the earlier
twelve-family reaction registry; the earlier corpus, registry, splits, and model
remain available unchanged.

The subsequent [pre-training audit](COMPOSE_LIPID_V8_PRETRAINING.md) now demonstrates full
protected-TRAIN graph representation, including bromine, and records the remaining exact-program
and precursor-holdout blockers. It does not grant training admission.

## Run and verify

The downloaded source files are in `data/source_cache/compose_lipid_drive_2026-09-18/`.
All inputs, including the README, both decomposition keys, source checksums, existing
registries, atom vocabulary, and historical split guards, are pinned in
`configs/multireaction/compose_lipid_v8_import_v1.json`.

```bash
uv run forge data compose-lipid import \
  --config configs/multireaction/compose_lipid_v8_import_v1.json \
  --output results/phase1/compose_lipid_v8_import_v1

uv run forge data compose-lipid verify \
  --result results/phase1/compose_lipid_v8_import_v1/result.json
```

Import requires a fresh output directory. It streams gzip records into a private
SQLite index and publishes the index, catalogue, and receipt together only after
validation succeeds. Failed imports do not publish a partial corpus. Verification
is read-only and authenticates the configuration, inputs, implementation, and
output artifacts. The large source archives and reproducible SQLite index remain
local data artifacts; the configuration, catalogue, and receipts belong in Git.

## Data contract

- Every universe row is retained, including molecules above the provider's
  80-heavy-atom band. A provider size label never sets FORGE's model support.
- The balanced subset is joined to the universe by the provider's opaque
  `target_id`; overlapping source fields must agree. Provider IDs are retained
  separately from FORGE's canonical constitutional digests.
- Every balanced target must have exactly one assignment. Source split labels
  are preserved verbatim. The FORGE view names `test` as `heldout`, retains
  `reference`, and places universe-only records in `unassigned`.
- Only source TRAIN molecules are parsed for graph checks. Historical M0, Ugi,
  multireaction, and twelve-family product/component partitions constrain the
  training view. Protected identities are retained under `quarantine`, with the
  original source assignment and the exclusion reason.
  This compares complete product identities against the protected identity set;
  checking whether a new product *contains* a held-out precursor still requires
  the later exact decomposition step and is not established by this import.
- Identical constitution strings under different source IDs remain separate
  provenance records. All TRAIN members of these duplicate classes are
  quarantined, including duplicates of heldout or universe-only records. No
  duplicated molecular weight is admitted.
- Source anchors, virtual enumerations, source citations, design lanes, ordered
  precursor labels, coupled-role definitions, variable multiplicity metadata,
  and reference-only groups retain their original meanings.

The supplied JSON key is the program catalogue's source of truth. Ugi precursor
ordering is additionally read from the accompanying Markdown contract through an
explicit configuration binding. Nominal role counts are retained as *nominal*;
they are not treated as validated atom-level occurrences. Missing anchor metadata
remains missing. Component IDs and design-axis values are never invented into
component structures.

## Python interface

```python
from pathlib import Path
from forge.corpus.compose_lipid import ComposeLipidCorpus

corpus = ComposeLipidCorpus(
    Path.cwd(),
    Path("results/phase1/compose_lipid_v8_import_v1/result.json"),
)
for record in corpus.iter_records(split="train", family="aldehyde_ugi3"):
    source = record["source"]          # untouched balanced-release record
    roles = record["role_metadata"]    # labels and counts, not inferred SMILES
    original_split = record["assignment"]["split"]
```

An explicit split is required. Available views are `train`, `calibration`,
`heldout`, `reference`, `unassigned`, and `quarantine`. This is an inspection and
qualification interface: `iter_training_records()` refuses training while exact
program qualification is absent. Every returned training sampling weight is zero.

## Relationship to existing chemistry and model code

The following seven source programs have related transforms in the existing
registry. The catalogue records this relationship for subsequent qualification;
it does not assert that a single transform implements the complete source program.

| v8 source program | Existing transform to examine |
| --- | --- |
| `aldehyde_ugi3` | `ugi_3cr_agile` |
| `amine_epoxide_opening` | `epoxide_opening_amine` |
| `aza_michael_acrylate` | `aza_michael_amine_acrylate` |
| `iphos_ring_opening` | `iphos_amine_dioxaphospholane` |
| `passerini_3cr` | `passerini_3cr` |
| `reductive_amination` | `reductive_amination_amine_aldehyde` |
| `aryl_reductive_amination` | `reductive_amination_amine_aldehyde` |

The other sixteen program definitions are retained with no claimed executable
adapter. Repeated-arm equality, reactive-site selection, coupled ketones,
multi-stage ordering, and source subseries still require their own exact program
checks. A matching generic motif does not qualify them.

Import runs a small deterministic whole-graph representation probe through
FORGE's existing sparse tensorizer and vocabulary, using protected TRAIN rows
only. Exact graph round trips establish representation compatibility for those
specific examples. They do not establish reaction reconstruction, trained-model
size support, source-evidence qualification, or generation quality.

## Admission and evidence boundary

The provider sets `training_admissible=false` throughout the balanced release.
FORGE preserves those values. The import status is
`imported_training_unqualified`; it does not launch training or change any gate.
Future admission must use a separate versioned policy, qualify exact family-specific
decomposition and forward reconstruction, preserve historical and release holdouts,
and assign family-balanced weights after qualification. The raw universe's family
counts are not a sampling prior.

The provider's referenced split-leakage audit is not included in the delivered
folder, so this import does not claim to reproduce it. Its source-study overlap
caveat and the reported/virtual evidence distinction remain applicable.

The input inspection found two departures from the README: some reported anchors
above 80 heavy atoms carry `model_supported`, and some different source IDs carry
identical constitution strings. The import receipt records the measured counts;
the original bytes and failed strict import attempts are preserved. These findings
are handled by retaining source evidence and applying independent FORGE admission
checks, rather than editing the source release.

## Measured import

The complete run retained **3,182,837 source rows**, containing **3,178,450 distinct
constitution strings**. There are **4,387 duplicate classes** spanning **8,774 rows**.
It retained all **200,000** balanced targets and assignments. Of the provider's
159,782 TRAIN assignments, **8,909** conflict with historical protected identities
and another **1,137** belong to duplicate classes. The inspection view therefore
contains **149,736 TRAIN rows**, with **10,046** quarantined and all original split
labels preserved. None is yet admitted for training.

There are **1,085** disagreements between provider size labels and its stated
80-atom band. All such records remain in the index. Two deterministic TRAIN graph
probes per family produced **45 exact sparse round trips out of 46**. One
`aldehyde_ugi4` example contains a neutral bromine atom state absent from the
existing atom vocabulary. The vocabulary was not extended during import.

These counts and all input hashes are recorded in
`results/phase1/compose_lipid_v8_import_v1/result.json`.

An additional identity audit found **1,171** duplicate constitution strings across
provider splits and **zero** crossing active FORGE TRAIN and another active split
after quarantine. Exact queries and input hashes are in
`results/phase1/compose_lipid_v8_validation_v1/identity_split_audit.json`.
This comparison does not establish precursor or near-duplicate separation.

## Validation

All **38 final focused checks**, including fifteen importer tests, pass. Black
and Ruff pass on the five touched Python files. All 30 existing vendor checks and
read-only verification of the real v8 import pass.

The repository-wide run during development reports **2,606 passed, 149 failed,
21 errors, and 92 skipped/xfail**. It caught one new CLI dependency-boundary error;
that was fixed and the complete architecture module passes in the final focused
run. Every other failure/error identity appeared in the earlier repository report.
The full run predates the final boundary correction and two additional tests;
no final-source repository-wide pass is claimed. Global Phase 1 completion remains
unmet. Original full-suite logs and JUnit bytes are preserved in gzip archives.

The complete validation receipt is
`results/phase1/compose_lipid_v8_validation_v1/result.json`, SHA-256
`7476494c84db989f2b4ee875e57441a405e644f720fd1406dd59531f49edf81b`.
