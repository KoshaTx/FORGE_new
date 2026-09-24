# Handoff back to Codex — Claude session, 2026-09-17/18

> Historical session record. The model-facing component grouping, split, and
> target measure were corrected after this handoff. See
> [`MODELING_READINESS_V8_1_2026-09-18.md`](MODELING_READINESS_V8_1_2026-09-18.md)
> and the current [`HANDOFF.md`](HANDOFF.md). Counts below describe the preserved
> v8 split and must not be used as current evaluation claims.

Picked up at `9a4e588` after the prior Codex session hit its usage limit
mid-task. Seven commits, `9a4e588..be54b65`. Working tree clean; suite green.

This records what was verified, what changed, what was found, and — in §7 —
what was *not* checked, so nothing is silently inherited as settled.

## 1. What the prior session actually left

It stopped roughly nine minutes past the finish line of its last task.

`scripts/build_post_instruction_generator_splits_v8.py` **completed** at 03:40.
All five outputs are on disk, `receipt.json` says `complete: true`, and all four
output SHA-256s match the receipt. Producer hashes for the script,
`generator_splits.py`, and the config still match, so the artifact is
reproducible from the tree.

`docs/HANDOFF.md` was written at 03:31 — *before* that build finished — so it
still listed the grouped split as step 1 "to do". That was the only stale claim;
nothing was wrong, just nine minutes behind.

The larger problem was invisible: the entire v8 pipeline was **untracked**. The
200k release selector, all 23 family enumerators, `post_instruction_exact_replay_v8.py`,
every `src/compose_lipid/data/*_design_v8.py`, and all four family tests named in
the handoff's own validation command were files git had never seen. Since
`artifacts/` is gitignored, no part of the v8 lineage was reproducible from the
repository.

## 2. What changed

| commit | files | what |
|---|---:|---|
| `dfdd8a6` | 1,630 | track the v8 pipeline; add split audit, test, browser |
| `0fa0c73` | 2 | collaborator corpus note; `--universe` browsing |
| `03300bc` | 1 | declare `aldehyde_ugi4` provenance |
| `bd75c18` | 1 | data-file placement paths for external readers |
| `47aa408` | 3 | per-family decomposition key (md + json) |
| `42ce408` | 2 | merge family chemistry and source literature into the key |
| `be54b65` | 1 | make the delivered folder self-contained |

**Authorship within `dfdd8a6` matters:** of its 1,630 files, **5 are this
session's** — `scripts/audit_post_instruction_generator_splits_v8.py`,
`audits/post_instruction_generator_splits_v8_leakage_2026-09-17.json`,
`tests/unit/test_post_instruction_generator_splits_v8.py`,
`scripts/browse_corpus.py`, and an edit to `docs/HANDOFF.md`. The other **1,625
are the prior session's untracked work**, swept in by a bulk `git add`. They were
staged, not reviewed. `docs/decisions/0053-*.md`, for instance, dates from
2026-09-13 and is not this session's.

New files authored here:

- `scripts/audit_post_instruction_generator_splits_v8.py` — independent split audit
- `audits/post_instruction_generator_splits_v8_leakage_2026-09-17.json` — its result
- `tests/unit/test_post_instruction_generator_splits_v8.py` — 9 unit tests
- `scripts/browse_corpus.py` — offline RDKit structure browser
- `docs/CORPUS_FOR_EXTERNAL_USE.md` — external collaborator note
- `docs/FAMILY_DECOMPOSITION_KEY.md` + `.json` — per-family decomposition key

## 3. The split, verified

Frozen over all 200,000 targets: train **159,782**, calibration **19,975**,
test **19,975**, reference **268**. 23 formal evaluation families; the three
`reported_*` groups are reference-only.

Panels: unseen precursor identity **7,002**, unseen exact combination **6,327**,
unseen regional topology **5,830**, source-study transfer **828**.

The audit recomputes every invariant from the frozen files rather than trusting
the build-time assertions in `generator_splits.py`. Result: `PASS`, 15/15
bindings verified, **zero** on all six leakage measures, and the combination
panel confirmed to use only train-seen components.

Two further checks run this session, both clean:

- **Cross-form component identity.** Component ids exist in two surface forms —
  bare 64-hex hashes and family-scoped `lumi:amine:<hash>`. If one molecule
  appeared under both, holding out one form would not hold out the other. Zero
  leak by exact id; two apparent leaks by stripped-prefix matching turned out to
  be false positives — `MONO-C14-D10` is a *carboxylic acid* in `o_esterification`
  and a *thiol* in `preassembled_thiol_yne_tail_amidation`. Different molecules,
  shared design code. The family scoping is doing its job. **Do not match
  component codes with the family prefix stripped.**
- **Panel coverage per family.** Every family's test rows are essentially fully
  panel-covered; there are no untested test rows hiding in any family.

### Declared caveat, carried forward

Source-study isolation holds per `(family, study)`. `pmid:38409275` is held out
under `amine_epoxide_opening` (5 test targets) while **22 training targets from
the same paper** sit under the sibling family `epoxide_opening_o_acylation`.
Inside contract, outside a literal "unseen paper" reading. It is recorded in the
audit JSON and in `docs/HANDOFF.md`.

## 4. Corpus provenance findings

**`aldehyde_ugi4` is entirely its source paper's grid.** All 96 components are
`tier: source_transcribed` from one author reagent table, and the family is
their exact complete cross product:

```
32 amines x 16 aldehydes x 36 acids x 12 isocyanides = 221,184
221,184 enumerated + 845 reported anchors            = 222,029  (matches the universe count)
```

Nothing in it is designed beyond the source. It is a faithful reconstruction of
published combinatorial space, not novel enumeration. **Consequence for
modelling:** strong per-family performance there partly measures reproduction of
a published grid. Only **1 of those 96 reagents** is in the held-out component
panel — the family's 10 held components are mostly source-anchor components — so
unseen-reagent generalization in that chemistry is close to untested.

**`aldehyde_ugi3` is the opposite.** In the `joint_family_expansion_v1`
inventory its `agile:` components are 31 `source_partner` against 40 designed
(32 `designed_interior_homologue`, 8 `designed_joint_interior_homologue`). Its
17,484 virtual rows split across `agile_author_released_virtual_design` 11,136,
`agile_missing_aldehyde_homologue` 6,336 (ours), and
`separate_muscle_named_context` 12 — the last being the rows that broke the
split build twice and needed the bounded-context override.

**Component codes are not globally unique.** A heuristic pass over the banks
found **122 of 26,057** bare codes mapping to more than one molecule across
families (`DEA-S3` is an acid in family 12 and an amine in family 22). Key any
component lookup on `(family, bank, code)`.

## 5. The decomposition key

`docs/FAMILY_DECOMPOSITION_KEY.md` documents all 23 families: reaction program
invariant, role decomposition with counts, which `primary_metadata` field names
each role, authorized knobs, axis values present in the data, all **55**
architecture subfamilies with port signatures, never-collapse rules, and **32**
source libraries with PMIDs/DOIs.

The field→role mapping was derived **by perturbation** — each metadata field
altered in isolation, recording which role's component identity changed — not by
reading field names. Eight families vary their repeat count per row
(`occupancy` / `events` / `event_count`); the two Ugi families carry components
positionally in `precursor_ids`.

The chemistry and citations come from
`configs/corpus/user_family_architecture_knob_registry_v7_4.json`, which was
already in the repo and is the authoritative per-family source. It was not
referenced by any handoff before this.

## 6. External handoff (the collaborator's copy)

`~/Desktop/COMPOSE_Lipid_corpus_2026-09-17/` holds the bundle: the three data
files, `README.md` (= `docs/CORPUS_FOR_EXTERNAL_USE.md`), both decomposition-key
files, and `SHA256SUMS`. All three data copies were verified byte-identical to
the originals, and those hashes match the build receipts.

The note states the claims boundary, why the 200k is rebalanced and why
per-family work should start from the 3.18M universe instead, the `aldehyde_ugi4`
provenance caveat, and the decomposition workflow. The delivered folder is
self-contained; the repository is optional.

## 7. What was NOT verified — read before relying on anything above

- **The 1,625 swept-in files were staged, not reviewed.** They were confirmed to
  be `.py`/`.json`/`.md` totalling ~924 K, and the named test suite passes. No
  line-level review was done.
- **The split build was not re-run.** Its outputs were verified against its
  receipt; the construction itself was not reproduced from scratch.
- **The 78% component-id resolution figure and the 122-code-collision figure
  come from throwaway probes**, not committed scripts. Neither is reproducible
  from the repo as it stands. An earlier version of the coverage probe reported
  **39.9%** before it was found to be skipping uncompressed `.jsonl`
  inventories — treat both numbers as indicative, and rebuild the probe properly
  before acting on them.
- **No id→SMILES table exists.** Component identifiers do not resolve to
  structures anywhere in the shipped or committed artifacts.
- **`browse_corpus.py` universe mode** was measured at ~0.24 GB resident for
  3.18M rows. That is lower than expected and may reflect macOS memory
  compression; it was not independently confirmed.
- **Nothing model-side has been done.** No target measure, no paths, no
  training.

## 8. Current state and what is next

Suite: `21 passed` (the command in `docs/HANDOFF.md` §Validation).
Working tree clean at `be54b65`. No background processes running — both
`browse_corpus.py` servers from this session are down; restart with
`python3 scripts/browse_corpus.py --open` or `--universe --open`.

**Disk: 23 GiB free as of 2026-09-18.** This moved a great deal during the
session (5.3 → 4.1 → 3.4 → 23 GiB) without deliberate cleanup, so re-check
before a large write rather than trusting this figure. The earlier estimate for
step 2 — path materialization at roughly **8 GiB**, extrapolated from
`training_corpus_programs_v1` (1.1 GiB for 27,260 targets, scaled 7.34x to
200,000) — now fits, where it did not at 4.1 GiB.

Remaining steps, unchanged from `docs/HANDOFF.md`:

1. freeze the train-only target measure — **next**; must be fitted on the train
   split alone, the frozen calibration/test assignments must not be recomputed
2. project supervision records into exact model programs, materialize paths
3. candidate batches, CPU reconstruction/runtime checks
4. bounded one-seed regional-context health comparison
5. freeze the production chemistry process

Offered this session and **not** done: a committed component dictionary
(`(family, role, id) → SMILES`, with round-trip replay as the acceptance gate);
role labels for the full 3.18M universe (the `split_groups` equivalent, which
exists only for the 200k); and a check of which large artifact directories are
orphaned versus still bound by a current receipt.

## Appendix — per-family evaluation strength

Held-out component counts per family. The builder allocates ~30% of each
family's test quota to the unseen-component panel, so the spread tracks
component-bank size rather than inconsistency. `aryl_reductive_amination` is the
thinnest probe: 27 held of 4,031 components across 6,046 targets.

| family | targets | components | held out | held % | test rows |
|---|---:|---:|---:|---:|---:|
| `preassembled_thiol_yne_tail_amidation` | 13,274 | 1,422 | 22 | 1.5% | 1,327 |
| `ketone_isocyanide_amide` | 11,869 | 2,223 | 49 | 2.2% | 1,187 |
| `thiolactone_aminolysis_michael` | 11,853 | 1,948 | 45 | 2.3% | 1,185 |
| `aldehyde_ugi4` | 11,134 | 394 | 10 | 2.5% | 1,113 |
| `acid_epoxide_diester_multistep` | 10,700 | 5,611 | 82 | 1.5% | 1,070 |
| `alpha_isocyanoester_dihydroimidazole` | 10,439 | 2,241 | 52 | 2.3% | 1,044 |
| `amine_alkylation` | 10,075 | 6,975 | 79 | 1.1% | 1,008 |
| `disulfide_michael` | 9,675 | 4,774 | 91 | 1.9% | 968 |
| `iphos_ring_opening` | 9,369 | 5,487 | 72 | 1.3% | 937 |
| `aza_michael_acrylate` | 8,776 | 1,104 | 32 | 2.9% | 878 |
| `passerini_3cr` | 8,759 | 4,040 | 50 | 1.2% | 876 |
| `amine_epoxide_opening` | 8,541 | 1,964 | 34 | 1.7% | 854 |
| `vitamin_b5_multistep` | 7,746 | 85 | 10 | 11.8% | 775 |
| `a3_amine_aldehyde_alkyne` | 7,680 | 172 | 5 | 2.9% | 768 |
| `ketone_ugi4` | 7,371 | 152 | 6 | 3.9% | 737 |
| `reductive_amination` | 7,289 | 2,615 | 46 | 1.8% | 729 |
| `maleate_addition` | 7,259 | 1,467 | 15 | 1.0% | 726 |
| `aldehyde_ugi3` | 6,925 | 181 | 8 | 4.4% | 692 |
| `aza_michael_acrylamide` | 6,396 | 755 | 30 | 4.0% | 640 |
| `epoxide_opening_o_acylation` | 6,339 | 1,326 | 22 | 1.7% | 634 |
| `aema_aza_thiol_addition` | 6,126 | 373 | 18 | 4.8% | 613 |
| `o_esterification` | 6,091 | 1,404 | 32 | 2.3% | 609 |
| `aryl_reductive_amination` | 6,046 | 4,031 | 27 | 0.7% | 605 |
