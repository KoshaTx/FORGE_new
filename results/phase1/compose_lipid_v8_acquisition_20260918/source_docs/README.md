# COMPOSE-Lipid v8 corpus — note for external collaborators

**The data files are not in this repository.** They are provided separately
(see *Receiving the data*). This note is the document of record for what the
corpus is, how it was built, what may be claimed from it, and how to check that
what you received is what we sent.

Snapshot: 2026-09-17. Repository lineage: `dataset/source-corrected-corpus-v3`.

---

## 1. What this is

A corpus of lipid-like molecular graphs enumerated across **23 exact
reaction-program families**. Most families were rebuilt from their own
literature sources with family-specific components and their own permitted
design knobs
(head topology, spacer, substitution, tail length, branch position and size,
positional unsaturation, internal-ester position, occupancy, site, stage order,
symmetry). Coupled or repeated arms stay coupled. No global reactive-handle
pool was used as the design rule, so the families are not interchangeable
slices of one generator — they are separate chemistries.

How far a family goes beyond its sources varies, and for one family it is zero.
See §4.1 before assuming any family is novel enumeration.

Two layers are available, and they answer different questions:

| | rows | what it is | use it for |
|---|---:|---|---|
| **Candidate universe** | 3,182,837 | every globally unique constitution the 23 programs produced | per-family experiments, scaling studies, your own subsampling |
| **Balanced release** | 200,000 | a tempered, coverage-selected subset of the universe | a ready-made training corpus with frozen splits |

The universe is the bigger, less opinionated artifact. If you want to run your
own experiments per family, **start from the universe** and apply your own
selection; the 200k encodes our modelling choices, which may not be yours.

---

## 2. The claims boundary — please read before publishing

Everything here is **graph-level chemistry only**. Each virtual row is an exact
graph construction under a declared reaction program: the product graph is what
that transformation yields if it proceeds as written, on those components.

These files are **not** evidence of, and must not be reported as:

- synthesizability, yield, or purity;
- commercial or synthetic availability of any precursor;
- reaction selectivity under real conditions;
- ionization, formulation, delivery, potency, or any biological activity;
- a historical or literature-attested route for any virtual row.

11,267 rows are **reported source anchors** — constitutions transcribed from
published work, carrying `source_anchor: true` and their PMIDs. Those have
literature provenance for the *molecule*. The remaining 2,971,570 (universe)
are exact virtual constructions and have none. Any per-row claim beyond graph
identity needs independent qualification, and the two strata should not be
pooled when making evidential claims.

No biological outcome data is included in, or was used to build, any of these
files.

---

## 3. What you receive

Everything needed is in the delivered folder. The repository is optional — see
§7.

**Data** — gzipped JSON Lines, one JSON object per line:

| file | rows | size | sha256 |
|---|---:|---:|---|
| `global_candidates.jsonl.gz` | 3,182,837 | 537 MB | `05e3151e4cb44cec63d5752202ffd8b4691134bae40126bdc86d33326e9fca30` |
| `accepted_targets.jsonl.gz` | 200,000 | 34 MB | `0d363b9602421f3db3b4553ef28330da1342744960b3b341a9e05f24efc095b3` |
| `assignments.jsonl.gz` | 200,000 | 33 MB | `c99324118408adfd4e83ccbe1576977d86375d9c7d077b1ce288e07a98b6e15d` |

**Documentation** — travels with the data:

| file | what it is |
|---|---|
| `README.md` | this note |
| `FAMILY_DECOMPOSITION_KEY.md` | per-family chemistry, roles, knobs, subfamilies, source papers |
| `family_decomposition_key.json` | the same, machine-readable |
| `SHA256SUMS` | checksums for the three data files |

Verify before use — in the folder, run:

```bash
shasum -a 256 -c SHA256SUMS
```

Three `OK` lines. These checksums are the ones recorded in the build receipts,
re-verified against the files on 2026-09-17.

### Row schema

`global_candidates.jsonl.gz` and `accepted_targets.jsonl.gz`:

| field | meaning |
|---|---|
| `target_id` | sha256 of the canonical constitution; the join key everywhere |
| `constitution` | canonical, stereochemistry-free SMILES — the molecule |
| `primary_family` | which of the 23 reaction programs is the row's primary assignment |
| `route_families` | every family that can reach this constitution (5,550 rows are reachable by more than one) |
| `heavy_atoms` | heavy-atom count |
| `size_disposition` | `model_supported` (≤80 heavy atoms) or `above_80_heavy_atom_hold` |
| `source_anchor` | `true` for reported literature molecules |
| `primary_metadata` | the family-specific design axes and component ids for that row |
| `training_admissible` | always `false` — see §2 |

`assignments.jsonl.gz` adds, for the 200k only: `split`, `test_panels`, and the
four grouping signatures used to build them.

Deduplication is global and by constitution: no constitution appears twice in
the universe, including across families. `route_families` is how you recover
cross-family reachability.

---

## 4. Per-family breakdown

`≤80` is the model-supported band; `hold` is above it. Size holds are highly
uneven — several families are entirely within band, while `disulfide_michael`
and `iphos_ring_opening` carry most of the large-molecule mass.

| family | universe | ≤80 | hold | anchors | in 200k |
|---|---:|---:|---:|---:|---:|
| preassembled_thiol_yne_tail_amidation | 465,048 | 465,048 | 0 | 112 | 13,274 |
| thiolactone_aminolysis_michael | 334,859 | 334,859 | 0 | 91 | 11,853 |
| ketone_isocyanide_amide | 271,560 | 271,560 | 0 | 900 | 11,869 |
| disulfide_michael | 254,389 | 177,658 | 76,731 | 52 | 9,675 |
| acid_epoxide_diester_multistep | 232,240 | 232,240 | 0 | 256 | 10,700 |
| aldehyde_ugi4 | 222,029 | 222,029 | 0 | 845 | 11,134 |
| alpha_isocyanoester_dihydroimidazole | 219,036 | 219,036 | 0 | 184 | 10,439 |
| amine_alkylation | 188,272 | 188,272 | 0 | 286 | 10,075 |
| iphos_ring_opening | 184,345 | 132,101 | 52,244 | 549 | 9,369 |
| passerini_3cr | 128,352 | 128,352 | 0 | 0 | 8,759 |
| amine_epoxide_opening | 97,439 | 87,303 | 10,136 | 661 | 8,541 |
| ketone_ugi4 | 91,395 | 63,827 | 27,568 | 72 | 7,371 |
| vitamin_b5_multistep | 80,398 | 80,340 | 58 | 18 | 7,746 |
| a3_amine_aldehyde_alkyne | 70,016 | 51,440 | 18,576 | 764 | 7,680 |
| maleate_addition | 68,463 | 47,400 | 21,063 | 463 | 7,259 |
| reductive_amination | 61,320 | 61,320 | 0 | 60 | 7,289 |
| aryl_reductive_amination | 48,090 | 23,530 | 24,560 | 168 | 6,046 |
| epoxide_opening_o_acylation | 42,805 | 33,432 | 9,373 | 25 | 6,339 |
| aza_michael_acrylate | 33,619 | 32,609 | 1,010 | 2,599 | 8,776 |
| aza_michael_acrylamide | 31,239 | 30,229 | 1,010 | 219 | 6,396 |
| o_esterification | 24,948 | 24,948 | 0 | 144 | 6,091 |
| aldehyde_ugi3 | 18,842 | 18,842 | 0 | 1,358 | 6,925 |
| aema_aza_thiol_addition | 13,865 | 10,156 | 3,709 | 1,173 | 6,126 |
| *reported_lnpdb_route_unassigned* | 255 | 255 | 0 | 255 | 255 |
| *reported_multistep_amino_lipid* | 9 | 9 | 0 | 9 | 9 |
| *reported_reference_lipid* | 4 | 4 | 0 | 4 | 4 |
| **total** | **3,182,837** | **2,936,799** | **246,038** | **11,267** | **200,000** |

The three italic `reported_*` groups are literature molecules we could not
assign to one of the 23 programs. They are reference material, not a family.

### Working out which part of a molecule is which

[`FAMILY_DECOMPOSITION_KEY.md`](FAMILY_DECOMPOSITION_KEY.md) documents all 23
families: the reaction that formed the bonds, the roles a molecule decomposes
into, how many of each, which `primary_metadata` field names each one, the
authorized design knobs, the 55 architecture subfamilies with their port
signatures, and the source papers (32 citations with PMIDs and DOIs).

The workflow is three steps:

1. Read the row's `primary_family` and look it up in the key. That gives the
   architecture — for `preassembled_thiol_yne_tail_amidation`, one
   `amine_head`, one `alkynoic_linker`, and a `thiol_tail` used **twice**.
2. Read the row's `primary_metadata` for which components fill those slots —
   `head_id`, `linker_code`, `tail_code` for that family.
3. Use the reaction invariant to place the cuts. For that family: *double thiol
   addition to one alkynoic-acid scaffold, then amidation to one complete
   amine.* So expect two thioether bonds and one amide, with both sulfur arms
   carrying the identical tail.

The architecture is fixed per family; what varies row to row is which
components fill it. Eight families also vary the repeat count itself — the key
names the field to read.

The key does **not** resolve component identifiers to structures; no id-to-SMILES
table exists in these files. Steps 1–3 tell you what to look for, and the
substructure match is yours to run.

If you publish from a family, cite its source papers from the key.

### 4.1 Provenance is not uniform — read this before using a family

Family size says nothing about how much of a family is new. Components come in
two kinds: **transcribed** from a published reagent table, and **designed** by
us as variants along a declared knob (tail length, branch position and size,
positional unsaturation, internal-ester position, and so on). The mix differs
per family, and in one case there is no mix at all.

**`aldehyde_ugi4` is entirely its source paper's grid.** All 96 of its
components are `source_transcribed` from a single author reagent table, and the
family is the exact, complete cross product of them:

```
32 amines x 16 aldehydes x 36 carboxylic acids x 12 isocyanides = 221,184
221,184 enumerated + 845 reported source anchors             = 222,029
```

Nothing in `aldehyde_ugi4` is designed beyond the source, and the count matches
the combinatorial total the source authors themselves report. Treat this family
as a faithful reconstruction of published combinatorial space, **not** as novel
enumeration, and cite the source paper rather than this corpus for it.

The contrast is `aldehyde_ugi3`, whose 71 components are 31 taken from its
source and 40 designed homologues — so a majority of that family's component
space is ours, not the paper's.

The remaining families carry explicit design axes in `primary_metadata`
(`tail_axis`, `acid_axis`, `head_axis`, `thiolactone_axis`, and their review
lanes), which is where you can see per-row how far a component sits from its
source. If the distinction matters to your claims, read those fields rather
than assuming.

---

## 5. Why the 200k is smaller, and why it is not a random sample

Family size in the universe reflects **enumeration combinatorics**, not
chemical importance. A family whose knobs multiply out — thiol-yne, at 465,048
— is not 34× more interesting than aema at 13,865; it simply has more knob
positions. Training directly on the universe would teach a model the shape of
our enumerator.

The 200k therefore applies, in order:

1. **all 11,267 source anchors pinned** — no literature molecule is dropped;
2. **family capacity tempered** rather than copied as raw abundance, so large
   families give up most of their mass (thiol-yne contributes 2.85% of its
   universe; aema contributes 44.18%);
3. **component coverage selected before family-axis filling**, so rare
   components survive rather than being crowded out by dense regions.

The result is a corpus where families are comparable in size, which is what we
wanted for training. **If your experiment is per-family, this rebalancing is
probably the wrong prior for you** — take the universe and select your own.

---

## 6. The frozen splits (200k only)

`assignments.jsonl.gz` carries a component-aware grouped split:

- train **159,782** · calibration **19,975** · test **19,975** · reference **268**
- 23 formal evaluation families; the three `reported_*` groups are
  reference-only and appear in no split.

Grouping is by chemical identity, not by molecule, so near-duplicates cannot
straddle the boundary. Four held-out panels, each answering a different
generalization question:

| panel | targets | asks |
|---|---:|---|
| `unseen_precursor_identity` | 7,002 | a component never seen in training |
| `unseen_exact_combination` | 6,327 | seen components, never in this combination |
| `unseen_regional_topology` | 5,830 | an unseen regional arrangement |
| `source_study_transfer` | 828 | molecules from a held-out source study |

An independent audit
(`audits/post_instruction_generator_splits_v8_leakage_2026-09-17.json`)
recomputes every invariant from the frozen files: zero leakage on all checks,
and the combination panel is confirmed to use only train-seen components.

**One declared caveat.** Source-study isolation is defined per
`(family, study)`. `pmid:38409275` is held out under `amine_epoxide_opening`
(5 test targets), but 22 training targets from the same paper remain under the
sibling family `epoxide_opening_o_acylation`. Within contract, but if you
report `source_study_transfer` as a literature holdout, this footnote belongs
in the report.

---

## 7. Loading

```python
import gzip, json

def rows(path):
    with gzip.open(path, "rt") as handle:
        yield from map(json.loads, handle)

# one family out of the universe, model-supported band only
family = "aldehyde_ugi4"
smiles = [
    r["constitution"]
    for r in rows("global_candidates.jsonl.gz")
    if r["primary_family"] == family and r["size_disposition"] == "model_supported"
]
```

Stream it — the universe does not need to be materialized in memory, and
loading all 3.18M rows as dicts will cost far more RAM than the 537 MB on disk.

To attach splits to the 200k, join `accepted_targets.jsonl.gz` and
`assignments.jsonl.gz` on `target_id`.

### Optional: the repository

You do not need the repository — the folder above is self-contained. If you do
have access to `compose_lipid`, it carries the code and a copy of these notes,
but never the data. To use `scripts/browse_corpus.py` (a small offline
RDKit structure browser for either layer), drop the three files into the paths
its receipts expect, relative to the repository root:

```
artifacts/corpus_build_v2/post_instruction_global_union_v8/global_candidates.jsonl.gz
artifacts/corpus_build_v2/post_instruction_balanced_release_v8/accepted_targets.jsonl.gz
artifacts/corpus_build_v2/post_instruction_generator_splits_v8/assignments.jsonl.gz
```

Then:

```bash
python3 scripts/browse_corpus.py --open            # the 200k, with splits shown
python3 scripts/browse_corpus.py --universe --open # all 3.18M
```

It needs RDKit and nothing else, runs entirely offline, and never writes to the
data files. Everything else in the repository — the enumeration and audit
scripts, the split leakage audit, this note — is readable without the data.

---

## 8. Receiving the data

The three files are transferred out of band. On receipt:

1. check the sha256 values in §3 — do not use a file that does not match;
2. keep the filenames, since the receipts reference them;
3. cite this note's snapshot date, so it is unambiguous which build you have.

Questions about provenance for a specific row: quote its `target_id`, which is
stable across all three files and is derived from the constitution itself.
