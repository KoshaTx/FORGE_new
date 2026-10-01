# COMPOSE v8 pre-training qualification

Training on v8 remains disabled. The graph representation now covers the entire protected
TRAIN inspection population, including bromine. Seven family subsets now have complete computed
program records supported by their source controls; their source-bank and architecture limits
remain explicit. A3 has a separate conditional replay diagnostic with unresolved source conflicts.
The cross-family precursor audit below combines the seven exact-program ledgers, preserves all
23 families in its accounting, and identifies the remaining holdout gaps. Neither program
qualification nor a representation pass grants training admission.

## Runnable checks

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_pretraining \
  --config configs/multireaction/compose_lipid_v8_pretraining_v1.json \
  --output-dir results/phase1/compose_lipid_v8_pretraining_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_pretraining \
  --verify results/phase1/compose_lipid_v8_pretraining_v1/result.json

uv run python -m experiments.phase1.multireaction.compose_lipid_representation \
  --config configs/multireaction/compose_lipid_v8_representation_v1.json \
  --output-dir results/phase1/compose_lipid_v8_representation_replay
```

Execution requires a fresh output directory. Both commands authenticate the imported corpus,
its input data and implementation. They read its SQLite index without changing source flags,
provider splits or historical guards. Completed outputs publish atomically; failed execution
cannot publish a passing result. No model training, molecular generation or v8 held-out graph
parsing occurs. Existing historical split identities are normalized solely to enforce exclusions;
they do not fit the new vocabulary or component pool. The smoke test initializes a small CPU
model and performs only a forward pass.

## Graph representation

All **149,736** protected TRAIN graphs across **23** families reconstruct their original
constitutional identities exactly under both tested encodings. This population is the
balanced release after the importer's product-level protections, not the complete 3.18M-row
universe and not a precursor-qualified training set.

The first audit found **893 bromine-containing rows** whose atom state was absent from the
old vocabulary. It also exposed a separate mismatch: **5,879** valid graphs exceed the old
local valence capacities when aromatic bonds are counted as three half-bond units. Explicit
aromatic N-H, substituted aromatic nitrogen and fused aromatic bonds require care under that
accounting. The failing audit and its per-row ledger remain preserved.

The second audit uses FORGE's existing constitutional **Kekulé** encoding. Single, double and
triple bond tokens retain the original valence limits; sanitization must recover the exact
original aromatic graph. Every TRAIN graph passes both exact reconstruction and capacity
checks, without discarding aromatic molecules or increasing those limits. Maximum observed
support is **96 atoms and 8 closure edges**. The vocabulary has **8 atom states**, with three
active sparse bond classes. A 26-record CPU smoke panel spanning every family, maximum size,
maximum closure count and bromine produces finite outputs.

`QualifiedAtomVocabulary` adds explicit, immutable, opt-in neutral monovalent support. RDKit's
periodic table must confirm the declared extension. Ordinary vocabularies retain their old
behavior; charged, aromatic or hydrogen-bearing bromine is not admitted. Tests require a
single-bond bromine graph to decode and an overbonded bromine graph to abstain.

The successful vocabulary artifact is:
`results/phase1/compose_lipid_v8_representation_v1/atom_vocabulary.json`.
Its `status: pass` covers representation only. A future admitted training configuration must
use its Kekulé mode, three bond classes and explicit valence extension together. It cannot
reuse a checkpoint with different atom or bond indices. Refit and recheck the vocabulary on
the final precursor-protected TRAIN population before training.

## Reconstruction and precursor checks

The seven related existing registry transforms are diagnostic leads. Their roles, handle
policies, site multiplicity constraints and bounded-search rules remain unchanged. A repeated
program is never counted as reconstructed after just one local cut. An ambiguous inverse or
forward site class abstains.

Across the full TRAIN inspection population:

| Related-transform outcome | Rows |
| --- | ---: |
| Unique exact replay, no recovered historical protected precursor | 8,315 |
| Unique exact replay containing a historical protected precursor | 3,594 |
| Ambiguous inverse decomposition | 9,172 |
| Ambiguous forward site class | 122 |
| Repeated-program adapter required | 12,002 |
| No exact reconstruction under the related transform | 5,648 |
| Missing role or multiplicity metadata | 351 |
| No related executable program available | 110,532 |
| **Total** | **149,736** |

The **3,594** protected-component rows must be excluded from a future training view even
though their complete products passed the earlier identity guard. The remaining rows do not
automatically qualify: unresolved decompositions can hide further protected precursors.
Recovered components and historical fold decisions are recorded in `train_checks.jsonl.gz`.
No conflicting structure mappings were observed among the source labels in the unique replay
subset; this says nothing about the unresolved population.

Even the 8,315 clear related-transform replays do not establish equivalence to the source's
complete v8 programs. Their evidence basis is computed transform consistency only. None enters
an L1 training objective yet; no executed synthesis, precursor-route closure or procurement
claim follows from these checks.

Related-transform reconstruction coverage is **11,909/149,736** unique, unambiguous replays,
including the protected-precursor cases. Source-program qualification coverage remains **0/23**.
Chemical decomposition precision is **not established**: there is no independent, complete v8
component/site truth set against which to score these proposed decompositions.

## Holdout metadata

The split check reads opaque labels and frozen signatures, not held-out molecular graphs.
It does not treat the labels as chemical identity hashes.

- Of 7,002 `unseen_precursor_identity` rows, 5,942 have complete role labels and at least one
  unseen label. The remainder cannot be verified from these labels alone.
- Twelve held-out Ugi rows carry an empty `precursor_ids` list. Their exact IDs and reasons are
  preserved in `split_label_audit.json`; they are not counted as successful checks.
- Of 6,327 `unseen_exact_combination` rows, 6,089 have complete role labels. All 6,089 use
  provider-TRAIN-seen labels, but 26 acquire an unseen label after FORGE's historical product
  exclusions. The original assignments remain intact; an eventual FORGE evaluation must report
  the changed eligibility rather than silently retain the original panel claim.
- The source-study overlap caveat in the source README remains applicable. This audit does not
  independently requalify study isolation or regional-topology signatures.

## Missing source contract

On 2026-09-18 the advertised upstream branch `dataset/source-corrected-corpus-v3` resolved to
commit `9a4e5884720b883918052db0733a316d0e47caaa`. Its published tree contains v5 exports, not
the v8 registry, enumeration programs or the split audit referenced in the delivered README.
The Drive handoff contains product graphs and the decomposition key but no executable v8
program definitions or component-ID-to-structure lookup.

To complete qualification, recover the exact v8 mapped transforms/programs with stage order,
reactive sites, repeated-role coupling and multiplicity. Component structure/site provenance
is also needed where exact decomposition remains ambiguous. Then replay every admitted row,
apply global precursor-identity protections, retain unresolved rows outside training, and
rebuild balanced training weights only over qualified rows. No raw family-count weighting,
gate relaxation or automatic training launch is authorized by these diagnostic receipts.

The preserved results are `results/phase1/compose_lipid_v8_pretraining_v1/` and
`results/phase1/compose_lipid_v8_representation_v1/`. Test reports and remaining repository
validation limits are recorded separately in
`results/phase1/compose_lipid_v8_pretraining_validation_v1/`.

## Validation outcome

- All **69** focused import, qualification, representation, decoder and architecture checks pass.
- All **38** historical pipeline regression checks pass. Archived source authentication remains
  exact on path and digest; new execution still requires matching active source bytes.
- All **30** vendored assets verify. All 15 touched Python files pass Black and Ruff; whitespace
  checks pass. The pre-training receipt and representation artifacts are authenticated.
- Final repository-wide tests: **2,630 passed, 148 failed, 21 setup errors, 92 skipped/xfail**.
  Every remaining failure/error identity appeared in the previous full report. This is an identity
  comparison, not a complete causal diagnosis of those failures. The initial run's eleven source
  authentication regressions are resolved without changing historical results or data/config pins.

Repository-wide Phase 1 definition of done is **not met**, and scientific training qualification
is also incomplete. Both limitations remain explicit. The validation result SHA-256 is
`e5929024154edbfd02d27296f87fe0f8370255428cefae866fc28c2a907ab703`.

## TRAIN-only component consistency follow-up (2026-09-18)

The subsequent check consumes the original pre-training receipt and re-enumerates only its
ambiguous fixed-arity cases with the unchanged registry adapters. It retains every inverse
candidate, including candidates with more than one forward product. A component label is scoped
by family and role. Candidate structures must agree wherever that scoped label repeats in TRAIN.
No held-out graph or held-out component label enters the inference.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_components \
  --config configs/multireaction/compose_lipid_v8_components_v1.json \
  --output-dir results/phase1/compose_lipid_v8_components_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_components \
  --verify results/phase1/compose_lipid_v8_components_v1/result.json
```

The consistency solver intersects candidate supports until stable. It never chooses an arbitrary
candidate. A contradiction holds the entire connected group of products. Nonempty local supports
alone do not prove global consistency: only groups with exactly one candidate for every product
receive a demonstrated common assignment. Even that assignment remains conditional on the source
labels denoting stable component identities; it is not independent chemical truth or source-program
qualification. Tests include an unsatisfiable cycle with nonempty local supports.

| Outcome across all 149,736 protected TRAIN inspection rows | Rows |
| --- | ---: |
| Component assignment with unique forward replay; source-program qualification still missing | 11,051 |
| Protected precursor: quarantine in the derived view | 5,430 |
| Source-label / related-transform candidate conflict | 4,722 |
| Other unresolved programs or components | 128,533 |

All **4,450** previously ambiguous aliphatic `reductive_amination` rows acquire a unique common
assignment under the label assumption. Of these, **1,714** contain protected precursors; **2,736**
remain conditional transform-consistency leads. All **4,722** previously ambiguous
`aryl_reductive_amination` rows belong to contradictory constraint groups and remain unresolved.
This is not a reductive-amination substructure hit-rate claim.

The quarantine count increases from **3,594 to 5,430**. Besides the 1,714 newly resolved aliphatic
cases, **122** Ugi cases already had unique inverse components but ambiguous forward sites; their
components also require protection. Their forward ambiguity is retained in the candidate ledger,
not cured by the exclusion decision. Original corpus assignments, source flags and historical
guards remain unchanged. `admission_view.jsonl.gz` assigns every row either quarantine or hold,
with zero training weight and `training_admitted: false`. Unresolved rows may hide further protected
precursors, so neither this view nor the earlier vocabulary is a final qualified training dataset.

Unique inverse assignments with unique forward replay cover **16,359/149,736** rows under the
shared-label assumption, including protected precursors. Another 122 have unique inverse assignments
but ambiguous forward sites. **Decomposition precision remains unestablished**, and source-program
qualification remains **0/23**. No model training or generation occurred.

The result SHA-256 is
`55ff93818a30a3d6e2ddfda4145617ac9127c36de8a3a395b9beaaf909cedfa8`.
Verification authenticates the source chain and reruns the consistency/exclusion calculation over
the saved candidates. It does not independently adjudicate the chemistry or rerun RDKit enumeration.
Candidate enumeration remains attributable to the pinned implementation, registry and TRAIN inputs.

### Concrete source-contract gap

`results/phase1/compose_lipid_v8_components_validation_v1/conflict-witness.json` records two TRAIN
products sharing the same `head_id`, whose proposed head sets are disjoint under the one-event
related transform. Run the adjacent `reproduce_conflict.py` to reproduce this witness. A partial
reaction program can leave already-attached exterior material in a proposed head; the observed
conflict does not itself distinguish that explanation from incorrect or differently scoped labels.
It does not establish defective source chemistry.

The published branch remains at `9a4e5884720b883918052db0733a316d0e47caaa`. The shared Drive folder
still lists the same seven artifacts. Refetched `README.md` and `family_decomposition_key.json`
are byte-identical to the imported copies. The source refresh is recorded in
`results/phase1/compose_lipid_v8_components_validation_v1/source-lookup.json`.
The exact request for the remaining source definitions and ambiguous-row evidence is in
`docs/COMPOSE_LIPID_V8_SOURCE_CONTRACT_REQUEST.md`.

### Follow-up validation

All **58** focused checks pass, including **12** new component-consistency tests. All **30**
vendored assets verify; Black, Ruff and whitespace checks pass. The full run reports **2,642
passed, 148 failed, 21 setup errors, and 92 skipped/xfail** across 2,903 cases. Its failure/error
identities exactly match the prior full run; this is not a complete diagnosis of their causes.
No new component test fails. The global Phase 1 definition of done remains unmet.

Logs, exact compressed full reports, source hashes and the baseline comparison are preserved in
`results/phase1/compose_lipid_v8_components_validation_v1/result.json`, SHA-256
`7ba5573c6e24aac7959fcd4eb75b7b80026eaa05775fcb3a1b39bda4a9cc8c07`.
Training qualification remains incomplete independently of the repository test failures.

## Training-readiness goal: historical replay portability (2026-09-18)

The active objective covers all **23 v8 reaction-program families**: complete source-qualified
programs, exact family reconstruction, precursor protections and holdouts, a final balanced TRAIN
view, full vocabulary/size support, and passing repository verification. It does not launch training.

The first readiness repair authenticates historical Python inputs by their original logical path
and exact digest through the existing move ledger or frozen source archive. Historical result
comparison normalizes only authenticated `inputs.*.path` locations; every scientific field,
input digest and output ledger check remains exact. Data files cannot use the source-relocation
fallback. Four pre-edit implementations were preserved in the content-addressed source archive.
No archived module was executed and no historical result or evidence pin was rewritten.

This recovers **17** failing historical checks. Every test and helper function in the six affected
test modules is unchanged at the AST level; only input bindings and imports changed. All **15** new
provenance tests pass, including digest tampering, incorrect logical identities, malformed archive
entries and changes to scientific result fields. The focused run has **43 passed and 3 failed**;
all three remaining failures require the absent original third-wave supplier page at its frozen
digest. A newly downloaded page cannot substitute for that evidence.

All **30** vendored assets verify. Black/Ruff pass for the 12 implementation/test files and the
report summarizer. Full suite: **2,674 passed, 131 failed, 21 setup errors, 92 skipped/xfail**, across
2,918 cases. There are no new failure/error test identities relative to the prior component audit;
all v8 tests pass. The remaining failures are not all diagnosed. Global Phase 1 definition of done
is still unmet.

The local COMPOSE checkout remains clean at `a27215c4507ccf40cb6940ad5257a1c2c083c058`, with no
additional v8-named tracked export. The v8 program-definition and precursor-qualification holds
remain in force; these test repairs admit no training rows. Logs, source hashes, function-preservation
checks, exact baseline comparison and a runnable summarizer are in
`results/phase1/compose_lipid_v8_readiness_portability_v1/`. The result SHA-256 is
`0bf1a5f7917d46813791e763e0e2eef2fae4485546dcf46498ef045b1188ab37`.

## Training-readiness goal: complete Passerini program (2026-09-18)

Recovered the primary [PNAS article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11804478/) and its
supplement for the 144-member Passerini library (DOI `10.1073/pnas.2409572122`). Article pages 2–4,
supplement pages 2–6 and the full-resolution first figure were visually inspected. Asset hashes,
retrieval failures, locators, procedure context and source conflicts are preserved in
`results/phase1/compose_lipid_v8_passerini_source_v1/adjudication.json`.

The frozen registry reconstructs Passerini product connectivity, but its two oxygen origins differ
from the published Mumm-rearrangement mechanism. A separate registry variant swaps only product
atom-map labels 3 and 5: the aldehyde oxygen becomes the ester bridge and the acid hydroxyl oxygen
becomes the amide carbonyl oxygen. This is a mechanistic atom-origin convention, not isotope-tracing
evidence. The frozen registry, historical results, reactant templates, role policies and product
connectivity remain unchanged. The new variant is used only by this explicit audit.

The complete program consumes one amine-bearing acid head, one aldehyde tail and one isocyanide
tail. It requires unique constitutional inverse and forward reconstruction with no bounded-search
truncation, retained tertiary amine in the complete acid head, and element/hydrogen/charge balance.
The amine check is structural; it does not establish pKa or biological activity. Two independently
transcribed source controls, H1A1B1 and H2A4-Ole, pass. Atom-order and SMILES serialization changes
leave the disposition unchanged.

| Passerini TRAIN inspection outcome | Rows |
| --- | ---: |
| Complete program checks pass | 7,007 / 7,007 |
| Contain protected historical precursors | 1,728 |
| Clear of the currently known historical precursor set | 5,279 |
| Admitted to training | 0 |

This gives **1/23 source-family programs** a qualified single-event transform on its inspected
TRAIN population. It does not retrospectively change the earlier 0/23 receipts. The 5,279 rows
still await global precursor protection, chemical holdout qualification and final dataset assembly.
Computed precursor identities are not proof of the provider's component-ID mapping. Coverage is
7,007/7,007; **decomposition precision over the corpus remains uncalibrated**. Two source controls
cannot estimate that precision. The v8 acid heads extend beyond the nine drawn primary-library
heads, and none inherits experimental execution or efficacy labels.

The supplement contains product-heading/reactant inconsistencies in H2A4B3/B4 and A4B4-S3/S4.
Those entries are preserved as conflicts and are not exact executed-route controls. No L2 or
procurement evidence is admitted by this audit. Its disposition is `admit_transform_consistency`.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_source_program \
  --config configs/multireaction/compose_lipid_v8_passerini_program_v1.json \
  --output-dir results/phase1/compose_lipid_v8_passerini_program_fresh

uv run python -m experiments.phase1.multireaction.compose_lipid_source_program \
  --verify results/phase1/compose_lipid_v8_passerini_program_v1/result.json
```

Use a fresh output directory for execution. Verification authenticates source/implementation pins,
recounts the full saved ledger against the prior TRAIN inspection and replays the primary controls;
it does not rerun all 7,007 chemistry calculations. A separate fresh execution did rerun them and
produced identical registry and compressed-ledger hashes, summaries and source-control outcomes.
Receipt checks reject removed implementation entries, changed evidence tiers, false completion
flags, dropped/duplicated TRAIN rows, altered components and hidden precursor protection.

Validation: **112 focused checks pass**, including 34 new source-program tests; all 30 vendor
assets verify, and Black/Ruff pass. The full run reports **2,708 passed, 131 failed, 21 setup
errors and 92 skipped/xfail** across 2,952 cases. The failure/error identities match the previous
portability run exactly, with no new v8 failures. Existing failures remain unresolved; global
Phase 1 completion is not met. Full reports, input hashes, the 23-family status table and a
runnable summarizer are in `results/phase1/compose_lipid_v8_passerini_validation_v1/`. Its result
SHA-256 is `3440f5267e5b22d5cedeed52ca29bc3ba4aab65d0a63b5f1f68fb0ffa613aed6`.

## Training-readiness goal: repeated A3 reconstruction (2026-09-18)

Recovered the earlier v5 precursor dictionary and construction metadata from COMPOSE commit
`9a4e5884720b883918052db0733a316d0e47caaa`. These are additional source evidence, not a replacement
v8 corpus. Opaque component IDs supply candidate structures only; their spelling is not a
constitutional hash. Molecular parsing is restricted to IDs referenced by protected v8 TRAIN.
The acquisition manifest and upstream documentation are retained with the A3 validation receipt.

The [Han primary article](https://www.nature.com/articles/s41551-024-01267-7), author manuscript
and publisher supplement support single-event and repeated amine–aldehyde–terminal-alkyne
coupling. Main Figs. 1–3, SI PDF pages 5–6 and neighboring pages were visually inspected. Two
independently transcribed controls, `11(aB)2` and `31hP-M`, reproduce their drawn constitutions and
reported formulas. The controls establish computed transform consistency; they do not add
experimental outcome labels. The SI prints inconsistent CuCl amount/equivalent values for
`31hP-M`; both values remain in the source-conflict record.

A new, separately pinned source registry specifies the event and reactive-site queries. Frozen
registries are unchanged. A generic executor retains the declared event count, exhaustively
enumerates every permitted site at each stage and never prunes forward paths to fit the target.
It requires a unique final constitutional product, exact inverse replay conditioned on the declared
fixed side components, and full element/hydrogen/charge balance with one water molecule per event.
Any enumeration bound invalidates the check. Ambiguous intermediate states can converge after full
substitution; tests cover four and six events as well as the two primary controls.
The first new registry query also matched formic-acid/formamide carbonyls. Version 2 narrows it
to genuine aldehydes and formaldehyde; negative controls preserve the original failure. All
observed A3 chemistry outcomes are unchanged. The first diagnostic artifacts and their exact
implementation snapshots are retained, but are superseded by program v3 for current verification.

| Protected A3 TRAIN outcome | Rows |
| --- | ---: |
| All rows accounted for | 5,556 |
| Complete candidate tuples available | 4,263 |
| Exact, unique computed reconstruction | 2,478 |
| Target recovered alongside another distinct product | 1,785 |
| Missing candidate component lookup | 1,293 |
| Successful reconstructions with known protected precursors | 842 |
| Admitted to training | 0 |

All 1,785 competing-site cases produce exactly two distinct final products. Selecting the saved
target from that pair would conceal uncertainty in the attachment assignment. Missing roles occur
in 602 head lookups and 777 alkyne lookups; these overlap within the 1,293 unresolved rows. Every
declared aldehyde ID resolves in the earlier dictionary. The remaining 1,636 consistent rows are
only clear of the currently known protected-precursor set; global protection is still incomplete.

There is also a source-key conflict: v8 labels Han Libraries 3–5 as two ordered asymmetric events,
whereas primary Fig. 3e and Methods B show one event for amine 31. The original key is preserved.
No row is assigned to a corrected paper architecture by resemblance, and no physical reaction order
is inferred from a deterministic serialization of commuting events. Consequently this audit does
**not** increase the complete-family qualification count beyond 1/23. Decomposition precision across
the v8 population remains uncalibrated; exact reconstruction is a computational check, not chemical
selectivity or evidence of experimental execution.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_source \
  --config configs/multireaction/compose_lipid_v8_a3_program_v3.json \
  --output-dir results/phase1/compose_lipid_v8_a3_program_fresh

uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_source \
  --verify results/phase1/compose_lipid_v8_a3_program_v3/result.json
```

The verifier authenticates source and implementation pins and recomputes every TRAIN replay,
including missing-component and protected-precursor accounting. Independent verification passed.
The program result SHA-256 is
`c982b8f0245b5b71133074b9f12a6380aaa3df021900f8aed4268a5c107829ec`; the deterministic
compressed ledger SHA-256 is
`d41b2bdf2c9b98e3bf7b1ecd561ee2b44fb0b9ec8de9736ac73ee2dfc93957c5`.

The earlier v5 export also exposes a cross-version identity issue: provider target IDs are not
stable across all families. Comparing exact constitutional strings across all 149,736 protected
v8 TRAIN rows finds 4,642 overlapping v5 products, including **376 prior validation/test products**
that must be excluded from the final TRAIN view. These are certain overlaps; differing strings do
not prove chemical disjointness across canonicalization versions. No heldout v8 graph was parsed.
The runnable audit and per-target exclusion evidence are in
`results/phase1/compose_lipid_v8_a3_validation_v1/prior_identity.json`. Its SHA-256 is
`867a5d57925dd35385f0943c1394b2dd997a2219912062ece8cd00817a3f8744`.

A3 contains 71 of those overlaps and two prior validation products. Both already have protected
precursors; one reconstructs consistently and the other has a missing component lookup. Program
v3 records these overlaps by constitutional strings, preserves the original provider IDs and
explicitly leaves nonmatches unresolved. Its chemistry, precursor assignments and ambiguity
findings equal those in v2; v1 and v2 have byte-identical chemistry ledgers.

Final focused validation: **132 passed**, including 45 new tests; all 30 existing vendor assets,
the newly pinned source registry, Black/Ruff and whitespace checks pass. The development full run
reports **2,741 passed, 131 failed, 21 setup errors and 92 skipped/xfail**, with failure/error
identities identical to the preceding Passerini run and no new audit failures. That run started
before the final aldehyde and cross-version identity fixes; the final focused run covers those
fixes. No successful full-suite or final full-source-snapshot claim is made. The complete
23-family status, snapshots and test evidence are in
`results/phase1/compose_lipid_v8_a3_validation_v1/result.json`, SHA-256
`88a386b95c5ee9ce67c7251b1001b8c6d344b8a085b588c721c99d7c2036f1b1`.

## Enforced product exclusions and Ugi-3 source events (2026-09-18)

The prior-product finding is now an enforced preparation view. It accounts for all 149,736
previously protected TRAIN rows and excludes **376 prior validation products**, leaving **149,360**
across all 23 families for further program qualification. Exact source-string overlaps propagate
to every TRAIN alias of the authenticated constitutional identity, including aliases in another
family. Opaque provider IDs do not establish identity. Unmatched strings remain unresolved, and
the reader refuses to expose a training iterator. Original source rows, admission flags and splits
are unchanged. This closes the known exclusion defect, not global holdout qualification.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_protection \
  --config configs/multireaction/compose_lipid_v8_product_protection_v1.json \
  --output-dir results/phase1/compose_lipid_v8_product_protection_fresh

uv run python -m experiments.phase1.multireaction.compose_lipid_protection \
  --verify results/phase1/compose_lipid_v8_product_protection_v1/result.json
```

The product-protection result SHA-256 is
`5afcbadb23d35d077db3298e7d9d65e7fff913f8256f0a6a7e7552147c67e12b`.

For the single-event Ugi-3 invariant, the
[AGILE article and supplement](https://www.nature.com/articles/s41467-024-50619-z) and
[Chen PNAS article and supplement](https://doi.org/10.1073/pnas.2309472120) provide three independent
controls: H9, R6 and iso-A11B5C1. Their transcribed products reconstruct uniquely in both directions
and reproduce formulas C36H69N3O3, C35H73N3O and C41H79N3O3. The reacting head nitrogen must be a
primary NH2, including the N-aminopiperidine head in H9; a requirement that its neighbor be carbon
would incorrectly reject that source control. The aldehyde query excludes formate/amide carbonyls.

The executor uses the unchanged frozen Ugi transform and multiplicities. Additional predicates
check the actual mapped precursor atom that formed the product bond. They never remove competing
forward products to manufacture uniqueness. Full atom, hydrogen and charge inventories must agree;
net balance does not establish isotope origins. The assembly-introduced amide oxygen retains the
existing solvent/water convention. Both sources use an acidic phosphorus catalyst; there is no
carboxylic-acid reactant component.

The SI procedures and conflicts are preserved in
`results/phase1/compose_lipid_v8_ugi3_source_v1/adjudication.json`. AGILE article/SI catalyst wording
differs, and Chen SI page 20 contains an inconsistent isolated-amount unit and product label.
Computed controls do not validate those experimental details or add executed-route labels.

| Ugi-3 preparation outcome | Rows |
| --- | ---: |
| Original protected TRAIN | 2,967 |
| Excluded prior validation products before replay | 138 |
| Remaining rows checked | 2,829 |
| Unique, exact, source-consistent program | 2,577 |
| Exact inverse tuple but competing forward products | 122 |
| No exact inverse | 130 |
| Consistent programs with known protected precursors | 1,514 |
| Consistent and clear of currently known exclusions | 1,063 |
| Admitted to training | 0 |

There are no conflicting family/role/component-label assignments among the uniquely inferred
tuples. Every provider design lane and original precursor label is retained. This qualifies the
shared one-event constitutional invariant on the stated subset. Assignment to the separate
published reagent banks and architecture strata remains unresolved; no AGILE/PNAS cross-grid is
created. Consequently the complete-family count remains **1/23**. Decomposition precision and
experimental selectivity across the population remain uncalibrated.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_source_event \
  --config configs/multireaction/compose_lipid_v8_ugi3_program_v1.json \
  --output-dir results/phase1/compose_lipid_v8_ugi3_program_fresh

uv run python -m experiments.phase1.multireaction.compose_lipid_source_event \
  --verify results/phase1/compose_lipid_v8_ugi3_program_v1/result.json
```

Verification independently replays every row and primary control, as well as recomputing the
enforced preparation view. The Ugi-3 result SHA-256 is
`7d4d247fdacf590a49737771e77f3bfbc1492da92253f5d813c3f8cde3be5f6a`.
The all-family training-readiness goal remains active: global precursor protection, qualified
holdouts, remaining source programs/architectures, the final balanced dataset and repository-wide
validation are still required. No training was launched.

Validation for this increment: **164 focused checks pass**, including 32 new tests; all 30 vendor
assets and nine touched Python files' Black/Ruff checks pass. The verifier reproduced every Ugi-3
row, control and known exclusion. The full suite was not repeated for this increment; its latest
retained run still has 131 failures and 21 setup errors and does not validate the new source snapshot.
The reproducible 23-family status and validation receipt are in
`results/phase1/compose_lipid_v8_source_event_validation_v1/result.json`, SHA-256
`7a6b917ba4b5566f24588e920d3587aa5daae15f88f8a8ca2220ce06c9f57e64`.

## Repeated aza-Michael program qualification

The Akinc primary article and supplementary information now provide independently transcribed
controls for the shared acrylate and acrylamide addition program. The new registry overlay leaves
the frozen parent registries unchanged, keeps the N/O interfaces separate, and restricts the nucleophile to
non-acyl neutral primary/secondary amines. Each acceptor must have one qualified handle and is
preserved as a complete component. Free competing amines in the acceptor and competing acceptors
in the head are excluded by explicit role policies.

The inverse enumerates complete precursor tuples through every declared event before selecting
any component identity. It requires one repeated acceptor identity, then independently enumerates
all forward products and checks element, hydrogen and charge balance. Any search truncation,
competing tuple or competing product excludes the row. Missing occupancy is excluded without
assuming that every N-H site reacted.

| Family | After prior-product exclusions | Exact complete programs | Missing program metadata | Exact programs with known protected precursors | Clear of currently known exclusions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aza-Michael acrylate | 5,376 | 5,065 | 311 | 2,516 | 2,549 |
| Aza-Michael acrylamide | 4,879 | 4,879 | 0 | 2,503 | 2,376 |

All rows remain **training-unqualified**. These counts cover the enforced preparation view:
24 acrylate rows were already excluded as prior validation products. No component-ID conflicts
were observed within either family among these inferred tuples. This does not prove cross-family
identity consistency or complete precursor holdout protection.

Source controls 1L10, 1L12 and 1N12 reproduce their independent structures and neutral formulas.
The reported five-tail 98N12-5(1) isomer correctly has two possible forward constitutional products
under a program lacking attachment sites. That ambiguity control rejects an unsupported exact
attribution; it does not contradict the source's isolation of a particular isomer. Source feed
ratios are also kept separate from actual occupancy. The supplement's fractional vinyl resonances
for 1N12 are preserved as an analytical caveat, without a purity claim.

These controls support **computed transform consistency**, not experimental execution or calibrated
corpus-wide chemical precision. Source architecture and precursor-bank assignments remain open,
including the other acrylate subfamilies. No global amine/acceptor Cartesian pool is constructed.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_inverse \
  --config configs/multireaction/compose_lipid_v8_michael_program_v1.json \
  --output-dir results/phase1/compose_lipid_v8_michael_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_inverse \
  --verify results/phase1/compose_lipid_v8_michael_program_v2/result.json
```

The current program result SHA-256 is
`cd7d3389f1ea2ea95add131164bf5bdab8dfcd8cc0cee14cf33eefc79896a841`.
The first completed run and its implementation snapshot are retained. Tightening authentication
of overlapping registry/adjudication asset names left every compared scientific field and the
entire ledger byte-identical. The current verifier independently replays every row and source
control and recomputes the protected preparation view.

Validation: **199 focused tests pass**, including 35 new tests. All 30 vendor assets verify.
The increment's logs, replay comparison and all-family status are in
`results/phase1/compose_lipid_v8_michael_validation_v1/`. The repository-wide suite was not repeated;
the last retained full run still reports 131 failures and 21 setup errors. The training-readiness
goal remains active across all 23 families, and no training was launched.
The validation receipt SHA-256 is
`bbba291412b1edb7fb518b578ac129b632dcf2400b14476561aedecabb25e27f`.

## Source-scaffold reductive amination (2026-09-18)

The [Jiang primary source](https://doi.org/10.1038/s41565-023-01548-3) and
[Xue primary source](https://doi.org/10.1038/s41467-024-45422-9), including visually inspected
supplements, now support independent final-product and coupled-precursor controls. Jiang's
IR-19-Pyrazole and IR-117-17 and Xue's 5-A2-7b2 reproduce exact constitutional products and their
neutral formulas. Xue's A3-7b2 independently checks the triacylated aryl aldehyde precursor.
Source locators, reaction conditions, analytical evidence and conflicts are pinned in
`results/phase1/compose_lipid_v8_reductive_source_v1/adjudication.json`.

A separately qualified registry overlay requires the complete Jiang hydroxyaldol-derived
aldehyde or Xue's 3,5-diacylated/2,4,6-triacylated aryl aldehyde. Fragment checks retain the complete
precursor, verify the source core and identical permitted acyl arms, and reject incorrect ring
positions, unequal arms and additional aldehydes. The frozen parent registries are unchanged.
These are source-scaffold and full reconstruction checks; no general C-N motif hit rate is used.

| Family | Preparation rows | Exact programs | Missing labels | Forward ambiguity | Exact with protected precursors | Clear of known exclusions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reductive amination | 5,810 | 5,800 | 10 | 0 | 1,987 | 3,813 |
| Aryl reductive amination | 4,724 | 4,558 | 13 | 153 | 2,278 | 2,280 |

Twelve prior validation aryl products were excluded before this analysis. The exact aryl subset
comprises 2,038 A2 and 2,520 A3 core assignments. This is a computed structural assignment, not
proof that each head/tail pair was experimentally tested. Global precursor protection, holdouts
and final balanced TRAIN admission remain open; **no training rows are admitted**.

The net inventory check accounts for one oxygen difference per reductive-amination event.
Chemically, aldehyde plus amine and reducing equivalents gives product plus water; the net
inventory entry is not a literal monatomic-oxygen byproduct. Source condensation/reduction order
is recorded, but intermediate-specific selectivity is not established by a net transform.

An additional diagnostic examines all 153 aryl exclusions. At two events, every row has one
complete inverse tuple and balanced reconstruction with the target among **two** forward products.
The recovered heads are consistent within H1, H2, H3 and H11. A one-event cut had left a partially
reacted head whose structure varied with the aldehyde, causing component-label conflicts.
Two events therefore explain the identity discrepancy but do not remove forward ambiguity.
The source's imine-formation stage before reduction requires an explicit stage/site contract;
the diagnostic neither changes the key nor admits these rows. Full results and a runnable probe
are in `results/phase1/compose_lipid_v8_reductive_validation_v1/multiplicity-probe.json`.

Source inconsistencies remain visible: Jiang's intermediate numbering and one acyl label differ
between prose and drawings; Xue's SI Fig. 12 input drawing and an acid name conflict with the
reported A3 products. The controls use the independently inspected final structures. No exact
L2 route or experimental success labels are added. Corpus-wide decomposition precision and
experimental selectivity remain uncalibrated.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_scaffold_event \
  --config configs/multireaction/compose_lipid_v8_reductive_program_v1.json \
  --output-dir results/phase1/compose_lipid_v8_reductive_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_scaffold_event \
  --verify results/phase1/compose_lipid_v8_reductive_program_v1/result.json
```

The independent verifier reproduced every row and primary control, including known product and
precursor exclusions. Program result SHA-256:
`91bf8f3d068d86a70c1f25e179f539bb9035ab2636c31a05b3701d1bb9d3dfeb`.
All **225 focused checks** pass, including **26 new tests**, and all 30 vendor assets verify.
The completed full run covers the final implementation/test snapshot: **2,846 passed, 131 failed,
21 setup errors, 92 skipped/xfail**. Failure/error identities match the previous full run exactly;
none are in the new audits. Logs, JUnit reports, source pins, failure comparison and all-family
status are preserved in `results/phase1/compose_lipid_v8_reductive_validation_v1/`.
Validation receipt SHA-256:
`702c5be0a264fe1af8cd552fadd6d6e98f85bdf65aa40f74fc0dc9eabe4d178b`.
Training remains disabled and the all-23-family readiness goal remains active.

The initial staged-family source packet was saved in
`results/phase1/compose_lipid_v8_staar_source_v1/`. Two independent STAAR drawing transcriptions
passed formula, theoretical ion-mass and additive inventory checks. That receipt records source
preparation only; the later complete stage-program qualification is recorded below.
Source conflicts are retained and no biological measurements are imported as FORGE supervision.

## Historical manifest recovery and latest full validation

Four legacy benchmark setup errors are now resolved by authenticating the exact archived source
bytes at their reviewed locations. This applies only to implementation evidence. Data paths,
input hashes, forbidden-input checks and frozen benchmark files are unchanged; archived code is
not executed. Fresh result metadata separately records the executing module and authentication
helpers.

`results/phase1/compose_lipid_v8_manifest_recovery_v1/replay.py` reproduces all three frozen
scientific payloads byte-for-byte and every original result field exactly. Its 42 focused tests
pass, including ten new provenance regression tests. All 30 vendor assets verify.

The latest completed full run reports **2,860 passed, 131 failed, 17 setup errors and 92
skipped/xfail** across 3,100 cases. Exactly the four manifest setup errors are resolved; there are
no new failure/error identities compared with the preceding full run. The implementation/test
snapshot remained unchanged throughout execution. Full evidence and a runnable recount are in
`results/phase1/compose_lipid_v8_manifest_recovery_v1/result.json`, SHA-256
`1f488acd47ae2908a88e0b8235e467b709bf72cfd7e74d8f5355dd4fe9ff54e1`.
Repository-wide completion and training readiness remain unmet. No checks were skipped or relaxed
to obtain this improvement.

## Complete STAAR sequential program (2026-09-18)

The Peña primary article and visually inspected SI now support a complete executable program:
thiolactone aminolysis exposes a thiol, then thiol addition to an acrylate completes the product.
The source mixes thiolactone and acrylate before adding amine; reagent addition order is recorded
separately from covalent stage order. Independent A4B2C3 and A4B2C8/CP-LC-0729 drawings reproduce
exactly, including their neutral formulas and theoretical protonated masses. Source locations,
conditions and the B2/B3 prose/drawing discrepancy remain in the pinned evidence packet.

The executor retains all intermediate and final outcomes, enumerates complete inverse tuples,
and checks atom, hydrogen and charge balance at each stage and across the complete program.
Ambiguous sites, truncated searches and incomplete single-stage explanations fail. The neutral
head must retain a nonacyl basic-site candidate after assembly; this is a structural condition,
not a pKa measurement. The separate overlay preserves the frozen parent registries.

All **9,314** enforced preparation products reconstruct uniquely through both stages, with no
component-label conflicts or component-contract exclusions. The earlier protection layer had
already excluded **52** prior validation products. Known protected precursors occur in **2,792**
of the exact programs; **6,522** are clear of those known exclusions. These counts describe
computed consistency and known protection only. Global precursor protection, holdouts and the
final balanced TRAIN dataset remain unqualified, and **no training rows are admitted**.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_sequential \
  --config configs/multireaction/compose_lipid_v8_staar_program_v1.json \
  --output-dir results/phase1/compose_lipid_v8_staar_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_sequential \
  --verify results/phase1/compose_lipid_v8_staar_program_v1/result.json
```

Independent verification reproduces every source control, program row and exclusion. Program
result SHA-256: `21eec94e2f56de024c22501705f33332a5fbc081729a611ae79ce0b89bc4a17a`.
The controls and adversarial tests qualify the declared transform, not corpus-wide chemical
precision, experimental selectivity or original source-bank membership. Constitutional identity
remains stereo-free; no biological measurements or experimental success labels are imported.

The STAAR full run is complete: **2,887 passed, 131 failed, 17 setup errors and 92 skipped/xfail**.
All **252 focused tests** pass. Its validation receipt is now finalized at
`results/phase1/compose_lipid_v8_staar_validation_v1/result.json`, SHA-256
`053a04b191e0bdfa0c3c017acfa7627ad77854ae724e4d82b1a2167c7560be92`.

## Cross-family precursor identity audit (2026-09-18)

This audit consumes the pinned Passerini, Ugi-3, aza-Michael, reductive-amination and STAAR
program ledgers. It recomputes each distinct precursor's canonical identity, checks component
labels against the original TRAIN metadata, and reapplies historical protections by molecular
identity across families and roles. A source label is interpreted only within its observed
family and role. Unknown labels never count as evidence of chemical novelty or disjointness.

All **149,736** original inspection records remain accounted for, including **376** prior-product
exclusions. Among the **149,360** preparation records, the consumed ledgers assess **39,939**
records from seven families: **39,200** are exact programs and **739** retain their original
program exclusions. The remaining **109,421** preparation records lack the complete, unambiguous
program evidence consumed by this audit. This includes A3: its earlier conditional diagnostic
is preserved, but its source conflict and incomplete inverse qualification cannot be promoted.

| Family | Exact program records | Known protected precursors | Clear of known exclusions |
| --- | ---: | ---: | ---: |
| Ugi-3 | 2,577 | 1,514 | 1,063 |
| Passerini | 7,007 | 1,728 | 5,279 |
| Aza-Michael acrylate | 5,065 | 2,516 | 2,549 |
| Aza-Michael acrylamide | 4,879 | 2,503 | 2,376 |
| Reductive amination | 5,800 | 1,987 | 3,813 |
| Aryl reductive amination | 4,558 | 2,278 | 2,280 |
| STAAR | 9,314 | 2,792 | 6,522 |
| **Total** | **39,200** | **15,318** | **23,882** |

These exact records contain **11,613** distinct precursor constitutions. **687** occur in more
than one family and **648** occur under more than one role name. No conflicting family/role-scoped
labels occur within the exact subset. Shared identities are not themselves evidence of leakage:
their exposure must be assessed against the particular evaluation claim and partition.

The evaluation audit selects only metadata fields from the imported database; no provider
calibration or held-out graph is fetched or parsed. Of **39,950** evaluation rows, **5,984** have
every role resolved through the preparation dictionary and **3,308** have all role identities
present among the exact rows clear of known exclusions. Familiar precursors are permitted in
the provider's `unseen_exact_combination` panel; this audit does not conflate that panel with
precursor-identity generalization.

**None of the 7,002 `unseen_precursor_identity` rows has a fully resolved precursor tuple.**
Of these, **1,060** have missing or invalid role metadata. Consequently zero label-implied
contradictions does **not** establish a valid chemical-identity holdout. The unresolved labels
could denote aliases of known chemistry. Independent component structures and the source split
audit are still required to close that gap. The evaluation labels never extend the TRAIN
dictionary or change any source partition.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_precursor_audit \
  --config configs/multireaction/compose_lipid_v8_precursor_audit_v1.json \
  --output-dir results/phase1/compose_lipid_v8_precursor_audit_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_precursor_audit \
  --verify results/phase1/compose_lipid_v8_precursor_audit_v1/result.json
```

The output contains the complete per-product exclusion ledger, a cross-family identity index,
and every metadata-only evaluation decision. Independent verification redoes the joins,
identity checks and summaries; it consumes the existing pinned chemistry evidence and does not
claim a new chemical-precision calibration. Rehashed output changes, wrong identities, label
conflicts, attempted prior-product readmission and TRAIN/evaluation mixing fail closed.

**Zero rows are admitted to training.** The 23,882 records clear of known exclusions are an audit
subset, not the final balanced TRAIN dataset. Remaining source programs, unseen-precursor
identities, architecture assignments where unresolved, final weights and repository-wide tests
must close before training readiness. The existing training interface remains disabled.

Validation and the exact frozen-input recovery inventory are retained in
`results/phase1/compose_lipid_v8_precursor_validation_v1/`. The inventory identifies first failing
test preconditions and existing configuration SHA-256 pins; it is not an exhaustive dependency
graph. Missing scientific artifacts must be restored with their original bytes, without
substituting new supplier pages, inventing results or skipping checks.

The final validation has **282/282 focused tests passing**, including 30 new regression checks.
All **30** vendor assets and touched-file Black/Ruff checks pass; independent audit verification
reproduces every ledger and summary. The completed full run reports **2,917 passed, 131 failed,
17 setup errors and 92 skipped/xfail**, with exactly the previous failing/error identities and
an unchanged implementation/test snapshot. No existing test was skipped or weakened. Phase 1's
repository-wide definition of done is still unmet.

Audit receipt SHA-256:
`371cf8c61c9a631f364672e4162d22f3725cfd2b8bf111d33bd69873ced3812a`.
Validation receipt: `results/phase1/compose_lipid_v8_precursor_validation_v1/result.json`.
The recovery inventory records **48** missing paths from **106** first failing preconditions:
**42** artifact paths and **6** legacy source paths. Other failures require further diagnosis;
this inventory does not claim that restoring those paths alone will close the entire suite.

## Aldehyde Ugi-4 and expanded precursor audit (2026-09-18)

The newly recovered Nature Materials primary source (DOI `10.1038/s41563-024-01867-3`)
supports an independent 119-23 four-component control. Its drawn product, named product and
neutral formula agree. The pinned registry overlay derives atom queries from the existing
registries and adds the source-drawn acid-carbonyl/amine bond; it accounts for explicit net
water loss. Original registries and the atom-conserving source-event checker remain unchanged.
The separate condensation checker preserves complete unfiltered inverse/forward uniqueness,
source reactive-site witnesses and element/hydrogen/charge balance.

After 24 prior-product exclusions, **8,163 of 8,450** preparation products reconstruct exactly.
All **287** excluded inverses require three amine handles, beyond the inherited qualified
multiplicity of one or two. The diagnostic retains every inferred fragment and role rejection;
it does not expand the bound or claim that the underlying products are chemically invalid.
Known protected precursors exclude **4,314** exact records; **3,849** clear known exclusions.

The source packet is `results/phase1/compose_lipid_v8_ugi4_source_v1/`. Its README distinguishes
the final control from an earlier development check and records replay commands. The exact
program receipt is `results/phase1/compose_lipid_v8_ugi4_program_v1/result.json`, SHA-256
`ae5fbd1fe263e3dd0e6904e1ad4c6c605614da05541407b3d133f5b154e1c5e3`.
This supports computed aldehyde Ugi-4 consistency, not ketone Ugi-4, experimental bank
membership, measured synthesis success or complete L2/L3 dossiers. Positive controls and
adversarial tests do not establish corpus-wide chemical precision.

The v2 precursor audit consumes this result alongside the five earlier receipts. It assesses
**48,389** records across **eight of 23 families**: **47,363** are exact, **1,026** are excluded
by program checks, and the other **100,971** preparation records still lack complete program
evidence. Of the exact records, **19,632** contain known protected precursors and **27,731**
clear those known exclusions. The audit contains **11,674** distinct precursor constitutions,
including **706** shared across families and **659** shared across roles, with zero conflicting
family/role-scoped labels.

Metadata-only evaluation resolution rises to **7,460 of 39,950** complete tuples. Nevertheless,
**all 7,002 unseen-precursor tuples remain unresolved**, including 1,060 with malformed role
metadata. The v5 dictionary concordance diagnostic supplies no new admitted identities or
aliases and cannot close this gate. No provider evaluation graph was fetched or parsed.

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_condensation_event \
  verify --result results/phase1/compose_lipid_v8_ugi4_program_v1/result.json

uv run python -m experiments.phase1.multireaction.compose_lipid_precursor_audit \
  --config configs/multireaction/compose_lipid_v8_precursor_audit_v2.json \
  --output-dir results/phase1/compose_lipid_v8_precursor_audit_v2_replay

uv run python -m experiments.phase1.multireaction.compose_lipid_precursor_audit \
  --verify results/phase1/compose_lipid_v8_precursor_audit_v2/result.json
```

Both program and audit receipts pass independent replay. **315 focused tests** pass, including
33 new condensation checks; all 30 vendor assets verify. The full test result is retained in
`results/phase1/compose_lipid_v8_ugi4_validation_v1/`. No training rows or weights are admitted.
The remaining source contracts, holdout identities, final balanced dataset and repository-wide
definition of done remain open.

A separate amine/epoxide acquisition attempt recovered Love et al.'s article and generic scheme
but not a usable primary supplement for an exact control. It remains precedent-only, with the
failed responses and abstention retained in `results/phase1/compose_lipid_v8_epoxide_source_v1/`.

The completed full suite has **2,950 passed, 131 failed, 17 setup errors and 92 skipped/xfail**
across 3,190 cases. The failing/error identities exactly match the preceding precursor-audit
run; none was added or resolved. The implementation/test snapshot remained unchanged during
validation. These results do not satisfy the repository-wide definition of done.

An unsent metadata-only request now enumerates all 7,002 unresolved unseen-precursor records:
**2,174** distinct family/role labels from well-formed metadata need component structures, and
**1,060** records need role-metadata repair. It also requests the source split audit. The export
checks its joins and counts against the independently replayed audit and reads no evaluation
graphs. See `results/phase1/compose_lipid_v8_ugi4_validation_v1/holdout-request.json` and
`unresolved-holdout-request.jsonl.gz`. Reproduce it with:

```bash
uv run python results/phase1/compose_lipid_v8_ugi4_validation_v1/prepare_holdout_request.py
```

The expanded audit receipt SHA-256 is
`7508dde592a69264d6e0cf18e4abe333006ebeedbe9a552136b44b1c51c2fce7`.
The complete validation receipt is
`results/phase1/compose_lipid_v8_ugi4_validation_v1/result.json`.
