# Twelve-library FORGE extension

FORGE now has an exercised shared generation interface for all twelve reaction families declared by
`compose_lipid`. The 2026-09-14 bounded run produced train-novel, exactly reconstructable products
in every family: 750/768 valid connected outputs and 378/768 exact computed programs. Four families
retain precursor/neutral target scope; these counts do not establish twelve complete ionizable-lipid
generators or production-quality realism. See [the generation results](#learned-generation-across-all-twelve-libraries-2026-09-14)
for the per-family breakdown, retained checkpoint and runnable commands.

The subsequent [repeat audit](#repeated-program-failure-audit-2026-09-14) isolates 208 of 334 failed
trained attempts in repetition-capable families to the identical-co-reactant requirement. These
mixed-reactant diagnostic matches remain unaccepted; the original generation metrics are unchanged.

The [precursor-reuse conditioning pilot](#precursor-reuse-conditioning-pilot-2026-09-14) did not
meet its preservation rule. It reached 392 exact programs versus 391 under matched extra training,
while validity and multi-step exact reconstruction regressed against the frozen checkpoint.

The sections below preserve the data qualification, representation and training stages that led to
this result, beginning with the 2026-09-13 request.

## Source identity

The `compose_lipid` manifest at `data/manifests/lipid_assets_v1.json` declares an R1 corpus with
SHA-256 `8a691e47e24a921233a05627094627e54f1a5a404e998c11ee69a51716d62f3a`. This is the same
100,296,410-byte file already vendored as `data/vendor/r1_reaction_enumerated_support_v1.csv`.
The manifest itself has SHA-256
`ab22d9428b0b1b3a0ff62598c175f779918626961d212ef8b1b06b8b725e678f`. Its declared 464,266 lines
include the header; FORGE counts **464,265 data rows**. No additional corpus copy is needed.

These are **reaction-enumerated support libraries**. Their names and registry control examples do
not establish that all products were experimentally synthesized, that all products are ionizable
lipids, or that precursor synthesis and procurement close. The existing source-executed Ugi,
BL and LX evidence remains a separate layer. No external biological labels or holdout structures
are imported.

| Registry family | R1 rows | Qualified exact single-event rows |
| --- | ---: | ---: |
| `ugi_3cr_agile` | 222,768 | 179,010 |
| `aza_michael_amine_acrylate` | 16,590 | 351 |
| `epoxide_opening_amine` | 16,590 | 351 |
| `passerini_3cr` | 77,571 | 77,571 |
| `reductive_amination_amine_aldehyde` | 21,654 | 459 |
| `thiol_michael_thioether` | 1,521 | 1,521 |
| `disulfide_coupling` | 780 | 780 |
| `carbamate_amine_chloroformate` | 16,590 | 351 |
| `urea_amine_isocyanate` | 55,032 | 351 |
| `acetal_aldehyde_diol` | 1,989 | 1,989 |
| `iphos_amine_dioxaphospholane` | 16,590 | 351 |
| `amide_coupling_acid_amine` | 16,590 | 351 |

## Implemented interface

`forge.assembly.families.RegistryAssemblyAdapter` implements the existing `AssemblyAdapter`
contract for a fixed-arity reaction selected from authenticated registries. It provides:

- Role assessment using the registry's required handles, forbidden patterns and site multiplicity.
- Sanitized, connected constitutional forward products and exact target reconstruction checks.
- Reverse decomposition followed by role qualification and exact forward replay.
- Explicit enumeration-saturation reporting. A forward check is admissible only when `exact` is
  true **and** `saturated` is false; decomposition raises on a saturated search.

The adapter rejects unknown roles, unsupported role counts, disconnected or isotope-labelled
inputs, and invalid outcome bounds. Atom-map numbering and stereochemistry do not change the
constitutional identity. Forward products receive no valence repair. Reverse fragments use the
existing template-hydrogen normalization, and every returned trace must replay exactly.

`load_assembly_libraries` authenticates an explicit registry union and rejects missing, duplicate
or unexpected families. The existing Ugi adapter and repeated BL/LX program adapters remain
available. A single fixed-arity Michael or reductive-amination event does not stand in for a
source-executed sequence of repeated additions.

`forge.corpus.combinatorial_libraries.read_library_records` streams source rows with family,
ordered roles, block IDs, complete precursor structures and original `realism_weight` values.
Unknown block identities, role mismatches, negative weights and nonfinite weights are errors.
**Zero weights are retained**: 32,304 urea rows have zero mass under the frozen source policy.
All twelve families still contain positive-weight examples. No epsilon floor, raw-frequency
sampling, or replacement weighting is introduced.

`tensorize_library_record` sends complete products to the existing sparse whole-graph tensorizer
and requires an exact constitutional encode/decode round trip. The existing atom vocabulary
already includes sulfur and phosphorus. This bridge supplies no invented precursor-origin,
reaction-core, or component-identity tokens to the model.

## Runnable qualification

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m experiments.phase1.multireaction.combinatorial_libraries \
  --repo-root . \
  --config configs/multireaction/combinatorial_libraries_v1.json \
  --output-dir results/phase1/combinatorial_libraries_v1
```

Use a fresh output directory for another run. Existing results cannot be overwritten. A completed
qualification with a failed gate writes a `blocked` result and the CLI exits nonzero; malformed or
changed inputs publish no completed result. Inputs, configuration, implementation, runtime version,
seed and parameters are recorded. Input and implementation drift during execution is rejected.

The qualification scans **every** declared row for precursor identity, registry role policy,
exact forward reconstruction, enumeration saturation, complete graph size, atom states and bond
states. A compressed per-row ledger preserves every issue. Identity counts use a temporary SQLite
table rather than retaining the entire corpus in memory; worker submission is bounded.

The sparse encode/decode panel contains at most sixteen deterministically hash-ranked rows per
family, selected independently of qualification outcome. It is a **sampled representation check**,
not an exhaustive encode/decode claim. Registry positive and negative controls measure executable
contract behavior; they do not estimate chemical synthesis precision. No reductive-amination
substructure hit rate is computed.

## Full-corpus result

`results/phase1/combinatorial_libraries_v1/result.json` has status **blocked**. It records all
464,265 rows and 458,047 unique constitutional products. Under the strict fixed-arity contract,
263,436 rows qualify and 200,829 do not. Of the latter, 108,870 fail registry role admission;
the rest fail exact single-event reconstruction after role admission. No enumeration saturates.
These counts do **not** establish that the nonqualifying products are chemically invalid.

Every row fits the existing atom and bond vocabulary. Observed support reaches 140 heavy atoms
and one closure. All 192 sampled sparse encode/decode checks pass. All thirteen registry positive
examples reconstruct and all fourteen registry negative examples are rejected. These controls
qualify the interface; they do not override the full-corpus failure.

The stored precursor IDs identify component types without a complete ordered reaction program.
Two diagnostic examples establish why that distinction matters: source row 222,847 (aza-Michael)
and row 333,622 (reductive amination) fail one-event replay but reconstruct exactly with two
copies of the saved tail using the existing repeated-program adapter. Both searches complete
without saturation. `repeat_examples.json` records the checks and input hashes. These two
examples do not attribute every other failure or establish source-executed synthesis.

`admission_summary.json` separates source-positive weights from qualified-positive weights. All
twelve families contain some positive-weight, single-event-qualified rows, including 144 urea
rows. Restricting training to those rows would discard much of the declared support, so this
initial qualification does not create such a training subset. It retains the complete failure ledger.

The result SHA-256 is `c08ce9c0a6ec4d30ab469f4bb8688c94aaeb54d673cd2d0bf86d7bdc23b80e2f`.

## Boundary before learned generation

The current trained semantic model still uses Ugi, repeated BL and repeated LX programs. The data
and assembly interfaces, followed by the computed-program and split work below, do not establish
trained family conditioning, realism, or generalization. These were the requirements before the
semantic/cache follow-up; the final section closes items 1–4 as representation plumbing only:

1. A versioned twelve-family semantic contract, including atom-origin ambiguity and assembly-
   introduced atoms. Repeated programs must retain their qualified step structure. Declare whether each
   library supervises whole lipid assembly or precursor construction: the acetal, disulfide and
   thiol-Michael products here have no nitrogen atoms and cannot supply nitrogen-based heads.
2. Connect the new qualified-program view, global identity partitions and versioned realism-weight
   mixture below to the model cache. Preserve all exclusions and the existing source-executed
   evidence layer; do not treat raw corpus rows as admitted supervision.
3. Declare the reduced eligible support explicitly in any training design. Do not describe a run on
   this view as training on all 464,265 source rows. Broader admission requires qualified evidence
   or a representation that resolves ambiguity without relaxing registry gates.
4. Full representation qualification and bounded CPU training checks before a new training run.
5. Per-family evaluation of structural validity, exact L1 reconstruction, diversity and component
   novelty, followed by realism assessment. Report aggregate and per-family results so a large
   family cannot conceal failures in a small family. Existing gates cannot be weakened.

Source-executed synthesis evidence and generated-candidate L2/L3 dossiers remain necessary for their
respective synthesis claims. Neither this corpus nor a successful reconstruction check closes them.

## Validation

All 30 vendor assets verify. All 50 focused checks pass, including the new adapter/corpus tests and
existing Ugi and repeated-program tests. Black and Ruff pass for the five new Python files.
The full repository run reports **2,382 passed, 167 failed, 21 errors, and 92 skipped/xfail**;
all 44 new tests pass. Remaining failure/error totals match the previously recorded full run, with
no claim that every old failure was diagnosed. Existing tests and gates were not changed.

Full logs, JUnit reports, input hashes and commands are retained in
`results/phase1/combinatorial_libraries_validation_v1/`. The validation report SHA-256 is
`35383a17a4031539bb10193325d78cca6d542079bbde96aa08b34f00b1899cbc`.
The repository-wide Phase 1 definition of done and twelve-family training admission are not met.

## Computed programs and global identity partitions

The follow-up implementation is `forge.assembly.library_programs` plus
`forge.corpus.library_program_dataset`. Its versioned contract is
`configs/multireaction/library_program_dataset_v1.json`:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m experiments.phase1.multireaction.library_program_dataset \
  --repo-root . \
  --config configs/multireaction/library_program_dataset_v1.json \
  --output-dir results/phase1/library_program_dataset_v1
```

Use a fresh output directory. The complete corpus and authenticated single-event ledger are
required inputs. Repeated programs retain the recorded accumulator and repeated-component
identities, with the accumulator allowed in either registry input position. This covers the
acid-first amide registry entry as well as the amine-first entries. Ugi and other fixed-arity
families retain their single-event contract. Registry transforms and role policies are unchanged.

The search exhausts complete breadth-first layers, up to twelve repeated steps, 256 outcomes per
expansion, 4,096 states per layer and 16,384 expansions per precursor tuple. A truncated layer
cannot admit any target from that layer; earlier complete layers remain valid. Distinct ordered
constitutional paths are counted up to two. Only one policy-valid minimal path permits unique
ordered-program supervision. This does not resolve atom-origin ambiguity or establish the
experimental reaction order. `replay_library_program` authenticates every supplied intermediate
with the strict adapter.

Raw transform execution is retained as a diagnostic when a precursor fails registry policy.
An exact raw witness in that case cannot acquire training weight. Saved targets are used for exact
identity checks and one proven size bound: if the template contains a single retained accumulator
atom, its existing graph cannot lose atoms, and adding a new heavy atom makes growth strict. Once
the entire frontier is at least as large as the largest saved target, further steps cannot match.
Templates that lack this proof receive no shortcut. No reactive sites are selected using targets.

The largest urea precursor group contains 245 saved targets. The bound reduced its measured search
from **72.82 s / 6,186 expansions** to **0.38 s / 51 expansions**. All recovered target fields,
including witnesses and path counts, were identical. Unresolved reason codes changed from the
state cap to the justified size stop. This is a one-group benchmark, not a corpus-wide speedup
estimate; its input and source hashes are in
`results/phase1/library_program_size_bound_benchmark_v1/result.json`.

`forge.corpus.library_splits` assigns one partition to each constitutional component identity,
irrespective of library, role or block identifier. It preserves historical restrictions from both
M0 split ledgers, original/current Ugi assignments, component registries and original/mixed BL/LX
splits. When historical schemes disagree, the new cohort uses the most restrictive assignment
(heldout, then calibration, then train); historical files remain unchanged. R0 identities are read
as existing digests, without loading or decomposing R0 molecules. Public assignment ledgers are
used for molecular identity and fold metadata only, without biological labels.

Unconstrained components use deterministic seed 20260913 and 70/15/15 hash thresholds. These are
assignment probabilities, not promised final proportions. A product requires all its precursors,
its selected intermediates and its own identity to be compatible with one fold. Mixed combinations,
conflicting duplicate products, conflicting family annotations and cross-partition intermediate
exposure are quarantined. A final audit requires zero identity overlap across active partitions
and zero protected nontraining identities in training. This is an exact-identity split, not a
claim of chemical-family separation or a newly untouched external test set. Existing checkpoints
must not be treated as unexposed to these new partitions.

Artifacts preserve both the complete source and the eligible view:

- `programs.jsonl.gz`: precursor groups, bounded search outcomes and deterministic witnesses.
- `row_partitions.csv.gz`: every original row, weight, product/group identity, fold and abstention.
- `products.csv.gz`: one record per constitutional product and its available qualified program.
- `component_partitions.json`: global component assignments and historical constraint sources.
- `result.json`: input/config/source hashes, counts, partition checks and admission gates.

The training view averages original `realism_weight` over a product's source rows, then normalizes
within each eligible training family and gives those families equal total mass. Duplicate source
strings do not become duplicate model examples. Original zero weights remain recorded, and products
with zero mean weight, unresolved/ambiguous programs or nontraining folds receive zero training
sampling mass. This new weighting contract is explicitly versioned; it does not alter frozen runs.
All original rows remain accessible even when they are ineligible for this view.

Universal corpus admission requires every source row to have a unique policy-qualified program;
evaluation readiness additionally requires positive-weight qualified products in all three folds
for every family. A failed gate writes a complete `blocked` result and returns a nonzero exit code.
Neither an eligible subset nor a mechanically exact program establishes twelve-family model
performance, improved realism, source-executed synthesis or L2/L3 closure.

### Completed program/split result

The full run took 748.99 seconds with two CPU workers. It retained all **464,265 source rows**,
**458,047 unique products** and **281,973 precursor groups**. It recovered exact computed programs
for **422,013 rows**. Of these, **294,444** have a unique minimal program that passes every registry
policy, an increase of **31,008** over strict single-event qualification. Observed recovered depths
range from one to four; the declared twelve-step bound was not reduced. No outcome, state or
expansion cap was reached in this run.

The remaining **169,821 rows** are not admitted: 42,252 lack an exact program using repeated copies
of their recorded components, 108,771 have no policy-valid minimal path, and 18,798 have multiple
policy-valid minimal paths. These are limitations of the declared computed-program contract,
not claims that the products cannot be synthesized. The pipeline does not invent missing precursor
identities or experimental order to explain an unmatched saved graph.

Every family has positive-weight, uniquely qualified examples in all three active partitions:

| Family | Qualified source rows | Eligible unique train | Calibration | Heldout |
| --- | ---: | ---: | ---: | ---: |
| `ugi_3cr_agile` | 179,010 | 36,112 | 133 | 2,100 |
| `aza_michael_amine_acrylate` | 3,666 | 2,065 | 21 | 72 |
| `epoxide_opening_amine` | 3,666 | 2,013 | 14 | 120 |
| `passerini_3cr` | 77,571 | 13,728 | 171 | 672 |
| `reductive_amination_amine_aldehyde` | 4,794 | 1,512 | 133 | 140 |
| `thiol_michael_thioether` | 1,521 | 858 | 21 | 18 |
| `disulfide_coupling` | 780 | 351 | 28 | 21 |
| `carbamate_amine_chloroformate` | 3,666 | 2,016 | 14 | 120 |
| `urea_amine_isocyanate` | 10,449 | 1,725 | 17 | 171 |
| `acetal_aldehyde_diol` | 1,989 | 624 | 114 | 49 |
| `iphos_amine_dioxaphospholane` | 3,666 | 1,953 | 28 | 96 |
| `amide_coupling_acid_amine` | 3,666 | 1,386 | 63 | 192 |
| **Total** | **294,444** | **64,343** | **757** | **3,771** |

The split audit reads 223,703 historical identity constraints and assigns 489 distinct component
identities across the 490 block IDs. It finds **zero identity overlaps across active partitions**
and **zero protected nontraining identities in training**. Of all distinct products, 303,857 are
quarantined; the other fold labels cover 144,181 train, 4,021 calibration and 5,988 heldout products.
Those fold totals include unqualified or zero-weight records and must not be confused with the
eligible counts in the table. Every noneligible product has zero training sampling weight.

The result status remains **blocked for universal corpus admission**. The source-preservation,
partition-protection and all-family fold-coverage gates pass. The all-source unique-program gate
fails and remains unchanged. A usable qualified view across all twelve families now exists, while
the full corpus is not admitted as exact program supervision.

The result SHA-256 is `04d0a2c9981727de4879a41b640c6ef11d4b1f0564cca18f0eb75401904de28a`.

### Follow-up validation

All **79 focused checks** pass, including the **24 new program/split tests**. All 30 vendor assets
verify. Black/Ruff pass for the twelve active Python files and the saved artifact verifier. The
full suite reports **2,406 passed, 167 failed, 21 errors, 92 skipped/xfail**. Every failing/setup-error
test ID exactly matches the preceding twelve-library run; no existing tests or gates were relaxed.

A separate artifact check authenticates every declared input, source and output hash, reads all
464,265 source/partition rows to verify original weight preservation, checks all 458,047 unique
product records and training weights, and independently replays **136** deterministic witness
controls through the strict adapter. This includes admitted, ambiguous and policy-rejected cases.
The witness replay is sampled; it is not a second full-corpus reconstruction claim.

```bash
PYTHONPATH=. UV_CACHE_DIR=/private/tmp/forge-uv-cache \
uv run python results/phase1/library_program_dataset_validation_v1/check_artifacts.py
```

Logs, JUnit XML, the verification script/receipt and the validation report are preserved in
`results/phase1/library_program_dataset_validation_v1/`. The validation report SHA-256 is
`a5604107a540890fddc03b071e31d25530b3cbff3e69e45ed3f1eec2e9365d0b`. The original pilot's exact source
and config snapshots are preserved beside its result and authenticate its earlier hashes.
No training or generator sampling occurred. Global Phase 1 definition of done remains unmet.

## Shared atom semantics and packed model cache

The qualified view is now connected to the existing reaction-program graph representation by
`forge.corpus.combinatorial_program_semantics` and `forge.corpus.combinatorial_program_cache`.
The semantic audit replays the unique computed witness for each of the **68,871** eligible products.
Ugi uses the settled specialized annotator so its assembly-introduced amide oxygen remains distinct
from all precursor roles. The other families use temporary isotope lineage during replay; those
labels are removed before publication and never enter model state. Every admitted graph must retain
complete precursor-role origins, reaction-core states and exact sparse constitutional round trip.

The first audit correctly blocked. It showed that demanding a unique source atom index was stricter
than the frozen Ugi semantic contract: the existing 1,100-product Ugi ledger allows multiple
symmetry-equivalent source mappings whenever the model semantic signature is unique. The same
distinction admits 8,896 additional Ugi products without changing any origin, core or fixed state.
For acetal products, the authenticated reaction product template is symmetric under reversal. The
versioned contract therefore uses `map_3_or_map_6` and `map_4_or_map_5` orbit states. These labels
preserve exact core membership and avoid arbitrarily numbering symmetry-equivalent atoms.

The passing audit admits **68,831** products: **64,316 train, 750 calibration and 3,765 heldout**.
It records **40 abstentions** separately: 39 disulfide products and one urea product did not replay
with exact atom lineage. They receive no cache entry and are not silently relabeled. Every family
still has exact semantic support in every active fold. Maximum observed support is 131 heavy atoms,
one closure and four origin components, within the unchanged 140-atom/one-closure bounds.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m experiments.phase1.multireaction.combinatorial_program_semantics \
  --repo-root . \
  --config configs/multireaction/combinatorial_program_semantics_v1.json \
  --output-dir results/phase1/combinatorial_program_semantics_v3

UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m experiments.phase1.multireaction.combinatorial_program_cache \
  --repo-root . \
  --config configs/multireaction/combinatorial_program_cache_v1.json \
  --output-dir results/phase1/combinatorial_program_cache_v2
```

The 6.2 MiB deterministic packed cache reloads through the established
`SynthesisProgramProductionCache` API. Its training measure uses mean source `realism_weight`
within each family, excludes the 40 semantic abstentions, renormalizes within family, and assigns
exactly **1/12** mass to every family. It contains no component identifiers, component structures,
fingerprints, fragment tokens or biological labels. Product graphs and their constitutional SMILES
remain in the established cache format so records can be authenticated and reconstructed.

The cache was exercised with a deterministic local plumbing gate using the lexicographically first
training record from each family. Two independent replicas of one CPU AdamW step produced the same
loss (**15.7110118866**), gradient norm (**4.9361653328**) and model state digests. All twelve
programs were present, 48 parameter-gradient tensors were nonzero, the target batch was unchanged,
and every adapter-fixed state survived noising. The sparse model now accepts the full shared collator
contract, including the role-morphology field it does not parameterize, and fails explicitly if a
potency condition is supplied to this architecture.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m experiments.phase1.multireaction.combinatorial_training_smoke \
  --repo-root . \
  --config configs/multireaction/combinatorial_training_smoke_v1.json \
  --output-dir results/phase1/combinatorial_training_smoke_v2
```

The semantic result, cache result/cache bytes and smoke result SHA-256 values are respectively
`256139e72dbb0628e1f244d3f3da2d28b0e5dd058e2c739fbb6fa5e74f12139b`,
`5d766f29c3c777be3b6744c63bd54a1b9e770e19f03a30d19db8909b5045a602`,
`b4f6e6a03f253a90f76915a3b1dc65f1bab5a77229ca1eba0c47c4f9dacbc66c`, and
`8fb701cbdb9e641092663a56f7fef6f4969960dcc06b3acd5ab4a1c2c3381319`.

This completes data-to-model plumbing only. One optimizer step does not establish convergence,
generation quality, improved realism or release readiness. No molecule was generated, no checkpoint
was retained or promoted, and no structural-validity, exact Ugi reconstruction, diversity or novelty
gate changed. A bounded twelve-family training design and per-family evaluation remain necessary
before a model comparison can be authorized or interpreted.

The combined new and adjacent regression suite passes **116/116** tests. Black and Ruff pass for
all 24 touched Python files, whitespace checks pass, and all 30 vendored assets verify. The final
repository-wide run reports **2,429 passed, 167 failed, 21 errors and 92 skipped/xfail** across
2,709 tests. Its complete set of 188 failing/error test identifiers is exactly equal to the prior
program-dataset baseline, with zero new or resolved identifiers. The unchanged failures remain,
so repository-wide Phase 1 definition of done is not met.

The read-only artifact checker, focused/full JUnit reports, full log, vendor verification log and
failure-set comparison are retained in
`results/phase1/combinatorial_shared_model_validation_v1/`. The validation result SHA-256 is
`bfe5167c181954cee4997dc9db339bdab7a0f123de7f5504b8ed36dd3f865ccf`.

## Twelve-family learning-capability gate

The next staged uncertainty was whether a passing optimizer step masked family interference or
insufficient capacity in the shared sparse architecture. The frozen learning gate selects one
lower-median-size training graph from every family, gives each family equal objective mass and
reuses one deterministic corruption at flow time 0.5. It requires all twelve family losses to
decrease, a final-to-initial equal-family loss ratio no greater than 0.25, and exact reconstruction
of all twelve tensors. This is deliberately a memorization test. It cannot measure calibration or
generation quality.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run python -m experiments.phase1.multireaction.combinatorial_overfit_gate \
  --repo-root . \
  --config configs/multireaction/combinatorial_overfit_gate_v1.json \
  --output-dir results/phase1/combinatorial_overfit_gate_v1
```

The gate passes. Across 1,536 CPU optimizer steps, equal-family loss falls from
**10.9852552414** to **0.0000597239**, a final-to-initial ratio of
**0.0000054367**. Every family improves and all **12/12** fixed-corruption records reconstruct
exactly. The selected records span 32–68 heavy atoms and zero or one closure. Every adapter-fixed
state remains exact, the clean batch is unchanged, gradients and losses remain finite, and no
generator call occurs. No diagnostic checkpoint is retained.

A second fresh execution reproduces the selected records, model contract, entire loss trajectory,
per-family losses and reconstructions, gates, and decision exactly. The primary result and replay
SHA-256 values are `82f2839b9a6fc1aaac5f4b043b949ea76cf54140b1efb03359e292583fe9bf9e`
and `ac0ef1493ae4e99c3b887204856c85cf99c9ca4e8d29bccd053325c4e423ef49`.

The result supports proceeding to a limited training pilot with prespecified per-family calibration
evaluation. It does not show heldout generalization, unconditional generation, improved realism,
or preservation of generation-level validity, exact Ugi reconstruction, diversity and novelty.
Those remain the required comparison outcomes before any model can be promoted.

The same **116** focused tests pass and all 30 vendored assets verify. The final full suite again
reports **2,429 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,709 tests. Its exact
188-test failure/error set equals the preceding saved baseline. The read-only verifier, replay,
JUnit reports and logs are retained under
`results/phase1/combinatorial_overfit_gate_validation_v1/`. The validation result SHA-256 is
`e2ce6e7982d62888d2d9bda1eb6d7828747b07ee5e61ffccc7bc6ffe95000c54`.

## Limited balanced-training pilot

The passing memorization gate permits a bounded transfer check, not a production run. The pilot
draws one positive-weight training product from each family at every step using `realism_weight`
within families and equal total mass between families. It trains the same 248,677-parameter sparse
architecture for 768 local CPU steps. Before and after training, it applies identical deterministic
corruptions to every one of the **750** calibration products. Heldout structures remain unopened.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
uv run python -m experiments.phase1.multireaction.combinatorial_training_pilot \
  --repo-root . \
  --config configs/multireaction/combinatorial_training_pilot_v1.json \
  --output-dir results/phase1/combinatorial_training_pilot_v1
```

The pilot passes its frozen decision rules. Across 9,216 balanced training examples, the first/last
32-step mean losses are **4.9519119486** and **0.6265338296** (ratio **0.1265236208**).
Equal-family calibration loss falls from **9.0746303814** to **0.9099036008** (ratio
**0.1002689435**), and all **12/12** families improve. Per-family final-to-initial calibration
ratios range from 0.0264 for acetal formation to 0.1688 for Ugi. All adapter-fixed states remain
exact and gradients and losses remain finite.

The key limitation is Ugi exact reconstruction. Its mean calibration loss falls from
**9.4827925890** to **1.6005689560** across 133 products, but exact tensor reconstruction remains
**0/133 before and 0/133 after**. The non-regression gate therefore passes without establishing
exact Ugi terminal reconstruction. Calibration denoising is an intermediate supervised measure;
it does not establish unconditional generation quality, improved realism, validity, diversity or
novelty.

A fresh replay reproduces the model-state hashes, 9,216-example sampling-ledger digest, training
trajectory and every calibration field exactly. Primary and replay result SHA-256 values are
`e8ad3a9734290d7f299096745a1ce1f23e52a5b196f3e996615071c44d79ebad` and
`53fdcc65af83b8c78f7533f7f15b287721cf6157560e42fd88aa8b2b6845b315`.
No checkpoint was retained, no molecule was generated, and no remote compute, route, oracle,
candidate-selection or biological-guidance call occurred.

The same **116** focused tests and all 30 vendor checks pass. The full suite remains at
**2,429 passed, 167 failed, 21 errors and 92 skipped/xfail**, with exactly the preceding 188-test
failure/error set. The verifier, replay, logs and JUnit reports are retained under
`results/phase1/combinatorial_training_pilot_validation_v1/`. The validation result SHA-256 is
`e2184686938f96eb271c24a8efa94b7cbb4a6d67dd9d442de1b2be2a638e37fa`.


## Learned generation across all twelve libraries (2026-09-14)

The current objective is to make FORGE generate across the combinatorial libraries. Ugi is one
reaction family. Repeated aza-Michael and reductive-amination programs are programs within their
respective families; a family label alone does not specify a full synthesis program.

The new command retains an inference checkpoint that exactly reproduces the frozen balanced pilot:
initial weights, final weights and training sample sequence all match. It uses the same 9,216
training presentations from the admitted 64,316-product training population. This step does not
silently blend in the separate historical expanded Ugi/BL/LX training corpora. It saves a
non-executable tensor checkpoint and verifies loading it through the common sampling interface.
Optimizer and RNG states are not retained, so it supports inference rather than faithful training
resumption.

Generation uses one shared sparse flow across all twelve families, with 64 attempts per family,
32 flow steps and batch size 16. Train-only semantic layouts are sampled using the existing
`realism_weight` measure. Node counts, component-role positions, core labels, program depth,
closure count and adapter-fixed states condition generation. Variable atoms, parent pointers,
bonds and closure endpoints are withheld. A metamorphic test changes all of those hidden targets
while holding the declared semantics fixed and obtains identical generated outputs. This is
**empirical-layout-conditioned generation**, not unconditional layout generation.

The trained checkpoint and its original random initialization receive the same layouts, source
marginals, noise seeds and strict valence decoder. The decoder makes one attempt and never repairs
or replaces failures. The previous Ugi-specific production decoder, gates and checkpoints remain
unchanged; this diagnostic does not establish noninferiority to those production arms.

| Family | Valid connected / 64 | Exact program / 64 | Distinct train-novel exact products |
| --- | ---: | ---: | ---: |
| `acetal_aldehyde_diol` | 64 | 61 | 56 |
| `amide_coupling_acid_amine` | 64 | 31 | 22 |
| `aza_michael_amine_acrylate` | 55 | 14 | 13 |
| `carbamate_amine_chloroformate` | 64 | 21 | 17 |
| `disulfide_coupling` | 64 | 54 | 36 |
| `epoxide_opening_amine` | 64 | 19 | 14 |
| `iphos_amine_dioxaphospholane` | 60 | 10 | 9 |
| `passerini_3cr` | 63 | 45 | 45 |
| `reductive_amination_amine_aldehyde` | 64 | 16 | 13 |
| `thiol_michael_thioether` | 64 | 54 | 50 |
| `ugi_3cr_agile` | 64 | 50 | 49 |
| `urea_amine_isocyanate` | 60 | 3 | 3 |

Across 768 trained attempts, **750** are valid and connected and **378** have an exact computed
program at the requested depth. Those 378 attempts contain **351** distinct constitutional products,
including **327** absent from the admitted cache training set. Every family has at least one such
novel exact product. The random-initialization control yields **23** valid connected graphs and
**one** exact program under the same budget and decoder. These are descriptive single-seed results,
not calibrated realism, efficacy or across-training uncertainty estimates. Urea's 3/64 and iPhos's
10/64 exact-program yields are substantial remaining limitations.

Novelty here is relative to all 64,316 training products in this cache, not all historical FORGE
corpora or the entire enumerated R1 universe. Component novelty uses all train-assigned source
blocks, rather than only blocks drawn during training. Among the exact trained attempts, **347**
have a novel component in every recovered witness. Multiple witnesses are retained instead of
selecting a convenient decomposition for the novelty claim. Diversity is reported as unique
fractions and inverse-Simpson effective product counts, with both attempt and valid denominators.

The evaluator reverses the intended registry reaction to the requested depth and forward-replays
every step against the unchanged role policy. Repeated programs require the same co-reactant
identity throughout, matching this dataset's supervision. Saturated reverse/forward enumeration or
an incomplete bounded search abstains even if an earlier branch found a witness. Distinct programs
are retained; no unique experimental route is asserted. Coverage is the exact-program fraction of
all attempts and of valid outputs. Every returned witness passes computed forward replay, while
**source-adjudicated chemical precision remains unmeasured**. Computed witnesses do not close L2/L3.

The frozen target-scope policy marks acetal, disulfide, thiol-Michael and Passerini as
`precursor_or_neutral_structure_support`. They are part of the shared generation interface, but their
outputs are not automatically complete ionizable lipids. The other eight provide assembly support;
that label also does not establish ionizability or lipid realism.

The NPZ loader loads arrays from all partitions into memory. This run materializes records only
from the training fold and does not use heldout records for training, layouts or metrics. This
explicit distinction supersedes shorthand about never loading heldout bytes in earlier pilot prose.

Commands (each output directory must be fresh):

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_generation \
  --config configs/multireaction/combinatorial_checkpoint_v1.json \
  --output-dir results/phase1/combinatorial_checkpoint_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_generation \
  --config configs/multireaction/combinatorial_generation_v1.json \
  --output-dir results/phase1/combinatorial_generation_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_generation_verify \
  --result results/phase1/combinatorial_generation_v1/result.json \
  --replay results/phase1/combinatorial_generation_replay_v1/result.json
```

The generation config pins the retained checkpoint at `combinatorial_checkpoint_v1/checkpoint.json`.
A fresh checkpoint run authenticates deterministic recovery; sampling can reuse the already pinned
checkpoint without retraining. A deliberate checkpoint replacement requires a new configuration
with its exact path and hash.

The read-only verifier authenticates inputs and artifacts, checks every layout against its training
record, reparses all saved products, forward-replays all **448** saved witnesses across both arms,
and recomputes validity, novelty, diversity and demonstration claims from the ledger. Fresh
sampling replay reproduces the scientific fields and both artifact files byte-for-byte.

- Checkpoint: `results/phase1/combinatorial_checkpoint_v1/checkpoint.json`.
- Primary result: `results/phase1/combinatorial_generation_v1/result.json`.
- Every attempt, failure and exact witness: `results/phase1/combinatorial_generation_v1/attempts.jsonl`.
- Layout provenance: `results/phase1/combinatorial_generation_v1/layouts.json`.
- Replay: `results/phase1/combinatorial_generation_replay_v1/`.
- Verification: `results/phase1/combinatorial_generation_validation_v1/`.

This completes the bounded cross-library generation demonstration. It does not promote the shared
checkpoint or establish production-quality realism, diversity, novelty or validity across all
families. The next model-quality decision should target the weakest reaction programs, preserving
all attempt denominators and unchanged assembly gates.

A local structure gallery is available at
`results/phase1/combinatorial_generation_validation_v1/review.html`. It displays the first three
distinct trained outputs with exact computed programs per family, alongside every family's full
attempt denominators. The gallery receipt pins its inputs and records the displayed attempt IDs.
All 36 embedded depictions were generated; a representative SVG was rendered and visually checked
for legibility. No scientific reviewer judgment was collected.


Generation-stage validation: **136 focused tests pass** with no failures or skips; all 30
vendored assets verify. Black/Ruff pass for the five new Python files, and whitespace checks pass.
The final full suite has **2,449 passed, 167 failed, 21 errors,
92 skipped/xfail** across 2,729 cases. The 188 failure/error identifiers exactly
match the preceding training-pilot baseline; all new generation tests pass in the full run.
The preserved validation result is `results/phase1/combinatorial_generation_validation_v1/result.json`
(SHA-256 `eccca1cd1f1946fb21ff5d22180dc96bef0d009ea973a4783528e3052e6b671c`). Repository-wide Phase 1 definition of done remains unmet because the full suite
still fails. No claim that every legacy failure has been fully diagnosed is made.


## Repeated-program failure audit (2026-09-14)

The audit replays all **896 saved attempts** from the seven families that support repetition,
including both model arms, invalid outputs and depth-one controls. It reproduces the original
identical-co-reactant results, then separately searches for exact programs with different
co-reactants at successive steps. Every step still passes the frozen registry role policy and
strict forward reconstruction, including exact accumulator handoffs. No generated product,
checkpoint, chemistry gate or historical result is changed.

Among **448 trained attempts** in these families:

| Classification | Attempts |
| --- | ---: |
| Pass the original identical-co-reactant contract | 114 |
| Fail that contract but have an exact full-depth mixed-co-reactant program | 208 |
| Have only a shorter policy-valid program within this bounded search | 30 |
| A raw exact inverse/forward candidate exists, but role policy blocks the first step | 29 |
| No verified first inverse/forward step | 50 |
| Invalid molecular graph | 17 |

Thus **208 of 334 original failures** are isolated to the identical-co-reactant condition within
this computed checker comparison. Those 208 cases remain **unaccepted under the original gate**.
The original trained all-library result stays at 378/768 exact programs. This is an attribution
finding about the checker contract, not a new model improvement or a rescore of that result.

For multi-step attempts only:

| Family | Attempts | Original identical-reactant exact | Additional mixed-only diagnostic matches | Any full-depth diagnostic program |
| --- | ---: | ---: | ---: | ---: |
| `amide_coupling_acid_amine` | 43 | 18 | 20 | 38 |
| `aza_michael_amine_acrylate` | 51 | 1 | 33 | 34 |
| `carbamate_amine_chloroformate` | 59 | 16 | 38 | 54 |
| `epoxide_opening_amine` | 47 | 9 | 28 | 37 |
| `iphos_amine_dioxaphospholane` | 47 | 1 | 23 | 24 |
| `reductive_amination_amine_aldehyde` | 54 | 9 | 41 | 50 |
| `urea_amine_isocyanate` | 64 | 3 | 25 | 28 |

The distinction matters most for two-step programs: aza-Michael has 1/51 original exact matches
plus 33 mixed-only matches; reductive amination has 9/54 plus 41. Urea is not explained by identity
alone: its four-step group has only 3/29 full-depth diagnostic matches, including one original
exact match. Different depth groups use different sampled layouts; this is not a causal estimate
of adding a reaction step.

The untrained control has one original exact program, no additional mixed-only matches,
19 outputs without a verified first step and 428 invalid graphs. A more permissive identity
comparison therefore does not generally turn this control into a working generator.

All **32 positive controls**, selected by the first cache indices in each represented TRAIN
family/depth stratum before checking results, reconstruct under the original contract. No audit
search hits an outcome, state or expansion bound. Truncated searches have dedicated abstention
states and tests; partial discoveries cannot become full-depth matches. All **770 returned
full-depth diagnostic programs** across both arms independently forward-replay. Fresh execution
reproduces the scientific payloads and detailed ledger exactly, and the verifier recomputes every
attempt and control.

A role-policy dead-end example is diagnostic only, never admitted. The alternative
`no_verified_first_step` classification can reflect a graph outside the registered transform,
normalization limitations or inverse/forward limitations; it does not prove chemical
impossibility or locate a particular erroneous model bond. Shorter witnesses show partial registry
consistency without proving that a specific bond was omitted. No reductive-amination substructure
hit rate is computed or reported. Computed replay precision remains distinct from unmeasured
source-adjudicated chemical precision, realism, ionizability and L2/L3 closure.

Code inspection also confirms that the current sparse model's forward interface accepts and
discards `repeat_group_states`, `component_position_states`, `component_instance_states` and
`role_morphology_states`; program, role, reaction-core and depth conditioning remain active.
The source hash and AST observations are recorded in
`results/phase1/combinatorial_repeat_audit_validation_v1/model_interface_audit.json`.
This is an implementation observation, not causal evidence that enabling those fields will fix
the failures. A repeated-role label also does **not** establish that two precursor graphs must be
identical; equality intent and correspondence must be qualified separately.

**Next experiment:** explicitly represent whether reaction occurrences reuse the same precursor,
and test whether the generator can obey that constraint. Preserve the current identical-reactant
arm and all validity, reconstruction, diversity and novelty denominators. If mixed-reactant
programs are desired, qualify them as a separate program contract with appropriate supervision and
evidence. Do not silently admit the diagnostic matches. For deeper urea programs, retain a separate
reaction-step/core reconstruction diagnosis; identity consistency alone cannot account for the
remaining failure rate.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_repeat_audit \
  --config configs/multireaction/combinatorial_repeat_audit_v1.json \
  --output-dir results/phase1/combinatorial_repeat_audit_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_repeat_audit_verify \
  --result results/phase1/combinatorial_repeat_audit_v1/result.json \
  --replay results/phase1/combinatorial_repeat_audit_replay_v1/result.json
```

The detailed ledger contains every classification, all full-depth witnesses, a deterministic
shorter-path example where available, and examples of role-policy dead ends. The output directory
must be fresh. This audit performs no model sampling, training, heldout-record evaluation, remote
compute or prospective candidate selection. The packed cache contains heldout arrays but only
TRAIN records are materialized for controls and source verification.

Primary result: `results/phase1/combinatorial_repeat_audit_v1/result.json`
(SHA-256 `34e41d0957aeca5d07754feec26a0ef9e82d251c449e9cd1878d2669f7687dc8`).

Repeat-audit validation: **86 focused tests passed**, no failures or skips; all 30
vendored assets verify. All four new Python files pass Black/Ruff, and whitespace checks pass.
The full suite reports **2,467 passed, 167 failed, 21 errors,
92 skipped/xfail** across 2,747 cases. The failure/error IDs exactly match
the preceding generation-stage baseline, with no new audit failures. Global Phase 1 definition
of done remains unmet because the full suite fails. Reports are preserved in
`results/phase1/combinatorial_repeat_audit_validation_v1/`; validation-result SHA-256:
`bb8e2825dd2bd25d990423fa995f4ca75bc42e5f54433da8b449f340cac82bf1`.

## Precursor-reuse conditioning pilot (2026-09-14)

The user authorized the next bounded experiment after the repeat audit: explicitly represent
precursor reuse and test whether the generator obeys it. This pilot adds a residual conditioning
path to the existing sparse model. At each denoising call it averages noisy atom embeddings over
qualified correspondence classes, then supplies the local and shared embeddings to a learned
linear projection. This is a **soft condition**. It does not tie sampled atoms, bonds or topology
pointers, guarantee equality, change the terminal decoder, or change the strict assembly checker.

Equality intent is qualified separately from repeated-role labels. Each repeated source product
must pass the original full-depth identical-co-reactant gate. Its co-reactant origin components
must have the expected multiplicity and exact, uniquely aligned fragment graphs, including core
labels and serialized atom states. The unique graph isomorphism handles different serialization
orders. Internal symmetry, incomplete source verification or an incompatible occurrence partition
produces a zero relation; the example remains in training and evaluation.

Qualification inspects only the selected TRAIN graphs. Among the 2,698 distinct cache records
needed for training or sampling, 1,106 have a qualified relation, 1,505 are not repeated, 69 have an
occurrence-partition mismatch and 18 have ambiguous correspondence. Among the 768 generation
layouts, 312 are qualified, 403 are not repeated, 46 have an occurrence-partition mismatch and
seven are ambiguous. Thus 312/365 multi-step attempts receive the additional condition. Every
abstention and all original denominators are retained.

Only correspondence classes enter the new path, with no target atoms, bonds, pointers, component
identifiers or precursor strings. Tests change variable source chemistry and topology while
holding the qualified relation and existing layout context fixed; generated graphs remain
unchanged. Nevertheless, correspondence is **additional source-derived structural information**:
this pilot tests its use on empirical layouts. It does not demonstrate a method for producing
those relations for arbitrary new layouts. A future program generator must declare and realize
reuse without consulting a completed target product.

The frozen protocol starts two models from the same retained checkpoint and exactly zero residual
projection. Both have 256,933 parameters, fresh AdamW states, 384 additional steps and 4,608
presentations, one per family per step. Within-family selection uses the unchanged `realism_weight`
training measure. Selected records, per-record flow times and every noisy coordinate match between
arms; the saved corruption digest is identical. The control uses each atom's own embedding in
place of the cross-copy mean, preserving capacity and the eligible-node mask. A third evaluation
disables cross-copy sharing in the trained reuse model. All four evaluations use the original 768
TRAIN layouts, source marginals, flow seed, 32 steps and strict valence decoder. The frozen model
reproduces its original attempt ledger exactly.

| Arm | Valid connected /768 | Exact programs /768 | Multi-step exact /365 | Ugi exact /64 | Train-novel valid attempts /768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen checkpoint | 750 | 378 | 57 | 50 | 719 |
| Matched self-only control | 745 | 391 | 45 | 48 | 710 |
| Reuse conditioning | 743 | 392 | 48 | 48 | 708 |
| Reuse model, sharing disabled | 744 | 389 | 45 | 48 | 708 |

The total exact gain against the original checkpoint is driven by the single-step population;
the intended multi-step target declines from 57 to 48. Against the matched control, reuse gains
only one exact program overall and three in the multi-step subset. Turning sharing off loses
three multi-step matches at fixed treatment weights. These within-seed differences show a small
effect on outputs, not a reliable improvement or a successful mechanism. Four-step urea remains
1/29 exact in all four arms; three-step urea remains 0/3.

Per-family results expose the regressions that an aggregate score would hide:

| Family | Frozen exact /64 | Control exact /64 | Reuse exact /64 | Frozen valid /64 | Reuse valid /64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acetal | 61 | 62 | 62 | 64 | 64 |
| Amide | 31 | 22 | 21 | 64 | 62 |
| Aza-Michael | 14 | 19 | 19 | 55 | 53 |
| Carbamate | 21 | 20 | 22 | 64 | 64 |
| Disulfide | 54 | 61 | 61 | 64 | 64 |
| Epoxide opening | 19 | 13 | 13 | 64 | 59 |
| iPhos | 10 | 17 | 18 | 60 | 60 |
| Passerini | 45 | 53 | 51 | 63 | 64 |
| Reductive amination | 16 | 10 | 11 | 64 | 63 |
| Thiol-Michael | 54 | 61 | 61 | 64 | 64 |
| Ugi | 50 | 48 | 48 | 64 | 62 |
| Urea | 3 | 5 | 5 | 60 | 64 |

The result also retains per-family/depth unique-product counts, inverse-Simpson effective product
counts, novelty against all cache TRAIN products, component novelty against all TRAIN-assigned
source blocks, paired exact gains/losses and every failed attempt. Diversity is not universally
preserved: for example, Ugi unique valid outputs fall from 64 to 62. Train-novel exact acetal
products fall from 56 to 51 despite the increase in acetal exact attempts. Greater aggregate exact
or unique counts do not satisfy the all-family preservation requirement.

**Decision:** the predeclared rule fails; retain the original checkpoint and do not promote this
intervention. The rule required more repeated exact programs and no decrease in any recorded
per-family validity, exact-reconstruction, diversity or novelty metric against both controls. It
is an observed-count screen, not a statistical noninferiority test. This single-seed pilot reuses
previously inspected TRAIN layouts and does not establish heldout generalization. The result
weakens the hypothesis that pooled noisy atom conditioning alone resolves the repeat failures.
A subsequent experiment would need to test a stronger way to enforce shared precursor graph
structure, while accounting for topology/core errors and retaining these same gates. No extra
training sweep was run after inspecting this outcome.

The first local execution took 56.53 seconds end to end; training took 5.38 seconds for the control
and 5.41 seconds for reuse on two CPU threads. These are measured diagnostic runtimes, not a
performance optimization claim. The experiment used no remote compute, heldout evaluation,
biological guidance, prospective selection or registry edits. Computed replay still does not
establish experimental chemistry, lipid realism, ionizability or L2/L3 closure.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_reuse_pilot \
  --config configs/multireaction/combinatorial_reuse_pilot_v1.json \
  --output-dir results/phase1/combinatorial_reuse_pilot_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_reuse_verify \
  --result results/phase1/combinatorial_reuse_pilot_v1/result.json \
  --replay results/phase1/combinatorial_reuse_pilot_replay_v1/result.json
```

Primary result SHA-256:
`dd21d0a12bdf38b725b22736e1f340dc347b231399a418397c28b3cd9a71184f`.
Both trained diagnostic checkpoints, source/input pins, every attempt, all selected training
indices and every qualified or abstained reuse relation are retained in the result directory.

Reuse-pilot validation: **104 focused tests pass**, with no failures or skips. All 30 vendored
assets verify; Black/Ruff and whitespace checks pass. Read-only verification recomputes every one
of the 3,072 attempts and 2,698 reuse plans and reloads both diagnostic checkpoints. A fresh full
execution reproduces both checkpoint tensors, every artifact hash and every compared scientific
field exactly. Full suite: **2,485 passed, 167 failed, 21 errors, 92 skipped/xfail** across 2,765
cases. All 18 new tests pass in the full run, and failure/error IDs exactly match the preceding
repeat-audit baseline. Global Phase 1 definition of done remains unmet. Reports and their
reproducible summary script are in `results/phase1/combinatorial_reuse_validation_v1/`;
validation-result SHA-256:
`f815329172936152914ac18eb135e8186dee1636329c8df94ae26e6cbbeb34dd`.

## Graph reuse with explicit preservation constraints (2026-09-14)

The active goal covers **all twelve combinatorial libraries**, with exact reconstruction under
each family's own complete reaction program. Ugi is one adapter within this scope. The next
bounded intervention enforces an equality relation at the generated graph level, using the
existing frozen checkpoint and the previously qualified precursor correspondence. It adds no
training and changes no assembly, validity, atom-count, cycle-count or reaction-depth gate.

For each qualified repeated layout, a wrapper captures the deterministic terminal graph without
altering the sampler. An originally valid but nonexact graph receives at most one proposal per
generated co-reactant occurrence: copy that occurrence's generated atoms and internal edges into
every corresponding occurrence. Cross-component edges, total atom count, total edge count and
fixed reaction chemistry must remain unchanged. The graph must sanitize and remain connected,
and the original **full-depth, identical-co-reactant** assembly checker must pass. Neither source
variable atoms nor source variable bonds are copied. Source-derived correspondence remains extra
TRAIN layout context, so this is not autonomous construction of new program layouts.

An initial discovery run selected the exact proposal requiring the fewest graph edits, then
canonical SMILES and donor index as deterministic tie-breaks. It increased exact reconstruction
from **378 to 578/768**, but failed the preservation rule: **13** novel valid attempts became
TRAIN-familiar and amide unique valid products decreased by one. The initial result is preserved
at `results/phase1/combinatorial_graph_reuse_discovery_v1/result.json` (SHA-256
`b218cf52fb5f261bfe477e2725ed44de7385be773326b98af0532f3a56974a70`). These losses were not
removed from the original report or accepted by changing its thresholds.

A separate, explicit admission stage addresses those losses. It visits the saved attempts in
their original order, retains every existing exact output and every invalid attempt, and considers
the same bounded exact proposals. If the original graph is TRAIN-novel, its replacement must also
be TRAIN-novel. Within the current family output inventory, a candidate must have fewer existing
occurrences than the original. Among feasible candidates, the original deterministic edit-count
ordering applies; when none qualify, the original output remains, including its failure status.
Every proposal and rejection remains available in the two stages' ledgers.

This stage **uses TRAIN membership and output multiplicities for selection**. It is a constrained
completion pipeline, not an improvement in the learned checkpoint. If the original multiplicity
is `a` and the candidate multiplicity is `b`, accepting only `b < a` changes the squared-count sum
by `2*(b-a+1) <= 0`. At fixed valid count, inverse-Simpson effective product count cannot decrease.
The rule also prevents loss of a distinct product. Keeping all existing exact outputs prevents
loss of their unique, novel and novel-component subsets. Thus preservation of these particular
count metrics follows partly by construction; it is not statistical noninferiority or a guarantee
about every measure of chemical diversity.

The admission rule and two fresh sampling configurations were frozen in
`configs/multireaction/combinatorial_reuse_confirmation_protocol_v1.json` before either fresh
run was inspected. Both use 128 attempts per family, 32 flow steps, the unchanged checkpoint and
independent layout/noise seeds: **2026091511/2026091512** and **2026091521/2026091522**. Layouts are
sampled within each family from TRAIN using the existing `realism_weight` measure. These are fresh
TRAIN layout/noise replications, not heldout generalization tests or independent training seeds.

| Population | Attempts | Valid, before → after | Exact programs, before → after | Novel valid, before → after | Sum of within-family unique valid products, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | 768 | 750 → 750 | 378 → 576 | 719 → 719 | 721 → 722 |
| Fresh run 1 | 1,536 | 1,495 → 1,495 | 771 → 1,137 | 1,435 → 1,435 | 1,418 → 1,421 |
| Fresh run 2 | 1,536 | 1,500 → 1,500 | 769 → 1,128 | 1,437 → 1,437 | 1,414 → 1,420 |

All three populations preserve every declared metric for **each** family: valid connected
attempts, exact programs, unique valid products, effective product count, unique exact products,
novel valid attempts, unique novel exact products, and exact programs with a novel component in
every recovered witness. Counts of novel exact products increase **327→525**, **657→1,023** and
**661→1,015**, respectively. Per-family distinct counts are summed here; this does not deduplicate
products across different families.

| Family | Discovery exact gain | Fresh run 1 exact gain | Fresh run 2 exact gain |
| --- | ---: | ---: | ---: |
| Acetal | 0 | 0 | 0 |
| Amide | +21 | +47 | +44 |
| Aza-Michael | +29 | +71 | +64 |
| Carbamate | +39 | +66 | +70 |
| Disulfide | 0 | 0 | 0 |
| Epoxide opening | +31 | +51 | +48 |
| iPhos | +27 | +53 | +63 |
| Passerini | 0 | 0 | 0 |
| Reductive amination | +35 | +61 | +53 |
| Thiol-Michael | 0 | 0 | 0 |
| Ugi | 0 | 0 | 0 |
| Urea | +16 | +17 | +17 |

All gains occur in repeated programs. Multi-step exact reconstruction increases **57→255/365**,
**125→491/755** and **116→475/738**. The five nonrepeated families pass through unchanged. Deeper
urea remains unresolved: discovery three-step/four-step counts stay **0/3** and **1/29**; each
fresh run stays **0/17** and **0/61**. The intervention does not correct every reaction-core or
topology error and abstains where correspondence is ambiguous or the occurrence partition fails.
All three- and four-step urea layouts in these populations have an occurrence-partition mismatch;
they receive **no** graph-copy proposals. Their unchanged scores therefore do not test whether
correctly enforcing reuse would help at those depths. Resolving the component-origin partition
is a prerequisite for that comparison. The per-family/depth qualification ledger is
`results/phase1/combinatorial_graph_reuse_validation_v1/qualification_scope.json`.

The proposal stage performs **488, 992 and 974** graph-copy proposals, respectively, with **471,
926 and 922** additional exact-program checks. These costs are separate from unchanged neural
sampling and from verification/evaluation calls. This is not a matched-total-compute comparison.
Discovery's original sampler ledger exactly matches the earlier frozen experiment; a fresh full
discovery replay reproduces all saved artifact hashes and scientific fields exactly. Read-only
verification reconstructs every graph-copy proposal, admission decision, final program witness
and reported metric. It also checks all pins, TRAIN layout selection and the frozen confirmation
protocol. Successful checksum verification alone is not substituted for those semantic checks.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_graph_reuse \
  --config configs/multireaction/combinatorial_graph_reuse_discovery_v1.json \
  --output-dir results/phase1/combinatorial_graph_reuse_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_graph_reuse_verify \
  --result results/phase1/combinatorial_graph_reuse_discovery_v1/result.json \
  --replay results/phase1/combinatorial_graph_reuse_replay_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_reuse_admission \
  --config configs/multireaction/combinatorial_reuse_admission_discovery_v1.json \
  --output-dir results/phase1/combinatorial_reuse_admission_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_reuse_admission \
  --verify results/phase1/combinatorial_reuse_admission_confirmation_1_v1/result.json
uv run python results/phase1/combinatorial_graph_reuse_validation_v1/report_findings.py
```

Each output directory must be fresh. Admission configurations bind a particular saved parent
result. To use a newly generated parent, create a new admission configuration with that result's
path and SHA-256; never rewrite historical input pins. The reproducible combined findings and
confirmation-protocol authentication are in
`results/phase1/combinatorial_graph_reuse_validation_v1/findings.json`.

**Decision:** retain the bounded completion mechanism as evidence of improved family-specific
reconstruction with preserved declared count metrics on the tested populations. Do not promote a
new learned model or claim improved lipid realism. Remaining work includes deeper reaction
programs, construction of new layouts with declared precursor reuse, generalization evaluation,
and repository validation. Acetal, disulfide, thiol-Michael and Passerini retain their documented
neutral/precursor scope; this result does not turn every library into a qualified complete-IL
generator. No heldout evaluation, biological optimization, remote compute, prospective selection
or L2/L3 evidence promotion occurred. The broader goal remains active.

Graph-reuse validation: **178 focused tests pass**, with no failures or skips. All 30 vendored
assets verify; all 11 new Python files pass Black/Ruff and whitespace checks pass. The final full
suite reports **2,511 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,791 cases.
All 26 new tests pass in the full run. Failure/error IDs exactly match the prior reuse-pilot
baseline; no failing test was weakened or skipped. Global Phase 1 definition of done remains
unmet. Full logs, input hashes, semantic verification, qualification scope and reporting scripts
are retained in `results/phase1/combinatorial_graph_reuse_validation_v1/`.

Combined-findings SHA-256:
`5b6f461bd371679988b3d14cbaccd14fad5714db210cf609fb4beb817c86e178`.
Validation-result SHA-256:
`a44f8b20104799627e8e04bc3c05d4af3742760df19992bc7732f05602376752`.

## Exact reaction occurrences within connected role blocks (2026-09-14)

The remaining urea partition failure was caused by conflating two different objects. The existing
cache correctly stores **connected same-role components** for serialization. A later reaction can
join atoms from different isocyanate additions, so one such component can contain several precursor
occurrences. Connectedness and a shared role do not establish a shared precursor occurrence.
The cache, its original semantics and all historical results remain unchanged.

The additional occurrence map replays every strict source-program witness with a temporary unique
label on each precursor atom at each addition. Every exact labelled outcome is propagated across
the supplied intermediate sequence. After clearing labels, every canonical graph mapping must
agree on the occurrence partition and cross-occurrence atom correspondence, modulo occurrence
names and common coordinate names. Independent internal atom symmetry still causes abstention.
All original outcome/state/expansion limits apply, and canonical matching is bounded at 4,096;
saturation fails closed. No chemistry constants are retyped: replay uses the vendored adapter.

The map must cover exactly the co-reactant role atoms already recorded in the cache. Every
occurrence must retain the same precursor coordinates, and its induced product atom states and
internal bonds must match those of the other occurrences. Reaction-core position labels may differ
between additions because an earlier co-reactant atom can participate in a later step; matching
precursor coordinates does not erase this difference. All existing fixed atom/bond checks remain
in force. The new sidecar supports arbitrary node-index sets rather than requiring contiguous spans.

Only layouts previously labelled `occurrence_partition_mismatch` receive this additional trace.
The preceding completion stage remains the baseline, with its successful exact outputs immutable.
Graph copying uses generated atom states and generated internal edges. It preserves cross-occurrence
edges and requires unchanged atom/edge counts, connectedness, validity and exact full-depth program
reconstruction. The same explicit TRAIN-novelty and within-family multiplicity admission rule
selects among the bounded proposals. Source structures provide correspondence context; their
variable atoms or bonds are not substituted into generated products.

All **46, 101 and 105** previously mismatched layouts qualify in the discovery and two saved-seed
replication populations, respectively. These populations were already inspected in earlier work;
they are **not new heldout evidence**. The method, inputs, three population configs and acceptance
rule were frozen before inspecting their occurrence-completion outcomes in
`configs/multireaction/combinatorial_occurrence_reuse_protocol_v1.json`. Acceptance requires positive
additional exact gain, an improvement in a previously untreated three-/four-step urea stratum,
and preservation of every declared count metric in each of the twelve families.

| Population | Attempts | Frozen model exact | Previous completion exact | Occurrence completion exact | Additional novel exact products |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | 768 | 378 | 576 | 584 | +8 |
| Saved-seed replication 1 | 1,536 | 771 | 1,137 | 1,156 | +19 |
| Saved-seed replication 2 | 1,536 | 769 | 1,128 | 1,146 | +18 |

Every gain is in urea; all other families are unchanged from the preceding completion pipeline.
All three populations preserve validity, novel-valid incidence, unique-valid count and effective
product count in every family. Existing exact and novel-exact outputs remain. Each additional
exact output is a distinct TRAIN-novel product within its family, and its recovered witnesses
all contain a TRAIN-novel precursor. These are constitutional membership comparisons, not proof
of chemical originality, realism, experimental feasibility or L2/L3 closure.

| Population and depth | Attempts | Exact before | Exact after |
| --- | ---: | ---: | ---: |
| Discovery, two steps | 32 | 18 | 25 |
| Discovery, three steps | 3 | 0 | 0 |
| Discovery, four steps | 29 | 1 | 2 |
| Replication 1, two steps | 50 | 26 | 36 |
| Replication 1, three steps | 17 | 0 | 1 |
| Replication 1, four steps | 61 | 0 | 8 |
| Replication 2, two steps | 50 | 20 | 30 |
| Replication 2, three steps | 17 | 0 | 5 |
| Replication 2, four steps | 61 | 0 | 3 |

The map makes deeper programs testable, but does not solve their generation. Of **133/288/309**
additional graph proposals, **87/145/193** fail the unchanged edge-count requirement; a further
**8/50/33** are invalid or disconnected. The remaining proposals incur **38/93/83** additional
strict assembly checks, of which **15/51/44** pass before admission. Proposal counts include
different donors yielding the same graph and must not be reported as distinct products.
Most remaining failures concern compatibility between generated occurrence interiors and the
connections left outside them. Increasing the permitted cycle count would not resolve the
scientific question and is not authorized by this result.

An independent count audit makes the connectivity obstruction explicit. With `m` occurrences and
generated internal edge counts `k_i`, copying donor `d` while fixing external edges changes the
total edge count by `m*k_d - sum(k_i)`. No available donor has zero change in **22/40**, **37/85**
and **50/94** attempts that receive proposals, respectively. Those attempts cannot pass this
completion stage's edge-count gate through donor choice alone. The audit reconstructs these
counts from the saved terminal graphs and agrees with every recorded edge-count rejection;
see `results/phase1/combinatorial_occurrence_reuse_validation_v1/connection_obstruction.json`.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_occurrence_reuse \
  --config configs/multireaction/combinatorial_occurrence_reuse_discovery_v1.json \
  --output-dir results/phase1/combinatorial_occurrence_reuse_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_occurrence_reuse \
  --verify results/phase1/combinatorial_occurrence_reuse_discovery_v1/result.json
uv run python results/phase1/combinatorial_occurrence_reuse_validation_v1/report_evidence.py
```

The output directory must be fresh. Per-attempt plans, proposals, admissions, original/final scores,
source/input pins and failures are saved. Verification reconstructs the source occurrences, every
proposal and admission, and every final score. The combined evidence report also checks the frozen
protocol and discovery replay. There are no new neural sampling or training calls in this stage.

**Decision:** retain the exact occurrence representation and bounded reconstruction gains. Preserve
the failed proposals and the prior cache contract. The next generative constraint must address
inter-occurrence connectivity jointly with the copied interiors while respecting the same atom,
cycle, validity and full-program gates. Arbitrary new-layout construction, heldout generalization,
chemical realism, complete-IL qualification of every library scope and repository-wide readiness
remain unproven. The broad goal remains active.

Occurrence-stage validation: **66 focused tests pass**, with no failures or skips. All 30 vendored
assets verify, all seven new Python files pass Black/Ruff, and whitespace checks pass. The full
suite reports **2,526 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,806 cases.
All 15 new tests pass in the full run; failure/error IDs exactly match the preceding graph-reuse
baseline. No failing check was relaxed or skipped. Global Phase 1 definition of done remains unmet.
All three occurrence results verify, and fresh discovery execution reproduces every saved artifact
hash and scientific field. Evidence SHA-256:
`a2d2e299b4aaafab27cdc8613c5629d2994c4b01329a25cefb66416e31c3e7c1`.
The validation result at `results/phase1/combinatorial_occurrence_reuse_validation_v1/result.json`
has SHA-256 `66adbc562e5a3ab29a02af71910084570f7f92cee214448b5b78c81de448d44a`.

## Joint precursor interiors and reaction connections (2026-09-14)

The next bounded completion addresses connections and repeated precursor interiors together.
It obtains an occurrence map by exact source-program replay, then qualifies the source edges
between occurrences and the remaining accumulator. Each such edge must join core positions with
the bond type specified in the vendored product template. The connections must form a tree over
the accumulator and co-reactant occurrences. This is **additional source-derived L1 program
context**, and is recorded explicitly. It does not establish construction of arbitrary new layouts.

For each generated donor occurrence, completion copies its generated atom states and internal
bonds to the other occurrences, replaces cross-occurrence edges with the qualified connections,
and retains the generated accumulator interior. It requires the original atom and edge counts,
fixed chemistry, connectedness, sanitization and exact full-depth identical-co-reactant program
check. The existing novelty and multiplicity admission rule applies, and existing invalid/exact
outputs are immutable. No new neural generation, training, heldout use or remote computation occurs.
The method and all three saved-run configs were frozen in
`configs/multireaction/combinatorial_connection_reuse_protocol_v1.json` before execution.

| Saved population | Attempts | Previous completion exact | Joint completion exact | Additional novel exact products |
| --- | ---: | ---: | ---: | ---: |
| Discovery | 768 | 584 | 599 | +15 |
| Replication 1 | 1,536 | 1,156 | 1,202 | +46 |
| Replication 2 | 1,536 | 1,146 | 1,197 | +51 |

All declared preservation metrics pass separately in **all twelve families**. Valid counts remain
750/1,495/1,500 and novel-valid attempt counts remain 719/1,435/1,437. Sums of per-family unique-valid
counts are 722→722, 1,421→1,421 and 1,420→1,421; effective product counts do not fall in any family.
Every added exact product is novel versus all TRAIN products, distinct within its family, and has
a novel precursor in every recovered witness. This remains explicit constrained admission using
TRAIN membership and current product multiplicities, rather than evidence of learned preservation.

| Reaction family | Discovery exact gain | Replication 1 gain | Replication 2 gain |
| --- | ---: | ---: | ---: |
| Amide coupling | +1 | +3 | +5 |
| Carbamate formation | +1 | +6 | +5 |
| Epoxide opening | +1 | +1 | +1 |
| Reductive amination | +2 | +1 | +1 |
| Urea formation | +10 | +35 | +39 |
| Remaining seven families | 0 | 0 | 0 |

Four-step urea improves from 2/29→9/29, 8/61→29/61 and 3/61→28/61. Three-step urea changes from
0/3→0/3, 1/17→7/17 and 5/17→7/17. These previously inspected TRAIN populations are descriptive
replications across saved noise/layout seeds, not independent holdout or across-training evidence.
All twelve families are evaluated; the extra gains in five families do not imply every family
improves at this stage. iPhos source correspondence remains ambiguous and abstains.

There are 175/429/454 additional proposals and 117/308/312 additional strict assembly checks.
Edge-count preservation rejects 58/121/142 proposals; the remaining nonexact program checks reject
89/203/204. Exact proposal counts of 28/105/108 include different donors yielding identical graphs.
The final 15/46/51 admissions are the distinct added products, rather than those proposal totals.

An independent audit reconstructs every accepted graph, checks its generated donor and accumulator
interiors, fixed chemistry and original atom/edge counts, and confirms all accepted graphs involve
connection changes. For 6/23/30 selected donors, copying the same interior while leaving connections
fixed would violate the edge-count requirement. This explains part of the observed recovery; it is
a diagnostic of admitted outputs, not an additional randomized causal comparison. See
`results/phase1/combinatorial_connection_reuse_validation_v1/evidence.json`, SHA-256
`ea6b6d02bc7afc1df5713ff60333f2a1565f148845561474a80e2a575f8efbde`.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_connection_reuse \
  --config configs/multireaction/combinatorial_connection_reuse_discovery_v1.json \
  --output-dir results/phase1/combinatorial_connection_reuse_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_connection_reuse \
  --verify results/phase1/combinatorial_connection_reuse_discovery_v1/result.json
```

Each output includes every plan, proposal, admission, original/final score and source pin. Verification
recomputes all scientific payloads. All three results verify, and fresh discovery replay reproduces
all scientific fields and artifact hashes. The replay-equivalence SHA-256 is
`baab5a3f77e61c38adab3366110553e7bfbdcf3cd666425172e695336ef967b1`.

**Decision:** retain the bounded connection completion and its negative cases. Source-supported
connections resolve some precursor-copy incompatibilities without weakening gates. Remaining
failures include generated precursor interiors, reaction chemistry and correspondence ambiguity.
The broader goal remains active: autonomous new-layout generation, chemical realism, complete-IL
qualification of every source scope, L2/L3 closure and repository-wide readiness are unproven.

Connection-stage validation: **117 focused tests pass**, with no skips, and all 30 vendored assets
verify. All seven new Python files pass Black/Ruff and whitespace checks pass. The full suite
reports **2,536 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,816 cases. All ten new
tests pass in that run; failure/error IDs exactly match the preceding occurrence-stage baseline.
No failing test was relaxed or skipped. The global Phase 1 definition of done remains unmet.
The validation report in `results/phase1/combinatorial_connection_reuse_validation_v1/result.json`
has SHA-256 `6b20edd4b76e5f58c716ab016e17431ede0b55d02131e4afc7089a9788ec6ddf`.

## Layouts without a source product: a rejected generation pilot (2026-09-14)

The next diagnostic separates layout construction from generation quality. A new compiler derives
semantic layouts directly from a generated product and its exact registry-supported reaction
program. It does not require a TRAIN product layout. Every witness is replayed to atom-origin and
core semantics under the existing policy. Distinct exact programs remain explicit alternatives;
ambiguous semantics within a witness or saturated enumeration cause abstention. Existing atom,
closure, vocabulary and full-depth identical-co-reactant reconstruction limits remain unchanged.

The audit processes all **768** saved discovery attempts and compares qualified layouts against
all **64,316 TRAIN records**. Qualified generated layouts occur in all twelve families, and eight
families have conditioning combinations absent from TRAIN under the sparse sampler's effective
inputs. Removing metadata that the model ignores matters: all twelve have new full sampler
signatures, which would overstate the change in conditioning if reported alone. These signatures
measure representation coverage, not product novelty or realism. Five otherwise exact products
abstain at the representation step.

A separate bounded expansion draws generated precursors from each registry role and recombines
them through the declared reaction program. It makes **256 draws per family**, seed **2026091601**,
and **4,740 forward expansion calls**. Every outcome and rejection is retained; any search
saturation abstains for the entire draw. This supplies new effective conditioning combinations in
nine families, leaving acetal, Passerini and thiol-Michael without a new combination in this
budget. These forward-assembled proposals are not neural outputs. Novel conditioning in every
family is an exploratory diagnostic, not a new requirement for the user's goal: a previously seen
count combination can still produce a novel molecule.

The existing generic count prior could not initialize on this cache because urea has multiple
reaction topologies at one depth. The new `CombinatorialLayoutPrior` reuses the same weighted TRAIN
count laws, support checks and adapter-fixed chemistry without assuming a unique nonfixed core
topology. It records the ambiguity: urea has one topology at depth one and two at each of depths
two, three and four. It fits **33 semantic bundles and 866 component-size support cells**. Sampling
works after the cache is closed, returns empty product SMILES and zero variable chemical targets,
and exposes neither source product layouts nor variable precursor graphs. Tests confirm identical
count draws to the previous implementation for the eleven unaffected families.

The frozen neural checkpoint then received **64 count-only layouts per family**, **32 flow steps**,
batch size 16 and two CPU threads, with layout seed **2026091603** and flow seed **2026091604**.
There was no training, recovery, filtering or retry. Every output received the original validity,
exact family-specific program, novelty and diversity assessment. The prespecified preservation
screen **failed**, so this prior is not a replacement for the existing pipeline.

| Reaction family | New effective contexts, saved generated products | New effective contexts, expansion | Count-prior valid / 64 | Count-prior exact / 64 |
| --- | ---: | ---: | ---: | ---: |
| Acetal | 0 | 0 | 62 | 1 |
| Amide | 0 | 26 | 59 | 8 |
| Aza-Michael | 6 | 130 | 42 | 13 |
| Carbamate | 2 | 29 | 63 | 0 |
| Disulfide | 25 | 124 | 64 | 20 |
| Epoxide opening | 3 | 62 | 62 | 0 |
| iPhos | 5 | 92 | 33 | 0 |
| Passerini | 0 | 0 | 63 | 0 |
| Reductive amination | 10 | 178 | 63 | 7 |
| Thiol-Michael | 0 | 0 | 64 | 0 |
| Ugi | 2 | 94 | 64 | 58 |
| Urea | 5 | 88 | 55 | 0 |

| Generation/completion arm | Attempts | Valid | Exact | Sum of per-family unique-valid products | Novel-valid attempts | Sum of per-family unique novel-exact products |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-layout neural baseline | 768 | 750 | 378 | 721 | 719 | 327 |
| Existing completed pipeline | 768 | 750 | 599 | 722 | 719 | 548 |
| Count-only neural layouts | 768 | 694 | 107 | 694 | 690 | 103 |

This is a distributional comparison: layouts, tensor shapes and random draws differ between arms.
It does not isolate a causal effect of fixed reaction cores. Ugi retains 58 exact outputs with its
existing core constraints while several families without fixed cores fall to zero. That motivates
a paired test using the same sampled layouts with and without a registry-qualified core scaffold,
including all qualified topology alternatives. The test must preserve exterior slots and chemical
support, use equal neural budgets, and retain every failed attempt under the same gates.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_context_expansion \
  --verify results/phase1/combinatorial_context_expansion_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_count_layout_generation \
  --config configs/multireaction/combinatorial_count_layout_generation_v1.json \
  --output-dir results/phase1/combinatorial_count_layout_generation_fresh
uv run python -m experiments.phase1.multireaction.combinatorial_count_layout_generation \
  --verify results/phase1/combinatorial_count_layout_generation_v1/result.json
```

Fresh output directories are required. Both verification commands recompute scientific payloads;
expansion verification also verifies the source generated-layout audit. Fresh count-generation
replay reproduces all scientific fields and artifact hashes. Consolidated findings, input/source
hashes, replay checks and the old-prior failure are preserved in
`results/phase1/combinatorial_generated_layouts_validation_v1/`.

**Decision:** retain the compiler and the all-family count-prior interface as diagnostic tools;
retain the failed generation result and keep the existing completion pipeline. The active goal
still requires evidence of improved generation with all preservation conditions. This diagnostic
does not establish improved chemical realism, qualification of every library output as a complete
ionizable lipid, or L2/L3 closure.

Layout-stage validation: **72 focused tests pass** and all 30 vendored assets verify. All ten new
Python implementation/test/report files pass Black/Ruff and whitespace checks pass. The full suite
reports **2,549 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,829 cases. All thirteen
new tests pass in that run. Failure/error IDs and kinds exactly match the preceding connection-stage
baseline; no failing check was weakened or skipped. Global Phase 1 definition of done remains unmet.
The findings SHA-256 is `62268e8c4d816b9d3d26abf4172d7f11993e2eac1780884b6c696dbdcfc58c13`;
the validation result SHA-256 is
`b54de009696521390b681c736295f66f016e09cc8c339c4f357bdbc935d84a0d`.

## Paired reaction-core decoding and a ring-reservation defect (2026-09-14)

A terminal-core experiment now isolates added core constraints from neural randomness. It runs one
trajectory per layout, captures the terminal scores, and decodes those exact scores under both
conditions. An identity control re-decodes the original condition and must reproduce the frozen
sampler output. This avoids a confound in an in-trajectory fixed-mask comparison: the existing
sampler draws some noise only at variable coordinates, so changing fixed masks changes which random
draws reach exterior coordinates even under an identical seed.

The core prior uses all **64,316 TRAIN records** and their registry-replayed semantics, retaining
**34 typed core alternatives across 31 semantic bundles**. Urea's alternatives remain explicit.
It stores core atom states, core bonds and semantic labels, without product identity or precursor
interiors. Layouts retain their sampled exterior counts and existing atom/closure support. A core
that cannot be serialized within those slots abstains; there is one core draw, with no retries.
Ugi's existing fixed-core condition remains an identity control.

Both experiments use **64 layouts per family**, 32 flow steps, batches of 16, two CPU threads,
layout seed **2026091603**, core seed **2026091702**, and per-batch flow seeds **2026091703 + offset**.
Each experiment makes 768 neural attempts and retains all 768 outputs in each terminal condition.
There is no training, remote compute or heldout evaluation.

| Experiment | Valid, baseline → core | Exact, baseline → core | Novel-valid, baseline → core | Unique-valid totals, baseline → core |
| --- | ---: | ---: | ---: | ---: |
| Core slots sorted within existing blocks | 722 → 550 | 148 → 342 | 720 → 542 | 707 → 528 |
| Supported core traversal shared by both arms | 736 → 704 | 293 → 374 | 722 → 689 | 717 → 684 |

The first experiment has **179** core serialization abstentions: epoxide 26, Passerini 64,
reductive amination 25 and urea 64. The follow-up transfers only a weighted, supported **core
traversal order** to both paired conditions. Entire blocks and core slots move; sampled exterior
counts and typed core graphs do not change. An independent labelled-graph isomorphism check verifies
that preservation for all 768 requests. All twelve families then fit all 64 sampled cores within
their original atom and closure counts. This is additional core-order context for both arms;
the two experiments themselves are not a paired causal comparison.

| Family, supported-order experiment | Valid, baseline → core | Exact, baseline → core |
| --- | ---: | ---: |
| Acetal | 64 → 32 | 24 → 30 |
| Amide | 61 → 61 | 25 → 31 |
| Aza-Michael | 48 → 48 | 17 → 17 |
| Carbamate | 64 → 64 | 26 → 28 |
| Disulfide | 64 → 64 | 50 → 50 |
| Epoxide opening | 61 → 61 | 4 → 5 |
| iPhos | 62 → 62 | 11 → 14 |
| Passerini | 64 → 64 | 1 → 55 |
| Reductive amination | 62 → 62 | 14 → 21 |
| Thiol-Michael | 64 → 64 | 59 → 60 |
| Ugi | 64 → 64 | 61 → 61 |
| Urea | 58 → 58 | 1 → 2 |

Passerini gains 54 unique novel exact products with unchanged validity, novelty incidence and
effective product count in this paired population. However, **both overall preservation screens
fail**. In the supported-order experiment, acetal loses 32 valid outputs, carbamate loses one unique
product and effective diversity, and reductive amination loses one novel-valid attempt. Both
experiments also fail preservation against the existing completion pipeline, which remains in use.

Every acetal validity loss in the supported-order experiment is recorded as
`fixed_closure_valence_exceeds_support`. Code inspection finds that the strict decoder reserves fixed
parent bonds before variable parent choices but adds fixed closure bonds afterward. A variable
choice can therefore consume valence already required by an immutable ring edge. The next bounded
fix should reserve all immutable edges before variable choices, keep the original valence limits,
and test a fixed-ring regression. No corrected-decoder outcome is claimed here.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_core_decoding \
  --verify results/phase1/combinatorial_core_decoding_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_ordered_core_decoding \
  --verify results/phase1/combinatorial_ordered_core_decoding_v1/result.json
```

Both commands recompute the paired experiment. Fresh replay of each experiment reproduces every
scientific field and artifact hash. The consolidated findings are in
`results/phase1/combinatorial_core_decoding_validation_v1/findings.json`, SHA-256
`dcdf2b4440d207ed42844d5b1924d6ea181ec1e4f3b49ca713db2339edc0bf5a`.
**Decision:** retain the all-family core/order compiler and the negative preservation results.
The broad goal stays active; no model promotion, heldout improvement, chemical-realism improvement
or complete L2/L3 closure is established.

Core-stage validation: **53 focused tests pass**, all 30 vendored assets verify, and all nine new
Python implementation/test/report files pass Black/Ruff. The final full suite reports **2,555
passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,835 cases. All six new tests pass in
that run. Failure/error IDs and kinds exactly match the preceding baseline. The earlier full run
is retained: it began before the ordering follow-up was staged and had two additional provenance
failures, resolved by staging the pinned files without modifying their checks. Whitespace checks
pass. Global Phase 1 definition of done remains unmet. The validation report is
`results/phase1/combinatorial_core_decoding_validation_v1/result.json`, SHA-256
`3ee016e7589e1e3c2189cdc04f3ea4af6eadaaf2dbe2a93e1ce70b8525fc11f7`.

## 2026-09-14 — Fixed-closure reservation and admitted source-core completion

Reserving **all immutable edges before variable parent choices** repairs the diagnosed ring
capacity error. The new decoder front-end selects feasible parents using the original neural
scores, then passes the topology through the unchanged atom, bond, valence and graph checks.
Records without fixed closures delegate to the frozen decoder exactly. No capacity is increased,
no atom or closure budget is enlarged, and impossible immutable valence still abstains.

The paired count/core experiment uses the same layouts, typed cores, neural trajectories and final
scores as the preceding supported-order experiment. All eleven non-acetal families are identical.
The correction restores all **32** lost acetal outputs and raises their exact reconstructions from
**30 to 59**. Across 768 attempts, the corrected core arm has **736 valid and 403 exact** outputs;
the paired unconstrained readout has **736 valid and 293 exact**. The raw core arm still fails the
preservation screen: acetal and reductive amination each lose one novel-valid attempt, and
carbamate loses one unique product and effective diversity. This raw arm is not promoted.

The subsequent source-core stage adds one bounded proposal per saved trajectory to the existing
connection-completion pipeline. It exposes the typed reaction core from that trajectory's existing
TRAIN layout; precursor exterior graphs are masked. The original neural inputs, trajectory and
terminal states reproduce exactly. The proposal uses the corrected decoder and the unchanged
full-depth family-specific program check. Admission uses the existing TRAIN-novelty and product
multiplicity rule. Original invalid and exact outputs remain unchanged; admitted corrections keep
the original atom count, edge count and fixed graph. Every proposal and rejection is retained.

All three run configurations were fixed before a numerical result was admitted. Two execution
errors were preserved with exact source/config snapshots: an invalid original SMILES reached an
inapplicable graph-size comparison, and a NumPy boolean could not be serialized as JSON. Revisions
skip that comparison for immutable invalid originals and convert the proof flag to a native bool.
The three-population budget, seeds, gates, admission rule and acceptance criteria did not change.
The successful protocol is
`configs/multireaction/combinatorial_source_core_completion_protocol_v3.json`.

| Saved population | Attempts | Exact before → after | Valid, unchanged | Novel-valid, unchanged | Within-family unique-valid total, before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | 768 | 599 → **640** | 750 | 719 | 722 → 722 |
| Replication 1 | 1,536 | 1,202 → **1,254** | 1,495 | 1,435 | 1,421 → 1,422 |
| Replication 2 | 1,536 | 1,197 → **1,262** | 1,500 | 1,437 | 1,421 → 1,422 |

Each of the eight preservation metrics is nondecreasing **in every family**, not merely in the
aggregate. All 41/52/65 admitted replacements are distinct within their family and saved
population, novel versus all cache TRAIN products, and have a novel precursor in every admitted
reconstruction witness. Counts across populations do not assert global product uniqueness.

| Family-specific exact reconstruction | Discovery / 64 | Replication 1 / 128 | Replication 2 / 128 |
| --- | ---: | ---: | ---: |
| Acetal | 61 → 61 | 124 → 124 | 122 → 122 |
| Amide | 53 → 62 | 98 → 110 | 94 → 106 |
| Aza-Michael | 43 → 43 | 93 → 93 | 96 → 96 |
| Carbamate | 61 → 61 | 111 → 115 | 114 → 117 |
| Disulfide | 54 → 54 | 112 → 112 | 103 → 104 |
| Epoxide opening | 51 → 57 | 91 → 98 | 97 → 110 |
| iPhos | 37 → 44 | 83 → 91 | 85 → 91 |
| Passerini | 45 → 57 | 92 → 105 | 99 → 112 |
| Reductive amination | 53 → 56 | 101 → 107 | 92 → 101 |
| Thiol-Michael | 54 → 58 | 118 → 120 | 115 → 123 |
| Ugi | 50 → 50 | 99 → 99 | 103 → 103 |
| Urea | 37 → 37 | 80 → 80 | 77 → 77 |

Exact reconstruction always uses the requested family and complete requested program depth,
including the same-co-reactant requirement across repeated steps. Acetal, disulfide, Passerini and
thiol-Michael retain their declared neutral-product/precursor scope; this table does not relabel
every library output as a complete ionizable lipid.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_reserved_core_decoding \
  --verify results/phase1/combinatorial_reserved_core_decoding_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_source_core_completion \
  --verify results/phase1/combinatorial_source_core_completion_discovery_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_source_core_completion \
  --verify results/phase1/combinatorial_source_core_completion_replication_1_v1/result.json
uv run python -m experiments.phase1.multireaction.combinatorial_source_core_completion \
  --verify results/phase1/combinatorial_source_core_completion_replication_2_v1/result.json
```

All four commands authenticate inputs and recompute scientific outputs. Fresh replay of the
reserved decoder experiment and of source-core discovery reproduces every scientific field and
artifact hash. Independent artifact accounting, exact input pins, per-family metrics, preservation
checks and changed-attempt indices are in
`results/phase1/combinatorial_reserved_core_validation_v1/findings.json`, SHA-256
`dc71be770b58a52d1f09ac7c3caee72991a1a3feae144ee6e1ea0f2fdc5294a2`.

**Decision:** retain the admitted source-core stage as a bounded pipeline improvement. These are
previously inspected saved TRAIN populations and a frozen model supplied with additional
source-derived core constraints. They establish neither learned or source-independent core
reconstruction, heldout performance, chemical realism, efficacy nor complete L2/L3 route closure.
Ugi reconstruction remains one of twelve family-specific checks. The broad goal remains active.

Validation: **62 focused tests pass**, all 30 vendored assets verify, and all nine new Python
implementation/test/report files pass Black/Ruff. The full suite reports **2,560 passed, 167 failed,
21 errors and 92 skipped/xfail** across 2,840 cases. All five new tests pass. Failure/error identities
and kinds exactly match the preceding core-stage run; this comparison does not diagnose every
historical failure. Whitespace checks pass; raw failed-test log/XML bytes are preserved in gzip
form without changing their contents or the whitespace rules. All execution failures and the
initial validation report remain recoverable. Global Phase 1 definition of done is not met.
Validation report: `results/phase1/combinatorial_reserved_core_validation_v1/result.json`, SHA-256
`b1c36d3434e20aca43f6ff821e24f931b7228f3170eb87700cdd1d4aadfcbcb9`.

## 2026-09-14 — Unified generation command and fresh-population verification

The supported generation path is now one restartable command. It runs the frozen graph proposal,
constrained admission, exact occurrence reuse, registry-qualified connection completion and
source-core stages in order. Each completed stage records authenticated configuration and result
pins; reuse verifies both the semantic output and its binding to the requested predecessor.
Interrupted attempts remain intact. A live file lock prevents concurrent writers from sharing an
output directory. Changed requests, substituted stage chains and altered product exports fail
verification.

```bash
uv run python -m experiments.phase1.multireaction.combinatorial_generation_pipeline \
  --config configs/multireaction/combinatorial_generation_pipeline_fresh_v2.json \
  --output-dir results/phase1/combinatorial_generation_pipeline_fresh_v1

uv run python -m experiments.phase1.multireaction.combinatorial_generation_pipeline \
  --verify results/phase1/combinatorial_generation_pipeline_fresh_v1/result.json
```

The first command reuses and verifies an already completed run at that destination. With a fresh
output directory it executes the same frozen request. The top-level
`qualified_products.jsonl` contains **every** exactly reconstructed final product, its family,
requested depth, precursor witnesses and target scope. Duplicate attempts are retained; products
are not ranked or turned into a prospective candidate panel. Original failures and all intermediate
proposals remain in the stage ledgers.

Both integration-replay and fresh-population requests were fixed before their outcomes. An import
formatting correction during the initial integration attempt triggered final source authentication;
no completed pipeline result was admitted for that attempt. Its exact code, configs and log remain
saved. The revised request has identical stage configurations, seeds, budgets and gates. Completed
stage receipts were imported only after exact configuration and parent-binding checks, then
semantically reverified. The replay reproduces the earlier **640/768** exact products, and **all
five stage artifact sets and scientific comparisons are identical** to their historical references.

The new population uses **64 attempts per family**, source-weighted TRAIN layout seed
**2026091901**, flow seed **2026091902**, 32 flow steps, batches of 16 and two CPU threads. It was
not selected for a favorable outcome and was not extended after inspection. Of 768 layout draws,
275 reuse a layout identity seen in the reference population: new seeds do not imply unseen source
structures. This is fresh generation with TRAIN context, not a heldout-component evaluation.

| Measure | Paired raw generator | Completed pipeline |
| --- | ---: | ---: |
| Attempts | 768 | 768 |
| Valid connected outputs | 755 | 755 |
| Exact full family-program reconstructions | 384 | **637** |
| Sum of within-family unique valid products | 713 | **716** |
| Novel-valid attempts versus all cache TRAIN | 723 | 723 |
| Sum of within-family unique novel exact products | 323 | **576** |

The admitted stages add **174**, **11**, **30** and **38** exact reconstructions, respectively.
The final 38 are the incremental source-core gain over the preceding completed pipeline. All eight
declared metrics are nondecreasing in every family, and every family has novel exactly
reconstructable products. Neither an aggregate improvement nor Ugi alone determines acceptance.

| Family | Valid outputs, unchanged / 64 | Exact before → after / 64 | Unique novel exact products after |
| --- | ---: | ---: | ---: |
| Acetal | 64 | 63 → 63 | 54 |
| Amide | 63 | 28 → 53 | 47 |
| Aza-Michael | 57 | 18 → 49 | 48 |
| Carbamate | 64 | 24 → 55 | 47 |
| Disulfide | 64 | 55 → 56 | 34 |
| Epoxide opening | 64 | 15 → 56 | 51 |
| iPhos | 61 | 11 → 41 | 41 |
| Passerini | 64 | 45 → 54 | 53 |
| Reductive amination | 63 | 13 → 51 | 46 |
| Thiol-Michael | 64 | 58 → 61 | 57 |
| Ugi | 64 | 53 → 53 | 53 |
| Urea | 63 | 1 → 45 | 45 |

These are unchanged full-depth, family-specific exact program checks, including same-co-reactant
consistency in repeated programs. Explicit TRAIN membership and population multiplicities enter
admission. Preservation concerns the eight specified metrics; no preservation of every possible
notion of chemical diversity is claimed. Additional bounded completion and semantic verification
incur compute costs, including neural replay in the source-core stage.

Evidence, all input pins, historical artifact equivalence and fresh per-family results are in
`results/phase1/combinatorial_generation_pipeline_validation_v1/evidence.json`, SHA-256
`1d8e8e2b839a287d42cc629b858d4825a9b85156253af2a3fbb602a716cc758c`.
The result establishes a runnable, improved twelve-library pipeline under declared source context.
It does not establish source-independent core generation, heldout generalization, calibrated lipid
realism, ionizability for neutral/precursor library scopes, or complete L2/L3 route closure.

Validation: **53 focused tests pass**, including all fifteen new orchestration tests; all 30 vendor
assets verify; the five new implementation/test/report files pass Black/Ruff and whitespace checks.
The full suite reports **2,575 passed, 167 failed, 21 errors and 92 skipped/xfail** across 2,855
cases. All fifteen new tests pass; failure/error identities and kinds match the preceding run.
The original integration traceback is preserved as byte-identical gzip data, and the first
validation report remains in an authenticated archive. Validation result:
`results/phase1/combinatorial_generation_pipeline_validation_v1/result.json`, SHA-256
`6316b978fc5594ffb37765f56d0f9552a5790f035f291d783e47729ce10ffeb4`.
The goal remains active because the repository-wide Phase 1 requirement `make test` still fails.
Further completion work must repair those failures or restore their authentic required artifacts;
the fresh pipeline improvement does not excuse weakening their gates.

## 2026-09-14 — Recover original validation artifacts without changing scientific pins

The twelve-family pipeline still verifies with its preservation screen passing. Repository
readiness work corrected the active installation smoke's stale project-file hash and restored
missing historical data. The prior smoke specification and archive manifest were preserved before
the engineering-only pin update; the scientific configurations, model, sampler and gates were
unchanged.

All five original graph-cache inputs match their frozen hashes. Deterministic replay reproduced
the original tensor archive exactly. The metadata needed only its gzip OS header byte restored;
all other bytes and the decompressed CSV were unchanged, and the complete historical SHA-256
then matched. Both artifacts are restored and tracked. Their original result JSON is unchanged.

The two original Ugi L1 product/atom annotation tables were recovered through the isolated
historical implementation, with all seven input hashes and all ten annotation gates passing.
Both CSV payloads matched after the same gzip header correction; the original expected hashes
were retained. The eight imported historical modules and their tree manifest are tracked in the
isolated replay directory. This recovers existing supervision rather than creating a new dataset
or changing the twelve-family generation result.

Validation has two successive scopes:

- After the smoke/cache repairs, all 24 engineering checks and four cache tests pass. The full
  suite reports **2,577 passed, 165 failed, 21 errors and 92 skipped/xfail**. Two previous failures
  disappear and no new failing test identities appear.
- After that full run finished, the annotation tables were installed. The affected group reports
  **29 passed and two failed**, resolving **16** previously failing tests. One remaining test now
  reaches an incorrect source-file path in the historical chemistry-interface audit; another
  still lacks the expanded-chemistry dataset. No later full-suite count is inferred.

All 30 vendored assets verify; the five recovery/report scripts pass Black/Ruff and whitespace
checks. Raw failure logs and the affected JUnit report are preserved as byte-identical compressed
artifacts. No test was changed or skipped. The dependency inventory records 58 named first-failure
paths and 42 messages requiring further tracing; it is not a complete transitive dependency audit.
Several source snapshots are recoverable, while the C16/C18 route replay still lacks original
evidence receipts. Those absences were not filled with replacement observations.

The goal remains active. Evidence and commands are in
`results/phase1/combinatorial_repository_readiness_v1/result.json`, SHA-256
`0cfaf381af17fb07296d463cd37723c80d657da0876d222d78ac801686d524be`.
