# Data Provenance

Every asset FORGE consumes is **vendored by content hash** from an external source. FORGE does not fork
or modify the source repositories — it pins them. Verified 2026-07-28.

`make vendor` copies local assets and fetches exact public assets from immutable revisions into
`data/vendor/`, then writes `data/vendor/MANIFEST.json`.
`make verify` re-hashes and fails loudly on any mismatch.

## Source roots

| Alias | Absolute path |
|---|---|
| `$COMPOSE` | `/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm_claude_lipid` (git worktree of `compose_rgm`) |
| `$DATASETS` | `$COMPOSE/artifacts/datasets/compose_lipid_pretraining_v1` |
| `$REG` | `$COMPOSE/configs/lipid_reactions` |
| `$AGILE` | `/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/lipid_diffusion/external/AGILE` |
| `$COMBINATORIAL` | `/Users/rmaganti/Desktop/thesis_projects_ML/combinatorial_papers` |
| `$REACTION_DATASETS` | `/Users/rmaganti/Desktop/thesis_projects_ML/reaction_datasets` |

## Assets

| Vendored name | sha256 | Bytes | Source |
|---|---|---|---|
| `lnpdb_fc7c389.csv` | `3493f27306419facd0958589030ed37f272f05c81ad47dd7bc1b8ce0284a57b3` | 20,118,803 | [`evancollins1/LNPDB` commit `fc7c389`](https://github.com/evancollins1/LNPDB/commit/fc7c38933b445eb54985014f2b8917606462eec1) |
| `r0_observed_real_structures.csv` | `3e2a35416c98f44e2261c06717c1fdee1b5ea393fb8013a38235631aa5ee6c0d` | 19,633,160 | `$DATASETS/r0_observed_real_structures.csv` |
| `r1_reaction_enumerated_support_v1.csv` | `8a691e47e24a921233a05627094627e54f1a5a404e998c11ee69a51716d62f3a` | 100,296,410 | `$DATASETS/r1_reaction_grounded_corpus_v1.csv` |
| `training_corpus_manifest_v1.json` | `03eafbbeac5d63b95df4cc488f884fef1c8e9eae0a7d13a1ba9a3df1005c4a39` | 4,229 | `$DATASETS/training_corpus_manifest_v1.json` |
| `corpus_fold_assignments.csv` | `92a39758bb35daf3590a83522dee31541b2586e1d18eef64a5cb403a7e50d369` | 2,460,121 | `$DATASETS/splits_v1/corpus_fold_assignments.csv` |
| `splits_manifest.json` | `51d2f714886cbed0982a47e7c8c78ad49ebe91383c80cf97ec7bfee0b582322a` | 8,322 | `$DATASETS/splits_v1/manifest.json` |
| `qualified_reaction_families_v1.json` | `961dea191bbf6c8d169aab09f6aee18082e51696e4056fc4dc9f6fc97ff82587` | 31,686 | `$REG/qualified_reaction_families_v1.json` |
| `qualified_reactions_v1.json` | `296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf` | 5,116 | `$REG/qualified_reactions_v1.json` |
| `ugi_3cr_building_blocks_v1.json` | `4ba3fec67b904fe507eec25215ce0c31a62fdc5b9536a3b33a5994acafc48f81` | 16,686 | `$REG/ugi_3cr_building_blocks_v1.json` |
| `building_block_pool_v1.json` | `7ad9699364f3bffe09019751e9437df5122becf0f4f6547f500fc849bbf08d18` | 195,786 | `$REG/building_block_pool_v1.json` |
| `AGILE_smiles_with_value_group.csv` | `1b3dd460125ba7d70d8cce266bd5febaabe09eeddc6a4622405c2706420b9686` | 196,871 | `$AGILE/AGILE_smiles_with_value_group.csv` |
| `agile_article_source_data.xlsx` | `e795d4e7ef1ec5c82b0358c626ab1eae059b76485a3ebdb7cbe5565aa6aac723` | 137,546 | [AGILE official article source data](https://doi.org/10.1038/s41467-024-50619-z) |
| `lantern_agile_curated.csv` | `d0ecfd94ec40a7f013af546113520175e70f78045af1798673c2680fe6f23c54` | 80,719 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/tree/11240f29ef92323649ae60d177b21df77e2d428b) |
| `lantern_agile_cis_trans_audit.csv` | `eceea1dcfc86063d0129d2b25918a17f5a0493bd3601641719f8d6ee29dff44c` | 22,696 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/tree/11240f29ef92323649ae60d177b21df77e2d428b) |
| `lantern_agile_random_split.npy` | `72d7cb6d5769c8b45f6634c530547c51147532539cfa2e728a7d630f24d699e8` | 3,341 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/data/splits/AGILE/random.npy) |
| `lantern_agile_murcko_split.npy` | `cd52f769972cde1e513a889ebfce0764017c56cd17032e3f9750b76f6e59632e` | 3,341 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/data/splits/AGILE/Murcko_scaffold.npy) |
| `lantern_agile_scaffold_balanced_split.npy` | `1f3720b6fc8ce2dfc3cf71f3e62cd394c74058d25641bf009eb7ddb146493c01` | 3,341 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/data/splits/AGILE/scaffold_balanced.npy) |
| `lantern_agile_circular.pkl` | `3a1331eeba6d16ccc9b12e53f68f957a1726a4b3c6ef4944cf97b6e353a0d619` | 18,132,419 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/data/fingerprints/AGILE/circular.pkl) |
| `lantern_agile_expert.pkl` | `23ba1fceaec5f10480ff7d14734d1ec70901f2e7680c129d88ee64405f9285a7` | 1,954,714 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/data/fingerprints/AGILE/expert.pkl) |
| `lantern_agile_mlp.pth` | `f419bfce435f1a08750b5d06902de9941b23343aa90d55aaefd6b3b2fa703382` | 4,500,961 | [`AsalMehradfar/LANTERN` commit `11240f2`](https://github.com/AsalMehradfar/LANTERN/blob/11240f29ef92323649ae60d177b21df77e2d428b/checkpoints/circular-expert-model-MLP.pth) |
| `uspto_50k.csv` | `1d69b90a299bd255b00342ccd15c3bc11d6c3047a3fecefaec539c3d04168dc9` | 5,788,436 | `$REACTION_DATASETS/USPTO_50K.csv` |
| `uspto_mit_data.zip` | `6d94a136e11f76fe464430cb95d1ae6db37b6ca352161ca4edddd9e6fe76a88a` | 49,713,357 | `$REACTION_DATASETS/uspto_mit_data.zip` |
| `agile_supplementary_information.pdf` | `ab21e9fa8ad4f2d991c97298c64232eec9942bce6b95f82419bfef67f5c830e9` | 12,499,955 | `$COMBINATORIAL/AGILE_supplementary_information.pdf` |
| `PMC11536852/publisher_page.html` | `f918e625753698706387783f831887ced721762a5cc6790598c6e681437e554c` | 474,753 | [JL_2024 article and Methods](https://doi.org/10.1186/s12951-024-02919-1) |
| `PMC11536852/12951_2024_2919_MOESM1_ESM.pdf` | `897d6925eb292113dcf5b4385acedac346545081986b1cd584b96b9823236427` | 7,619,117 | [JL_2024 supplementary information](https://doi.org/10.1186/s12951-024-02919-1) |
| `PMC11888472/supplementary_files.zip` | `ddba528699713ecb133b7c4b9c0582ca7877d8aacb4bbe924a352cb7ef47b7c5` | 3,176,809 | [XH_2025 PMC source package](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888472/) |
| `BORGHI_2024/sara2024.pdf` | `505cf441ccef01762325a008bf53da6d687962973579ab356cba616368608edf` | 8,287,754 | [Oxford Part II thesis reporting the exact 2-decyltetradecanol to 2-decyltetradecanal route](https://hla.chem.ox.ac.uk/protected/theses/part2/sara2024.pdf) |
| `PMC10723144/pnas.2309472120.sapp.pdf` | `51bdaccf337fe8f6d19bc9dbddf99ec6d1f312006d4d1ef2152a4fc2f6e4f1c6` | 2,765,858 | [JC_2023 native Ugi supporting information](https://doi.org/10.1073/pnas.2309472120) |
| `PMC10904786/41467_2024_45422_MOESM1_ESM.pdf` | `7f5b1c6413e4db3b09cb86296139ba76848b06de96b7165942643233ad6a3e2c` | 7,379,733 | [LX_2024 aldehyde-tail supporting information](https://doi.org/10.1038/s41467-024-45422-9) |
| `PMC10904786/bioc_full_text.xml` | `70a269e6fd7d5d132fd4b2344ec7f34f4124263e6665bf51ba7383a0ccf1d0ac` | 125,160 | [LX_2024 machine-readable main text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10904786/) |
| `rm_006_tail_1a.docx` | `f39b98e609328ff5fa3a6fe6506fca39f18b53cedcb0afa527921afce261c134` | 30,511 | `$COMBINATORIAL/RM_006_Tail_1a.docx` |
| `rm_007_tail_1b.docx` | `11d7f5753c7b1c1a8d02148d86e3f084c71158d4152685f9012c5a1ad9b21637` | 18,930 | `$COMBINATORIAL/RM_007_Tail_1b.docx` |
| `rm_008_tail_1c.docx` | `6b222e7bb2df23840cdede744b4754b61a99dabc6fe1e99e1b51b4a6c1f70959` | 46,828 | `$COMBINATORIAL/RM_008_Tail_1c.docx` |
| `rm_009_tail_1d.docx` | `ad8a3f153470e0b3b945c06213e0aab0bbaa0589d56c7aa05ac5fc1f822d15a5` | 46,832 | `$COMBINATORIAL/RM_009_Tail_1d.docx` |
| `rm_016_tail2b_corrected.docx` | `0a4f14cbae53c02921fc237e54fd7b399ce0e92204b22af1973c1c8c2065f45f` | 47,834 | `$COMBINATORIAL/RM_016_Tail2b_corrected.docx` |
| `rm_066_tail2a_bf3oet2.docx` | `f5fa383b1484e2c01512cf5664783d49ba64dbf3d89084fa45c518877058781c` | 51,922 | `$COMBINATORIAL/RM_066_Tail2a_BF₃OEt₂.docx` |
| `rm_067_tail2c_bf3oet2.docx` | `94d52884b8cf74b59d02b6678e53d9c73bafabc30865f1ca5e5889650078e9b0` | 51,997 | `$COMBINATORIAL/RM_067_Tail2c_BF₃OEt₂.docx` |

**Note the rename:** `r1_reaction_grounded_corpus_v1.csv` is vendored as
`r1_reaction_enumerated_support_v1.csv`. The original name asserts route-certification the data does not
have (see PLAN §5). Use the vendored name in all code and prose.

## What each asset is

- **R0** — 15,433 unique **real, measured** ionizable lipids. `molecule_enumeration_performed: false`.
  Sources: LNPDB 12,837 canonical (of 19,797 rows), LiON/LNP_ML 9,194, AGILE-1200. Carries leakage
  groups, region annotations, provenance, and prospective-lock status. **This is the realism anchor,
  but its AGILE-derived records and dependent M0-03 splits are quarantined from model training until
  rebuilt under the M0-07 B4-mixture/B5-pure-trans policy.** The rebuild must distinguish an
  individually measured single compound from a constituent observed only in a mixture.
- **Raw LNPDB** contains 19,797 row-level records from the immutable upstream revision. Its 12,837
  canonical lipid structures and all 19,797 `LNP_ID` values match the LNPDB subset of R0 exactly. It
  preserves the publication, experiment, lipid name, head, linker, and tail pairing that is lost when
  provenance is aggregated by canonical structure.
- **R1** — 464,265 products enumerated from 12 qualified transforms over a 490-block pool. Columns:
  `canonical_smiles, reaction_family, reactant_ids, reactant_roles, size_bin, charge_bin, ring_bin,
  n_tails_bin, tail_length_bin, linker_type, realism_weight`. **Reaction-enumerated support, not
  route-certified.** Family mix is Ugi-dominated (222,768 / 464,265 = 48%) — sample by `realism_weight`.
- **Reaction registry** — 11 families (+ Ugi-3CR in `qualified_reactions_v1.json` = 12) with
  atom-mapped reaction SMARTS, `required_handle_smarts`, `forbidden_smarts`, selectivity/stereo/
  protonation policies, known positive **and negative** examples, DOI sources, implementation hashes.
  **This is the chemistry source of truth. Never retype SMARTS from memory.**
- **Block pool** — 490 blocks. Provenance: `programmatic_rational` 390, `curated_literature` 75,
  `agile_measured` 25. The 79.6% programmatic majority is the defect M0-04 addresses.
- **Splits** — leak-free `held_reaction_family`, `held_scaffold`, `held_head`, `held_study`,
  `random_diagnostic`.
- **AGILE assay sources** — the root AGILE table carries 1,200 nominal Ugi-3CR library
  measurements with `A_smiles`/`B_smiles`/`C_smiles` components and `expt_Hela` /
  `expt_Raw`. Its response values agree with the official article source workbook. It
  is not itself an oracle-training table because B4 is a cis/trans mixture and the
  B5 single-compound graph requires correction to the pure-trans identity.
- **M0-07 reconciled AGILE oracle data** — the deterministic gate retains 1,100
  unique single-compound constitutional graphs, excludes the 100 B4 mixture
  measurements from single-graph supervision while preserving them in an audit
  ledger, and corrects the 100 B5 records to the pure-trans graph. LANTERN
  cross-validates HeLa labels and the B4/B5 identity policy. RAW labels are
  independently reconciled to the official AGILE workbook because LANTERN does not
  publish a curated RAW table. **AGILE remains a predictive general-transfection
  oracle, not an in-vivo endpoint oracle.**
- **Consolidated potency-study view** — new biological modelling code consumes one observation
  schema produced by `phase1-potency-study-corpus`, with raw LNPDB as its only row-level input for
  AGILE (`YX_2024`), `JC_2023`, and `LM_2019`. It preserves the LNPDB within-study endpoint z-score,
  canonicalizes the complete lipid graph under the constitution-only Phase 1 identity, and excludes
  all 200 B4 mixture observations through a hash-pinned component-label policy. It contains no
  reaction-program or component-supervision fields. The output is intentionally keyed by
  `(study_id, endpoint)` and targets are never pooled across studies or presented as raw assay
  measurements. Historical M0 results, the frozen raw-label HeLa pilot, and their reconciliation
  inputs keep their original bytes; this is the supported contract for new potency work.
- **Reaction-program views** use the same raw LNPDB catalogue for source rows, product graphs, study
  identity, and reported component columns. Registries, supplementary-information reviews, and
  source-specific corrections are evidence inputs: they decide whether a reported component or
  transform is admitted, abstained, or corrected, but they are not parallel observation tables.
  Exact atom-origin ledgers and balanced chemistry caches are derived training artifacts, not new
  sources. The virtual Ugi corpus remains a separate declared enumeration because those unmeasured
  products do not exist in LNPDB.
- **M0-07 LANTERN split artifacts** are retained exactly for reproduction and
  independent leakage auditing. The nominal `Murcko_scaffold.npy` artifact is
  not scaffold-disjoint: all six Murcko groups cross its train, validation, and
  test partitions. It is audit-only and cannot select a FORGE oracle. The
  separate `scaffold_balanced.npy` artifact is scaffold-disjoint and is an
  eligible structural holdout. The exact random split is a reproduction
  diagnostic only. LANTERN's validation partition is used as the calibration
  partition in the frozen FORGE contract.
- **USPTO-MIT** — 479,035 atom-mapped reaction records (409,035 train, 30,000 validation, 40,000
  test) for generic reaction pretraining. It does not carry reaction-class labels. Source:
  [`nips17-rexgen`](https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO).
- **USPTO-50K** — 50,016 class-labeled reactions used as a controlled retrosynthesis benchmark. It is
  retained separately from USPTO-MIT so a benchmark is not misreported as the large pretraining corpus.
  Source: [DeepChem's canonicalized release](https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv).
- **AGILE Supplementary Note 1** — upstream Tail A and Tail B synthesis routes in pages 1–7 of the
  [published supplementary information](https://doi.org/10.1038/s41467-024-50619-z). Final Ugi-3CR
  lipid assembly is L1 and is excluded from M0-09.
- **Cross-platform transfer sources** contain actual source-component
  reactions, not Ugi route labels. JL_2024 reports DCC/DMAP esterification of
  hept-6-ynoic acid with five named alcohols to give B12 through B16 in 62% to
  99% isolated yield. XH_2025 reports DCC/DMAP esterification of six named
  alcohols with a mono-2-ethylhexyl maleate acid to give asymmetric maleates in
  85% to 92% yield. Their article and supplement locators, conditions,
  purification, yields, and analytical-evidence states are hash-pinned in the
  transfer artifact. A proposed conversion of those alcohols into Ugi
  aldehydes is a separate reaction claim. One such route is now complete under
  the declared computational policy: the exact B16 alcohol is reported to
  yield 2-decyltetradecanal in 86% and the starting alcohol has current
  item-level US procurement evidence. The other six proposals remain
  incomplete, and no prospective Ugi outcome or biological label is inherited.
- **RM protocols** — seven internal custom acrylate/propiolate-tail protocols. The corrected RM-016
  document supersedes the older copy. Their yield, appearance, and LCMS fields are blank; missing
  outcomes are not treated as synthesis failures.

## Related structure corpora (not L2 reaction supervision)

- The Zenodo AGILE candidate file contains 12,276 unique, unlabeled final-lipid structures with
  computed descriptors (source file SHA-256
  `0b3c84321062efc7b5b4ef6a6954cc3340f22de61389418ad53e89e1c92719fe`; DOI
  [`10.5281/zenodo.11228093`](https://doi.org/10.5281/zenodo.11228093)). It is useful for
  representation learning and candidate-space evaluation, not as evidence for making tail blocks.
  A deterministic extraction retains only source-row identity and canonical SMILES in
  `data/derived/agile_virtual12k_smiles.csv.gz`. Its manifest preserves the complete 112 MB source
  hash, byte count, DOI, extraction schema, and derived-file hash. The compact derivative enables
  reproducible L1 decomposition without misreporting descriptor rows as reaction supervision. The
  resulting 93-component ledger supports a separate structure-program audit. Exact AGILE routes and
  reaction-family projections remain distinguishable in that audit, and procurement closure is
  evaluated separately. The corresponding terminal queue contains 33 proposed route leaves and 5
  unresolved heads. Historical source purchase claims are retained as evidence but do not establish
  current availability. A separate time-stamped item-level snapshot currently closes all 14 fatty
  acids, all six primary alcohols, the four shared diols, eight exact free-base primary amines, and
  three of the five initially unresolved heads. Three terminal candidates remain unresolved.
  High-purity oleylamine is catalogued, but the current review has not established accepted US
  stock or shipping under the frozen policy. The exact 3-aminoquinuclidine free base and
  1,1-dimethylhydrazine also remain unresolved pending explicit operational closure or a complete
  route from an accepted salt.
- R0 already carries source-specific region annotations. M0-09 counts the raw head, linker, and tail
  annotations so they can guide coverage priorities. Any block harvesting must use `R0_train` only
  and pass the M0-05 decomposition-precision audit.
- R1 remains useful as weighted **reaction-enumerated support** for product-prior ablations. It cannot
  train L2 because its block identities and `reactant_ids` were produced by the enumerator.

## Known measured facts (reproduce, do not re-derive from assumption)

| Fact | Value |
|---|---|
| R0 ∩ R1, exact canonical | 1,308 / 15,433 = 8.5% |
| …of which AGILE-1200 | 1,200 |
| Non-AGILE real lipids recovered by R1 | **108 / 14,233 = 0.76%** |
| R1 family mix | ugi 222,768 · passerini 77,571 · urea 55,032 · red-amination 21,654 · aza-Michael/epoxide/carbamate/iphos/amide 16,590 ea · acetal 1,989 · thiol-Michael 1,521 · disulfide 780 |
| R0 motif rates | amide 41.3% · aza-Michael 36.3% · epoxide 6.9% · iphos 3.7% |
| R0 kernel profile | Current hash-pinned file: median 53 heavy atoms, max 282; 70.48% at most 64; 91.23% at most 96; 344 rows contain F or Si outside declared C/N/O/S/P support. Stereo remains route-assigned under the flat-graph policy. |

**`reductive_amination` matched 99.8% of R0 and is a degenerate motif (essentially any C–N bond). It is
not reportable.**

## If the source paths are unavailable

`make vendor` fails with the list of missing local paths or any failed immutable public fetch, plus
the expected sha256. **Do not fabricate, synthesize, or download substitute data.** Report and stop.
Options for the human operator:

1. Run the agent on the original workstation where these paths resolve.
2. Copy `data/vendor/` to the execution environment out of band and run `make verify` only.
3. For tasks that do not need R1 (all of M0 except the M0-04 control), use `make vendor-partial`.
# Repository migration provenance

The current repository was initialized on 2026-08-18 from the former `forge` repository branch
`agent/active-design-20260815` at commit
`531d5a90694bb1dc3767861a07616f160fa0810e`. The former repository was left intact with its 238-commit
history and remote. Files were copied rather than pruned because a static reachability analysis could
not reliably identify result producers whose paths were assembled dynamically.

The one-time root `MIGRATION_NOTES.md` was removed after this provenance was incorporated here. Its
dated claim that 82 Phase 1 artifacts were unavailable is not an ongoing contract: several assets
have since been recovered. Current availability must be established from hash-pinned experiment
specifications with `forge doctor`, while historical source/config bytes are retained by digest under
`provenance/frozen-code/`.
