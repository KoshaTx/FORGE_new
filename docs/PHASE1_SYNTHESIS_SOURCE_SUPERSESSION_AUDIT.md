# Phase 1 synthesis-value source supersession audit

## Decision

The candidate `src/forge/value/synthesis.py` source at SHA-256
`6e77063f45012144912883f16a12698eda809446635c94d59d1ab6211993f189`
is behavior-preserving relative to the historical source pin
`dba5b11067513e4f431f78f3b9a316dfe0d58c5f046210c170c55b9a0e241e3c`
over every replay target admitted by this audit:

- synthesis-value audits v1, v2 and v3; and
- the current fresh-pool route-coverage output v5.

All four replayed component ledgers and all four replayed product ledgers are
byte-for-byte identical to their frozen parents. All four scientific summaries
and all four results after provenance normalization are semantically identical.
The raw result files are intentionally not byte-identical because they record a
temporary qualified-config hash and the candidate source hash.

This finding does **not** update any historical pin, authorize production
guidance, qualify a synthesis-success probability or access the sealed holdout.

## Method

The development-only audit:

1. authenticates every frozen config, result and ledger against an explicit
   SHA-256 pin;
2. copies one frozen config into a temporary directory under
   `results/phase1/`;
3. changes exactly one JSON field in memory:
   `inputs.value_source.expected_sha256` for synthesis-value v1-v3 or
   `inputs.value_source.sha256` for fresh-pool v5;
4. invokes the unmodified historical builder;
5. compares compressed ledger bytes, decompressed ledger semantics, summary
   semantics and a result normalized only for config/value-source provenance;
6. deletes the temporary config and confirms no historical artifact was
   rewritten.

The implementation rejects repository-external, `holdout` or `sealed` input
paths. It also rejects any source-record shape change, source-path change,
additional compatibility flag or candidate-source hash drift.

## Exact replay results

| Replay | Component ledger expected = observed | Product ledger expected = observed | Summary semantic SHA-256 expected = observed | Normalized result SHA-256 expected = observed |
|---|---|---|---|---|
| synthesis value v1 | `7e16da6ed54609137917b4fb4c936db211662ec5c2ec92bff464624f4d95e297` | `324aca317d1a1a6262f05202d42d0c4e8b4f853779e5be3926e5f0bf60624cc5` | `63b538795bd20b5ce68fbe092bea56374449519fb75f1455ac62b323f98ca000` | `3f8e6d12485fb535d871b60e41406acf5633ea0581f511dce90b5dad62a0e856` |
| synthesis value v2 | `95e691068a67156551427dfebd1a82368ac7e55dbcdcd7cec98c9e468e01b846` | `ace4bd653c4c8bcac5d88692d989cc1ce4eac744b9fea3975fb13905239cba1b` | `854cdb752f244e756f6622183656cd157a299408664c0a83258751a0adb20004` | `919bf8ef6598d71de45bad725aea1a89942ed3e272bb85f2c32e44f642642891` |
| synthesis value v3 | `1e1b652b99d4560ede34a2ab95dd440eff6d02301955d3df1463441898957699` | `cedd42814b82f6e176b52bf9f8588ca9da728b57be5dd38f5fb8230cad18d2fe` | `bd756da0379293e716ff97af748c592eb502d70e38d203f3501ec31046f47ec9` | `f6f43c4826d9b6c4fc434149de9bdfe0e65457ad9ab4668a336a9f381d88d028` |
| fresh-pool route coverage v5 | `570c3f43857e30d650e7ef6e21d25a5e0755766d41e6b0b91667b3fbe8a4cd63` | `1afc543d50e7b5d8230ae54ec8edae33c73933a8b55434634c21f88b532972b5` | `b52f67216a807836893f4b38cebde036199d4fa20cc443c1427a4a9265b5964e` | `e408f195b79c1addd6e93b5e4cfe9541e618b5b71c67493729e51ce49d625a23` |

Raw replay-result hashes differ from the historical result hashes only in the
provenance fields that the normalized comparison removes:

| Replay | Frozen result SHA-256 | Qualified replay result SHA-256 |
|---|---|---|
| synthesis value v1 | `a1137d14b876a3dc53167e28dbfb55b1fb54acd845d3cdffe37d25e6aa7d7f27` | `01ae50e16d14f87fec31007d5feab9662d3d4cc7d5fae6370920d5c0d3ac7a23` |
| synthesis value v2 | `d498aeec5588f3b3791e67f3122c522f1e47a89418865e3a175c3d6498de111d` | `471f1d87b1cac6a364cfa841513e15b6e9a4c2cf9efe602831ba796e873a4738` |
| synthesis value v3 | `75dceb84bef3f77694fcb3262b89d18ceb975e3d506df26bafda5b2803a5fcec` | `23ac5b56b0f80208c9c977457bb3380b226002710c89e054d1405869bb83330e` |
| fresh-pool route coverage v5 | `6f80724272a0e82a6bde6ce6581ce0383a016b452bc132a86a6289b51f34bc95` | `a0225e1aa51352423270fd6fd669b7299131c7ba4db6b861aeb239f76c8a222c` |

The frozen audit config is
`configs/route/phase1_synthesis_source_supersession_audit_v1.json` at SHA-256
`d8c5f90b58f7fe61ead7a991e6225e4c8506fbb3249ba7f827f3a2edfa5b33f4`.
The result is
`results/phase1/synthesis_source_supersession_audit_v1/result.json` at SHA-256
`a07c01b247f1c47e8596112bec99a8c6035cb03eae9e39072b58d443a7cde7bd`.

## Replay boundary

Fresh-pool v4 cannot be replayed by a single top-level source-hash
substitution. Its builder independently re-executes the immutable exact-C18
route audit, and that nested exact-route config also pins the historical
synthesis-value source. Rewriting the nested config or bypassing its ownership
check would invalidate lineage. Fresh-pool v5 is replayable because it consumes
the owned v4 ledgers and validates the exact-C16 assessment rather than
re-executing v4.

This is a provenance boundary, not evidence of behavioral drift.

## Minimal explicit vNext chain

1. Freeze this supersession audit as independent development evidence.
2. Create new exact-C18 and exact-C16 route qualification versions that pin the
   candidate source and reproduce their historical scientific outputs. Do not
   edit the old exact-route configs.
3. Freeze a new synthesis-value source qualification that owns this audit and
   both exact-route replays.
4. Create fresh-pool route-coverage v6 from immutable v5, binding the new source
   qualification as an input. Do not rewrite v1-v5.
5. Qualify the production zero-guidance and matched-budget chain against the
   v6 lineage before synthesis guidance is considered.

Until those steps pass, the correct operational state remains fail-closed.

## vNext chain outcome

The explicit chain above is now complete through prerequisite
requalification:

1. Exact-C18 and exact-C16 were replayed under the candidate source without
   changing their historical configs. Both step ledgers, assessments,
   summaries and normalized results reproduced. The replay config is SHA-256
   `046df61553f3155226cf0a5c5a317573e35630309886d6d01386c2314b2f725b`;
   its result is SHA-256
   `855436b3eab7388a5f5b02578ce5b27105754cbdcb4f48b504d27c4ee101848d`.
2. The candidate source and the compatibility and exact-route evidence were
   frozen under one source owner. The qualification config is SHA-256
   `8d57ccb8a8836a66d352333681486a12eddea6a9bdbbc9b3a2eb0826aad2bfc6`;
   its result is SHA-256
   `7b72a64e2bc53fbe55aed6751610e9426c735a41553dc95556da123269fd2ad4`.
3. Fresh-pool v6 was created from immutable v5 values. Its component and
   product ledgers remain byte-identical to v5, and no route-coverage value
   changed. The v6 config is SHA-256
   `2a77a52ceaaa8f6e983ef80017d3d4a023707078d01416a3d4dbe16c7bc3f183`;
   its result is SHA-256
   `339d11287ad1489336c1e949dab001aad05a169fa97dbe48c15c2b9857fec722`.
4. The historical production zero-guidance identity result and the synthetic
   matched-budget scheduling contract were requalified as prerequisites
   against v6. The prerequisite config is SHA-256
   `3ac7e569c830a501d2c892ee5053e191745fc39381e7176b8949ab105d634335`;
   its result is SHA-256
   `ec9d139db61d1809e09adc0bd2d1027b7c40d302849db4418068b87e410eec64`.

The chain stops here. It does not qualify a current vNext cumulative-source
production runtime, freeze a synthesis scalar, authorize nonzero guidance,
select candidates or access the sealed holdout. The L3 window was current only
at the receipt's frozen assessment time; a future production execution requires
an explicit L3 refresh.

## Development-only readiness continuation

The next additive receipts close the source-lineage and static-schedule gaps,
but continue to stop before nonzero execution:

1. Four cumulative-source inputs were rebuilt under the qualified current
   synthesis-value source without changing scientific summaries or any ledger
   bytes. The immutable derivative config is SHA-256
   `ea07140ccfa668ad66804da368a489fe017b2e295bacbab68dc0b219a836928c`;
   its result is SHA-256
   `44f18334fa2fbc00af5fb510137e7b4785ce5fc33748d591d6d894c7320f3c65`.
2. The qualified cumulative source now has input identity
   `0fca91fae36da762eda53695405ec74cf5f9eacd19aee6ca3e4495a6a4d427c6`.
   Eleven authenticated L3 windows overlap from `2026-08-03T03:04:10Z`
   through `2026-08-09T05:16:00Z`. The decision horizon is frozen as
   `[2026-08-03T17:30:13Z, 2026-08-09T05:00:00Z)`.
3. All 12 locked beta-zero route values reproduce exactly when the qualified
   cumulative source is evaluated through each terminal's original
   support-boundary qualifications. The generator was not rerun. The combined
   L3/source/zero-guidance receipt is SHA-256
   `50367fffbeebf699f77ff8ee709da505ab7c2c3a856dbaa935740f35c6d56187`.
4. The conservative guidance utility is binary route completion: one only for
   exact L1 plus strict current exact closure of all three component roles,
   zero for substantively assessed nonclosure and null for censored states.
   Censored states contribute a neutral incremental log-weight of zero without
   receiving a numeric route value or evidence promotion. The 3,975-product v6
   census contains 203 supported, 3,772 neutral and zero censored products. The
   utility receipt is SHA-256
   `4650bedd6503004c09efb72365d6cc03e65e5b8572aef08037d9277ac9fc0893`.
5. A nonexecuting grouped SMC schedule freezes 16 exact morphology programs by
   four stochastic particles per program for each of three calibration and five
   evaluation seeds. All 128 programs are disjoint across seeds, all 512
   particle seeds are unique and ancestry is restricted to
   `(seed, program_index)`. The result is SHA-256
   `2c1bca3c7ab2f48c970840b84587fadaa597d361a2539f6443b1e951796951ca`.

The remaining prerequisite is a real selected-model 64-particle grouped
lambda-zero execution that is bitwise identical through closures, exact L1 and
all three synthesis checkpoints. Static schedule qualification and the
historical 12-terminal replay do not establish that property. Nonzero guidance,
candidate selection and sealed-holdout access remain unauthorized.

## Reproduction

```bash
.venv/bin/python scripts/phase1_audit_synthesis_source_supersession.py
.venv/bin/python scripts/phase1_replay_synthesis_source_exact_routes.py
.venv/bin/python scripts/phase1_qualify_synthesis_value_source.py
.venv/bin/python scripts/phase1_audit_ugi3_fresh_pool_route_coverage_v6.py
.venv/bin/python scripts/phase1_requalify_ugi_source_guidance_prerequisites.py
.venv/bin/python scripts/phase1_build_ugi3_source_qualified_cumulative_inputs.py
.venv/bin/python scripts/phase1_qualify_ugi_route_completion_utility.py
.venv/bin/python scripts/phase1_requalify_ugi_current_source_zero_guidance.py
.venv/bin/python scripts/phase1_qualify_ugi_grouped_smc_schedule.py
.venv/bin/python -m pytest -q tests/test_synthesis_source_supersession.py
```
