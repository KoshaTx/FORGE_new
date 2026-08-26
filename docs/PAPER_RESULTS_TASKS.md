# Paper results execution tasks

Status: active ordered checklist, 2026-08-24.

This is the persistent execution order for completing the computational results in
`paper/v1/FORGE_ICLR2027_paper.tex`. Complete and verify one numbered task before starting the
next. A failed or null scientific result completes its task when the failure is authenticated and
reported. Gates are never relaxed to turn a negative result into a positive one.

The final conditioned FORGE replicates and finite-catalogue replicates are already complete. They
must not be retrained, seed-selected, or replaced:

- final FORGE source: `a154389d2f8109e2d91dc508c0d9ab06a10bbe61fbfa6411663a931bb3355c39`;
- final FORGE runs: `647a8c...`, `6f51e4...`, `cb1454...` for seeds 20260825--20260827;
- matched four-arm control runs: `891110...`, `305b54...`, `aeec35...` for the same seeds;
- finite-catalogue runs: `6ee5e5...`, `5a492f...`, `f20cbd...` for the same seeds.

Exact run IDs and artifact pins remain authoritative in each run's `run.json` and stage manifests.
Exact L1 replay is transform consistency, not synthesis-success probability.

## Ordered checklist

### 1. Final-arm production and catalogue adjudication

Status: **complete, 2026-08-24**.

Frozen result: `results/phase1/final_bl_core_production_adjudication_v1/result.json`.

- Extend the aggregate contract to accept the final one-arm
  `bl_core_constrained_repeat_aware` production results alongside the already completed Ugi-only,
  shared-null, cyclic-ID, and finite-catalogue controls.
- Verify paired seeds, 3,072 attempts per program, the common molecular support, immutable source
  inputs, zero route/oracle calls, no repair/retry, no candidate selection, coverage plus precision,
  and absence of the forbidden reductive-amination motif statistic.
- Preserve the historical aggregate as a frozen negative result. Write a new hash-pinned result
  under `results/phase1/` rather than rewriting historical evidence.
- Produce the values needed for `tab:production-comparison` and `tab:catalogue-comparison`, with
  seed means, sample standard deviations, and paired-seed bootstrap intervals for contrasts.
- Record the outcome in `docs/DECISION_LOG.md` and pass focused tests, formatting, `make verify`,
  and the repository's recorded test-baseline contract.

Completion gate: a runnable local command independently verifies all nine final, control and
catalogue runs and emits one complete, hash-pinned aggregate without retraining a model.

Outcome: all execution and provenance gates passed, every prespecified Ugi-retention gate passed,
and the aggregate preserved the null/cyclic comparisons as descriptive because those checkpoints
predate the final BL-core constraint. The historical negative aggregate was not modified.

### 2. Full program-semantic intervention

Status: **complete, 2026-08-24**.

Current checkpoint, 2026-08-24:

- The v2 final-checkpoint intervention, exact-H100 preflight descriptor, catalogue registration,
  and regression coverage are implemented.
- Local end-to-end smoke run `a1ce6cbd225c80ac7de593050d71bf646809409715a569722c1bb3ba3a4a6264`
  is verified: all three final checkpoints, all six paired conditions, 36 rows, and every execution
  gate passed.
- Exact-H100 preflight `a80832881ecdc5d8d326b334adb585399ea1911ad9be343941d21114a7038d96`
  and full run `a653dcea7db0ac2e745e4efc9738d1e4bb750cd70fe318ed7192ad33e937f4dd`
  completed and independently verified on source `c8260b6e...`.
- Frozen summary: `results/phase1/final_program_semantic_intervention_v1/result.json`.

- Reuse the three frozen final FORGE checkpoints. Do not retrain.
- With identical layouts and flow noise, compare factual program coordinates against wrong family
  IDs, cyclically wrong precursor roles, and null program coordinates.
- Run the full 3,072-attempt contract, not the two-attempt smoke result.
- Run a fresh exact-H100 preflight on the frozen executable source before paid inference.

Completion gate: all three final seeds are evaluated with paired attempt identity, and the result
reports output-change, validity, exact-L1, coverage, replay precision, ambiguity, and abstention
without treating molecule rows as independent replicates.

Outcome: factual exact-L1 yield was 39.69%. It fell to 3.62% when all program coordinates were
removed and to 17.09% under the LX family token. Forward and reverse precursor-role cycles caused
smaller consistent losses. The BL family-token contrast was weak and its paired-seed interval
crossed zero, so the result supports dependence on the joint coordinate system but not a universal
family-ID effect.

### 3. Transformer mechanism and FACT study

Status: **complete, 2026-08-25**.

Current checkpoint, 2026-08-25:

- Exact-H100 preflight `5a12915f236474ded3c1c1ccbe1186e0f4da42f304ea72105602cd264d1a0090`
  and all three production replicates completed and independently verified on source
  `c8260b6e691978e06d07e442ca25201f79a8a9ce8357a6147c3ad98c86aaacdd`.
- Verified production runs are
  `5f149b098c61a855dc1bf51b5ba1c3c31162d35fb5e84abeb174bde30a5eb54f`,
  `9f5633821b0234c367d7f1058703ba05034d2971c04ee45a7e9a26af83f311c1` and
  `3715b68bb7faaa8815ff575e07ff4c00f7983327322462ed8c2b463629172e46`.
- The hash-pinned 24-row seed ledger is
  `results/phase1/transformer_mechanism_study_v1/seed_rows.jsonl`; the partial paper renderer result
  is `results/phase1/transformer_mechanism_study_v1/paper/result.json`. The architecture table in
  `paper/v1/FORGE_ICLR2027_paper.tex` is populated from the generated rows.
- The full Transformer exceeded both FACT controls in every seed for Ugi and repeated aza-Michael,
  but both FACT controls exceeded it in every seed for repeated reductive amination. The retained
  conditional-dependence diagnostic remains negative, so FACT is reported as an architecture
  control rather than evidence for detected cross-role dependence.
- The finite-catalogue, shared-null, FACT-matched, FACT-generous and full-Transformer common Ugi
  assessments now pass for all three paired seeds. The manuscript renderer exposes parameter counts,
  1,700 optimizer steps, the $3\times3{,}072$ attempt budget and zero route/oracle calls for every
  mechanism arm. GPU-hours, sample steps and reaction-call counts were not recorded by the frozen
  runs and are reported as N/R rather than reconstructed.
- The completed production results remain frozen and were not retrained or seed-selected while these
  reporting artifacts were added.

- Completed three full exact-H100 replicates of the eight-arm study: full Transformer, input-only
  program, no role loss, no core loss, no routed adapters, no gradient-conflict control,
  FACT-matched, and FACT-generous.
- Replicate 0 was rerun on the frozen production source; earlier development runs remain excluded
  because they used changing executable fingerprints.
- Retain the completed negative conditional-dependence diagnostic. FACT is a control and cannot be
  used to claim detected cross-role dependence in the current Ugi data.

Completion gate: three paired seeds populate every retained mechanism row with common evaluation,
parameter counts, compute accounting, coverage, replay precision, novelty, and diversity.

Outcome: the gate is complete with unavailable historical compute fields explicitly reported N/R.
No arm is described as compute-matched on those unavailable quantities.

### 4. Learned finite-inventory selector

Status: **complete, 2026-08-25**.

- Run the three full exact-H100 replicates at seeds 20260825--20260827.
- Preserve the train-only inventory and zero out-of-inventory component ceiling.
- Use the common 3,072-attempt Ugi assessment with no repair, retry, route calls, oracle calls, or
  candidate selection.
- Obtain explicit approval before uploading private common molecular inputs to a new remote volume.

Completion gate: three passing common-assessment rows populate the learned-inventory baseline and
compute-parity tables.

Outcome: the exact-H100 preflight and all three full replicates passed. The frozen full run IDs are
`feb7af6d929fafb58b74ba9e5ff3ed83d0abf2c1357bb24cebe84572ebe823a7`,
`2024106a25c36eea9c2e9585bd59d345c29edf957b1452abe1631db1befa5eb0` and
`7e6e19c564269c904fe531938d6e50a96f9cf1c7a13210dbd571981657f5ca1d`. Every run retains the
train-only finite inventory and all 3,072 requested attempts. No output can escape that inventory
by construction.

### 5. Native external baselines

Status: **complete, 2026-08-25**.

Current checkpoint:

- RGFN completed all three native runs but emitted no generated molecular output under the strict
  matched native contract. The zero result is retained.
- The explicitly superseded GenMol/SAFE ledgers passed common assessment at all three seeds. Mean
  connected yield is 62.75%, but effective output count is only 3.25 molecules and exact Ugi-L1
  yield is zero; this is a severe-collapse negative result.
- All three DeFoG seeds completed, were hash-verified after download and passed both common exact-L1
  and structural-realism assessment. DeFoG produced $280.6\pm26.6$ exact-L1 products per 1,000
  attempts and $15.5\pm2.4$ held-component exact-L1 products per 1,000 attempts.

- Execute the already prepared pinned native requests for RGFN, DeFoG unconditional, and
  GenMol/SAFE at all three seeds, for nine full runs total.
- Train on the common train split and emit every requested attempt, including invalid or failed
  attempts. Do not substitute published values, released weights, repaired outputs, or a FORGE arm.
- Run each method in its pinned native environment and validate its request and receipt.

Completion gate: nine complete attempt ledgers pass the method-neutral common Ugi assessor and
populate the retained external-baseline rows. Ou DAG+Chem, SynFlowNet, and SynCoGen remain excluded
for their recorded implementation, reaction-arity, and licensing reasons.

### 6. Common evidence union and strict paper rendering

Status: **complete, 2026-08-25**.

Current checkpoint, 2026-08-25:

- Common molecular, exact-L1 and structural-realism assessment is complete across all three seeds
  for the finite catalogue, learned selector, RGFN, DeFoG, GenMol/SAFE, shared-null, FACT controls
  and FORGE Transformer.
- The completed-evidence renderer populates the main common benchmark, seed-level and decomposition
  appendices, finite-inventory coverage, available compute accounting and the method-blind
  structural-realism table. It also populates the exact-L1 column of the route-evidence table.
- The strict 25-page manuscript render passes LaTeX compilation and visual inspection with all
  currently admissible molecular, exact-L1, seed-level, decomposition, finite-inventory, compute and
  structural-realism results populated.
- The public method-blind union contains 11,313 distinct role/constitution components from all 27
  method/seed ledgers. The frozen exact-source library closes 29, explicitly abstains on every
  unsupported component and never reads the private method-membership ledger.
- The common route adjudication passes all gates. Under the deliberately non-exhaustive library,
  FORGE has zero complete dossiers and $786.8\pm100.8$ abstentions per 1,000 attempts despite the
  same exact-L1 yield. This is a negative evidence-coverage result, not proof of
  unsynthesizability. The paper reports it as such.
- The strict renderer now populates every retained route-evidence cell and contains no empirical
  TBD. A separate AiZynthFinder row was removed because proposal-only routes cannot change exact
  evidence closure.

- Assess shared-null and every retained method under the common Ugi molecular and exact-L1
  verifier.
- Apply the separate method-blind structural-realism diagnostic to every raw attempt ledger. This
  secondary, post-hoc analysis uses one source-study-held-out constitutional R0 reference and one
  frozen configuration for every method. Report fingerprint-manifold precision and coverage,
  descriptor-manifold precision and coverage, continuous descriptor/fingerprint distances,
  connectedness, uniqueness, Shannon effective molecule count and internal diversity. Invalid,
  failed, duplicate and out-of-support attempts remain in the requested-attempt denominator; QED is
  excluded. Aggregate by training seed with sample standard deviations, never by treating molecule
  rows as independent replicates.
- Freeze the method-blind union of exact role/constitution components only after every retained
  ledger exists.
- Complete or explicitly abstain on every union item, then adjudicate common L1/L2/L3 route-evidence
  closure. Do not call route closure a synthesis-success probability.
- Collect the hash-pinned seed rows and run the strict v1 renderer for tables, figure-data CSVs, and
  manuscript evidence records.
- Populate the already completed frozen Ugi route-cascade, synthesis-guidance, HeLa-guidance, and
  conditional-dependence results without rerunning them.

Completion gate: the strict renderer reports `complete`, no retained empirical row is TBD, and every
number resolves to a verified source artifact.

## Explicit non-blockers and manuscript cleanup

- No prospective synthesis, formulation, in-vitro, or in-vivo experiment belongs to this paper.
- The leave-one-reaction-family study is optional and never a hard gate. Either run its three seeds
  after task 6 as secondary evidence or remove its placeholder.
- The current role-specific held-component table asks for more than the production artifacts
  measure. Before final rendering, replace it with the supported component-disjoint reconstruction
  census and native held-component yield, or separately freeze and authorize a new role-conditioned
  experiment. Do not silently infer role-specific coverage from empty hit counts.
