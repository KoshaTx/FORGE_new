# Offline single-step proposal-backend decision

Status: **Graph2Edits selected as the first production candidate; isolated
macOS runtime smoke-qualified; no learned backend is activated**  
Evidence checked: 2026-08-03

This decision concerns only L2/L3 **route-hypothesis generation**. A returned
disconnection is not experimental evidence, cannot close a route and is not a
probability of synthesis success. Every proposal must still pass the existing
independent forward, operational, substrate-scope, evidence and terminal-leaf
checks.

## Decision

Use the official **Graph2Edits reaction-class-unknown USPTO-50K release** as
the first production-candidate backend, behind the existing
`SingleStepProposalEngine` protocol. It is the cleanest audited option because
the authors' repository co-locates:

- implementation code under the repository's MIT license;
- the exact raw and canonicalized train, validation and test files;
- the reaction-class-unknown `epoch_123.pt` checkpoint and training log; and
- the preprocessing, training and beam-search inference code.

Pin repository commit
`b12e97391c378e46e79e617d80017e95b21bba65`. The matching Syntheseus
checkpoint is explicitly described as released by the Graph2Edits authors,
and its Figshare record is MIT-licensed.

Use **RetroChimera 1.2.0 only as an optional quarantined diagnostic
challenger**. Syntheseus recommends it and its broad Pistachio training makes
it the most informative recall challenger, but no public release provides the
exact training-corpus snapshot digest required by FORGE. It must not implement
the production protocol, affect candidate selection or contribute to route
value until that provenance gap is resolved. See
[`RETROCHIMERA_PROPOSAL_BACKEND_PREFLIGHT.md`](RETROCHIMERA_PROPOSAL_BACKEND_PREFLIGHT.md).

The initial audit installed no model or package. A subsequent authorized
acquisition froze the official checkpoint and all six raw/canonicalized
USPTO-50K splits. Their inactive receipts are
`configs/route/graph2edits_proposal_backend_candidate_v1.json` and
`configs/route/graph2edits_usp50k_training_corpus_manifest_v1.json`. The
proposal-only adapter now verifies the checkpoint, corpus manifest, every split
and the checkpoint-selection log before construction. An ignored, isolated
Python 3.11 macOS arm64 environment has now passed import, checkpoint-load and
one nonbenchmark CPU smoke test. This does not activate the candidate: license,
production-device determinism and FORGE-specific benchmark gates remain open.

## Isolated runtime qualification

The qualified local environment is `.venv-graph2edits-py311-arm64`; the core
project `.venv` was not modified. The hash-bearing inputs are:

- `configs/route/graph2edits_runtime_py311_macos_arm64.in`;
- `configs/route/graph2edits_runtime_py311_macos_arm64.lock.txt`; and
- `configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json`.

The lock fixes `torch==2.2.2`, `numpy==1.26.4`, `rdkit==2024.3.6` and
`omegaconf==2.3.0`, together with all transitive dependencies and distribution
hashes. The two acquired Syntheseus wheels remain separately pinned by exact
path, byte count and SHA-256. The receipt authenticates the lock, wheels and
checkpoint, records the full installed-version inventory and carries a
self-hash over its canonical payload.

The dependency audit found one genuine incompatibility. An unconstrained
binary-only resolution selected `omegaconf==2.0.6`, whose historical
`PyYAML>=5.1.*` metadata is rejected by modern packaging tooling; `pip check`
therefore could not recognize a valid OmegaConf installation for Syntheseus.
The final lock uses `omegaconf==2.3.0` and its required
`antlr4-python3-runtime==4.9.3`. The latter has no matching wheel for this
platform and is installed from its hash-pinned pure-Python source distribution.
The final environment passes `pip check` without broken requirements.

The frozen `epoch_123.pt` checkpoint loaded on CPU as a 1,245,599-parameter
Graph2Edits model. A single hand-authored ethyl-acetate target (`CCOC(=O)C`),
which is not drawn from any FORGE benchmark or sealed holdout, was queried twice
for three proposals; normalized identities, order and opaque model scores were
identical. This is only a runtime smoke test. Its products are not route
evidence, were not evaluated for chemical quality and support no accuracy,
latency or synthesis claim. Receipt payload SHA-256:
`c69258f32735f9512c3bf60c4ac7849053eeaeaad0abc18d0ce32eb506c84b39`.

Portable verification from the isolated environment is:

```bash
PYTHONPATH=src .venv-graph2edits-py311-arm64/bin/python \
  scripts/qualify_graph2edits_runtime.py --verify-existing
```

The candidate manifest deliberately retains `activation_authorized=false`.
Neither the frozen 120-target benchmark nor any sealed holdout was accessed.

## Candidate comparison

| Candidate | License and artifact provenance | API and deterministic top-k | Reaction breadth and lipid-component fit | Disposition |
| --- | --- | --- | --- | --- |
| **Graph2Edits** | The official repository declares MIT for its implementation and contains `canonicalized_train.csv`, the other frozen splits and the released checkpoint at one pinnable commit. Syntheseus republishes the authors' checkpoint as [Figshare v1](https://doi.org/10.6084/m9.figshare.25047314.v1), whose record declares MIT and publishes MD5 `ee66e25c735e94bf1db14a5f49b00675`; a local SHA-256 is still required. The patent-derived dataset's reuse status remains part of institutional review rather than being inferred from the code license. | The author implementation performs deterministic autoregressive graph-edit beam search in `eval()`/no-grad mode when device, versions, seeds, beam width, edit limit and tie-breaking are frozen. Syntheseus exposes a practical batched API. | Trained checkpoint covers the ten curated USPTO-50K classes. Sequential graph edits can express common L2 operations around amines, aldehydes, alcohols, esters and amides. It is not lipid-trained, uses a learned leaving-group vocabulary and defaults to at most nine edit steps. | **Production candidate**, after the gates below. |
| **LocalRetro** | Syntheseus' 2023 checkpoint record is Apache-2.0, but the current upstream repository changed to CC BY-NC-SA 4.0 in January 2026; the `syntheseus-local-retro` package publishes no license metadata. The Syntheseus checkpoint was trained by Syntheseus, while upstream data are referenced through mutable Dropbox links without a snapshot digest. | Fast, practical Syntheseus wrapper. Ranked local-template application is deterministic once the environment is frozen. Its returned score only resembles a probability and is not normalized. | Local edits are attractive for long lipid-like chains and standard handle conversions. Coverage is limited to the packaged template vocabulary; unseen transformation templates cannot be proposed. | **Do not select** without a resolved code-license chain and exact training snapshot. |
| **Retro\*** | MIT search code, but the repository directs users to an unversioned Dropbox bundle for the one-step policy, value model, stock and test data, with no published cryptographic manifest. | `RSPlanner.plan()` is a multi-step AND-OR-tree search API, not a single-step proposal API. Reproducible ordering is not documented for the bundled policy/search stack. | Breadth comes from its separate template MLP. It supplies no special evidence for lipid-like components. | **Not a backend candidate**. FORGE already owns bounded recursive search; adopting Retro* would conflate search with proposal generation. |
| **AiZynthFinder public USPTO policy** | MIT code and a versioned [Zenodo model bundle](https://zenodo.org/records/11430881) with model/template MD5 values, but the release does not bind an exact processed training-corpus snapshot. | Mature offline `AiZynthExpander` API; template top-k can be frozen deterministically. It also brings a substantial planning stack FORGE does not need. | Broad USPTO template inventory is useful operationally, but still template-bounded and not validated on FORGE lipid components. No controlled evidence establishes superiority for this use case. | **Not selected**; same corpus-provenance gap and unnecessary planner duplication. |
| **ConRetroBert (2026 preprint)** | Repository packages USPTO-50K JSONL data, but checkpoints are mutable Google Drive links, no release is tagged and no license file is published. Some configs retain internal paths. | Runnable training/evaluation scripts exist, but there is no stable library or audited FORGE adapter. | The paper reports higher USPTO-50K accuracy, including a USPTO-FULL pretraining result, but this is not yet an independently controlled comparison and remains template-bounded. | **Promising research watch, not production**. It does not clear license, immutable-artifact or maturity gates. |
| **RetroChimera 1.2.0** | MIT package/checkpoints and versioned Figshare artifacts; checkpoint SHA-256 can be computed after acquisition. The exact Pistachio/USPTO training snapshot digest is not published. | Clean Syntheseus API and bounded top-k. Its released `probability` is a softmax over the returned set, not synthesis success. | Broadest public candidate in this audit and likely the most useful recall challenger, but Pistachio provenance is proprietary and it has not been validated on FORGE's component distribution. | **Quarantined diagnostic only** until corpus provenance is resolved. |

The controlled Syntheseus re-evaluation is also a reason not to choose by a
paper's top-1 number alone. Under uniform deduplicated evaluation on
USPTO-50K, Graph2Edits obtained 54.6% top-1 and 82.8% top-5 exact-match
accuracy; LocalRetro obtained 51.5% and 84.3%, respectively. Graph2Edits was
substantially slower in route search. FORGE must therefore benchmark accepted
proposal yield per wall-clock and planner-call budget on its own targets,
rather than treat benchmark accuracy as deployment evidence.

## Why Graph2Edits is only a baseline

The public checkpoint is small and drug-like relative to FORGE. Its
preprocessing rejects reactions requiring an `Add Bond` edit, learns a finite
leaving-group action vocabulary and, for USPTO-FULL training, drops leaving
groups observed fewer than 50 times. The default inference program limits a
route step to nine edits. These choices can miss valid transformations,
especially unusual hydrophobic substrates or handle-installation chemistry.

Long alkyl chains, branching, unsaturation and embedded ester chemistry may
also be outside the checkpoint's effective applicability domain even when the
reaction family is familiar. No candidate model supplies reaction conditions,
selectivity, purification behavior, procurement or exact substrate evidence.
Graph2Edits is selected because its provenance can be made exact and its
outputs are inspectable—not because its predictions are presumed correct.

The initial deployment question is therefore empirical:

> Does this frozen backend add forward-consistent, operationally admissible
> route hypotheses for FORGE component targets beyond deterministic templates
> and exact/analogue retrieval, at an acceptable compute cost?

## Exact integration contract

Do not change `forge.route.proposal_engine`. Add one optional adapter module:

```text
src/forge/route/graph2edits_backend.py
    Graph2EditsProposalBackend(SingleStepProposalEngine)
        manifest -> ProposalBackendManifest
        propose(request, maximum_proposals)
            -> tuple[SingleStepRetrosynthesisProposal, ...]
```

The adapter must satisfy all of the following.

1. **Local-only construction.** Require explicit local checkpoint and corpus
   receipt paths. Disable Syntheseus automatic checkpoint download and all
   runtime network access.
2. **Immutable identity.** Before model load, verify the Graph2Edits source
   commit, `syntheseus==0.8.0` (wheel SHA-256
   `c9bf6ea244badb209b7101a2d86b2b7ab40132b636e58bf09040dd2e7a66d32b`),
   `syntheseus-graph2edits==0.2.0` (wheel SHA-256
   `55656362070b429ba9b7ebd1bebfbfe3352db92f0ed263e9b368fd514a5d622d`),
   the full dependency lock, checkpoint SHA-256, corpus-manifest SHA-256 and
   inference-policy SHA-256.
3. **Corpus receipt.** Define `training_corpus_snapshot_sha256` as the digest
   of a canonical manifest containing path, byte count and SHA-256 for every
   exact file used for training, validation/checkpoint selection or a reported
   benchmark; for this release, bind all committed raw and canonicalized
   train/validation/test splits. Bind the repository commit and
   preprocessing-code hashes separately. A Git
   blob SHA-1, Figshare MD5 or DOI string is not a substitute.
4. **Caller-owned target.** Pass only
   `request.target.canonical_smiles` to the model. The adapter cannot rewrite
   `ProposalRequest`, its lineage or the backend manifest.
5. **Frozen inference.** Use the reaction-class-unknown checkpoint,
   `model.eval()`, inference/no-grad mode, fixed device and dtype, fixed beam
   construction, `max_edit_steps=9`, fixed oversampling factor, seeded random
   libraries, deterministic PyTorch algorithms and canonical tie-breaking.
   Record all fields in the inference-policy receipt.
6. **Deterministic normalization.** Parse each dot-separated result; remove
   atom maps; apply FORGE's stereo-free constitutional identity policy to each
   connected reactant; preserve multiplicity; sort each reactant tuple; reject
   invalid/empty outputs; deduplicate identical multisets; then rank by finite
   beam score, original beam rank and canonical multiset.
7. **Opaque score.** Store the finite beam path score only as `model_score`.
   It is not calibrated, must never populate `success_probability` and must
   never enter `V_syn` directly. Leave `predicted_reaction_class=None` unless
   an independently validated typed edit-to-family mapping is later frozen.
8. **Deterministic proposal ID.** Hash schema version, backend manifest,
   inference-policy digest, caller target identity and canonical reactant
   multiset. Do not include transient process state.
9. **Typed return.** Construct only
   `SingleStepRetrosynthesisProposal` records, re-rank contiguously and return
   through `validate_proposal_batch`.
10. **No evidence authority.** The adapter cannot create
    `RouteStepProposal`, evidence tiers, terminal-material claims, route
    closure, operational compatibility or synthesis-success probabilities.

The independent screen remains mandatory after inference. Exact unique
forward reconstruction is a proposal-admission criterion, not evidence that a
reaction will work. Only evidence-bearing records and L3 terminal closure may
later contribute route-completion value.

Because the selected checkpoint is reaction-class-unknown, the adapter cannot
name its own forward transform. A separate resolver must test only admitted,
hash-pinned upstream transforms and their permitted reactant-role assignments.
A proposal passes graph-consistency screening only when exactly one admitted
transform reconstructs the target uniquely. No matching verifier is a censored
`not_verified` outcome; multiple matching transforms are ambiguous. The
Graph2Edits edit path cannot verify itself, and the Ugi L1 assembly registry is
not a substitute for the upstream L2 transform registry. Passing this resolver
still does not establish substrate scope or experimental evidence.

That resolver is now implemented and frozen against 11 upstream L2 transforms
and 56 exact pair-and-target admissions; see
[`INDEPENDENT_L2_FORWARD_RESOLVER.md`](INDEPENDENT_L2_FORWARD_RESOLVER.md).
Its manifest remains inactive and unbenchmarked. It does not broaden any
exact-source transform into family scope.

The resolver is also bound into a nonexecuting three-lane benchmark runner by
`configs/route/single_step_proposal_benchmark_runner_binding_v1.json`. The
runner authenticates the frozen 120-target manifest and checks all activation
gates before any target loader or proposal engine can be called. Hidden exact
routes remain a separately hash-bound scoring input loaded only after the lane
output ledger is frozen. This completes benchmark plumbing, not benchmark
execution or backend qualification.

## Lipid specificity without a lipid-only whitelist

The proposal backend should not be restricted to reaction families already
reported in LNP papers. Such a whitelist would make publication density define
chemical support and would exclude transferable, standard organic chemistry.
Lipid specificity instead enters through a two-lane search and independent
qualification policy:

1. search exact lipid-component evidence, admitted family programs and
   retrieval before the learned lane;
2. let the frozen general backend propose additional bounded hypotheses;
3. rank and audit results by Ugi component role, required handle, preservation
   of long-chain, branched and unsaturated structure, chemoselectivity,
   operational compatibility, purification burden and terminal availability;
4. retain separate provenance states for exact lipid precedent, exact or close
   non-lipid precedent and model proposal only; and
5. require the same independent forward, evidence and L3 gates regardless of
   proposal source.

The production benchmark must compare (a) a strict lipid-precedented lane,
(b) the unrestricted learned lane and (c) the hybrid, under matched search
budgets. The initial Graph2Edits checkpoint is not fine-tuned on the current
local L2 inventory: 45 positive routes are too sparse and component-role
imbalanced, and the inventory has no experimental negatives. Lipid-specific
fine-tuning is reconsidered only after a larger component-family-disjoint
route corpus and negative-outcome set exist.

## Activation and evaluation gates

Before the adapter can enter production:

1. obtain institutional approval for the artifact/data license chain;
2. acquire the official checkpoint and exact repository data once, verify the
   published Figshare MD5, compute local SHA-256 and an extraction-tree hash,
   and write immutable acquisition/corpus receipts;
3. lock a compatible isolated environment rather than perturb the generator's
   environment;
4. demonstrate byte-identical top-k identities and ranks across at least ten
   repeated runs on the production device;
5. test malformed-output rejection, reactant multiplicity, canonical
   deduplication, stable ties, maximum-proposal enforcement and fail-closed
   manifest mismatches;
6. benchmark a frozen FORGE set containing known L2 routes, held reaction
   families, long linear and branched tails, cis-unsaturation, esters,
   aldehydes, formamides/isocyanide precursors, heterocyclic heads and
   adversarial incompatible targets;
7. report top-k known-route recovery, exact-forward-unique pass rate,
   post-operational-screen yield, novel-family yield, duplicate/invalid rate,
   latency, peak memory and accepted proposals per matched planner-call and
   wall-clock budget;
8. compare against deterministic templates plus exact/analogue retrieval under
   the identical bounded-search budget; and
9. freeze the acceptance thresholds before reading the final benchmark or
   prospective outcomes.

Until these gates pass, deterministic templates and retrieval remain the
production L2 backbone. The learned model is an additional hypothesis source,
not a replacement for the route evidence system.

## Primary sources

- [Graph2Edits paper](https://www.nature.com/articles/s41467-023-38851-5)
- [Graph2Edits official repository](https://github.com/Jamson-Zhong/Graph2Edits)
- [Official reaction-class-unknown checkpoint directory](https://github.com/Jamson-Zhong/Graph2Edits/tree/master/experiments/uspto_50k/without_rxn_class/27-06-2022--10-27-22)
- [Official committed USPTO-50K splits](https://github.com/Jamson-Zhong/Graph2Edits/tree/master/data/uspto_50k)
- [Syntheseus single-step model documentation](https://microsoft.github.io/syntheseus/dev/single_step/)
- [Syntheseus controlled re-evaluation](https://arxiv.org/abs/2310.19796)
- [LocalRetro paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8549044/)
- [LocalRetro official repository and current license](https://github.com/kaist-amsg/LocalRetro)
- [Retro* paper](https://proceedings.mlr.press/v119/chen20k.html)
- [Retro* official repository](https://github.com/binghong-ml/retro_star)
- [AiZynthFinder documentation](https://molecularai.github.io/aizynthfinder/)
- [ConRetroBert official repository](https://github.com/JahidBasher/ConRetroBert)
