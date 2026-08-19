# RetroChimera proposal-backend preflight

Status: **API and artifact licensing verified; production binding blocked on
cryptographic provenance**  
Evidence checked: 2026-08-03

This preflight is deliberately narrower than a route-planner integration.
RetroChimera may propose single-step disconnections for bounded search. Its
outputs are not experimental evidence, cannot close a route, and are not
probabilities of synthesis success.

## Official release surface

The current public package is `retrochimera==1.2.0`, released on 2026-06-17.
The release is tied by PyPI Trusted Publishing to Microsoft repository commit
`a6b518b138cd28d91b9011124f5da912a8dbb125` (tag `v1.2.0`). The package and
repository are MIT-licensed.

Immutable package artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `retrochimera-1.2.0-py3-none-any.whl` | 166,280 | `dd07426c16d997856a60d5914681bb4e46c3a7dd7605c57e4f5bca9c965fe11a` |
| `retrochimera-1.2.0.tar.gz` | 190,262 | `bf77de553bdb2f3ce58168f2e8bf014009875370176e01e334e52c2326b5559d` |

The official public inference API is:

```python
from retrochimera import RetroChimeraModel
from syntheseus import Molecule

model = RetroChimeraModel(model_dir="/local/checkpoint/directory")
predictions = model([Molecule(target_smiles)], num_results=maximum_proposals)
```

Each prediction is a Syntheseus `SingleProductReaction`; its reactants are a
bag of `Molecule` objects. RetroChimera 1.2.0 exposes both an ensemble `score`
and a `probability` in reaction metadata. Inspection of the released source
shows that `probability` is a temperature-scaled softmax over only the returned
top-k results. It therefore changes with the returned set and must not be
treated as a synthesis-success probability. FORGE should retain only the
opaque ensemble `score` in `SingleStepRetrosynthesisProposal.model_score`.

Official sources:

- [PyPI release and API](https://pypi.org/project/retrochimera/)
- [Microsoft repository at v1.2.0](https://github.com/microsoft/retrochimera/tree/v1.2.0)
- [v1.2.0 inference implementation](https://github.com/microsoft/retrochimera/blob/v1.2.0/retrochimera/inference/retrochimera.py)
- [RetroChimera paper, arXiv:2412.05269v2](https://arxiv.org/abs/2412.05269v2)

## Released checkpoints

All three Figshare checkpoint records explicitly apply the MIT license to the
released software artifact. Figshare publishes an MD5 for each ZIP, but not a
SHA-256.

| Training corpus | Figshare record | Bytes | Published MD5 | Artifact license | Additional package requirement |
| --- | --- | ---: | --- | --- | --- |
| Pistachio | [doi:10.6084/m9.figshare.30591107.v1](https://doi.org/10.6084/m9.figshare.30591107.v1) | 4,213,968,927 | `50406d29b96b165a68fef73fa31448e3` | MIT | none documented beyond the base package |
| USPTO-50K | [doi:10.6084/m9.figshare.30601718.v1](https://doi.org/10.6084/m9.figshare.30601718.v1) | 284,852,815 | `f85766b7b2b8693213b429bfb7b20dd6` | MIT | `retrochimera[graphium]` |
| USPTO-FULL | [doi:10.6084/m9.figshare.30597563.v1](https://doi.org/10.6084/m9.figshare.30597563.v1) | 4,607,889,148 | `47d9f2e3be297d32ce50eb3b7e61c868` | MIT | none documented beyond the base package |

The paper identifies the Pistachio training snapshot as reactions present as
of June 2023, using the product-grouped folds prepared in the cited prior work;
that corpus is proprietary. It identifies a particular prior-work version of
USPTO-FULL (96,023 test reactions) because multiple incompatible processed
USPTO-FULL datasets exist. Neither the paper, package, nor Figshare records
publish a cryptographic digest of the exact processed training corpus used for
any checkpoint.

The MIT license on a checkpoint does not make the underlying proprietary
Pistachio records available, nor does it authenticate an exact USPTO
preprocessing snapshot.

## Production-manifest decision

`ProposalBackendManifest` correctly requires both:

1. `checkpoint_sha256`; and
2. `training_corpus_snapshot_sha256`.

The first can be populated after an authorized one-time download by hashing
the exact Figshare ZIP and retaining a signed acquisition receipt containing
the DOI, file ID, published MD5, byte count, local SHA-256, package version,
and extraction-tree hash.

The second cannot currently be populated truthfully from the public release.
An MD5 of the checkpoint, a DOI string, a template library, or a locally
written provenance note is not a hash of the training-corpus snapshot.
Consequently:

- do not instantiate a production manifest with placeholder hashes;
- do not use `checkpoint-license-pending-audit` (the checkpoint license is
  resolved as MIT);
- do not claim an exact reproducible training snapshot for the Pistachio model;
- do not admit any RetroChimera output through the production proposal engine
  until corpus-snapshot provenance is resolved.

Resolution requires either an authoritative corpus-snapshot digest/manifest
from the model authors or a separately reviewed change to FORGE's provenance
contract. This preflight does not authorize the latter.

## Exact optional backend seam

Add no logic to `forge.route.proposal_engine`. A future optional module should
implement the existing protocol:

```text
src/forge/route/retrochimera_backend.py
    RetroChimeraProposalBackend(SingleStepProposalEngine)
        manifest -> ProposalBackendManifest
        propose(request, maximum_proposals)
            -> tuple[SingleStepRetrosynthesisProposal, ...]
```

The adapter contract should be:

1. **Offline and local-only.** Accept a provisioned local checkpoint directory
   and acquisition receipt. Never download weights or contact a service during
   inference.
2. **Fail closed on identity.** Require `retrochimera==1.2.0`, verify the pinned
   package artifact/commit, checkpoint archive SHA-256, extraction-tree hash,
   frozen inference-policy hash, and typed `ProposalBackendManifest` before
   loading a model.
3. **Isolate dependencies.** The official reproducibility environment pins
   Python 3.9.7, RDKit 2023.09.6, PyTorch 2.2.2/CUDA 12.1, and related compiled
   packages, whereas FORGE currently targets Python 3.10+ and newer RDKit.
   Compatibility has not been demonstrated. Prefer a locked optional
   environment or local worker boundary rather than mutating the core FORGE
   environment.
4. **Use caller-owned targets.** Convert only
   `request.target.canonical_smiles` to a Syntheseus `Molecule`. The backend may
   not change the request or manifest.
5. **Keep search bounded.** Initially hard-cap output at five proposals per
   target. The official release warns that lower-ranked outputs increasingly
   hallucinate and recommends no more than 5--10 without stringent filtering.
6. **Canonicalize deterministically.** Parse every returned molecule, apply
   FORGE's frozen constitutional/stereo-free identity policy, preserve
   multiplicity, sort the reactant tuple, remove duplicate reactant sets, and
   re-rank contiguously. Reject malformed or multi-product records.
7. **Retain rank score only.** Store finite `metadata["score"]` as an opaque
   `model_score`. Do not store `metadata["probability"]` as a success
   probability. Leave `predicted_reaction_class=None` unless a later typed and
   independently validated class mapping is added.
8. **Use deterministic IDs.** Hash the schema version, checkpoint identity,
   frozen inference-policy identity, caller request identity, canonical
   reactant multiset, and returned rank.
9. **Validate the batch.** Return only through `validate_proposal_batch`.

The inference-policy receipt must bind at least package version, adapter
version, device/dtype policy, model constructor arguments, parallel-submodel
setting, probability-temperature setting, maximum result count, and any
checkpoint-specific beam/application settings. The paper and README warn that
USPTO checkpoint reproduction requires dataset-specific inference settings;
Pistachio defaults must not silently be reused for USPTO.

## Proposal-only semantics

The adapter must never construct `RouteStepProposal`, `KnowledgeResult`, an
evidence tier, terminal availability, route closure, or a synthesis-success
probability. Every output remains a hypothesis and must pass the independent
FORGE screen:

- exact-L1 Ugi route root, supported component role, and qualified handle;
- exact, unique forward reconstruction by the qualified verifier;
- frozen operational-compatibility assessment;
- non-contradicted substrate-scope state;
- subsequent independent evidence and L3 terminal-material closure.

Even a screened proposal can only enter bounded search. It cannot contribute
route closure or synthesis value until independent evidence-bearing route
records establish those facts. RetroChimera's rank or score must never be fed
directly into `V_syn`.

## Activation gates

After provenance is resolved and before any production use:

1. provision one checkpoint in a locked environment and write the immutable
   acquisition/inference receipt;
2. run import, CPU/GPU, deterministic-repeat, malformed-output, and dependency
   isolation smoke tests;
3. benchmark a frozen set of exact known L2 reactions, held-family reactions,
   lipid-like targets, and adversarial negatives;
4. report top-k recovery, duplicate/invalid rate, exact-forward-unique pass
   rate, operational rejection/censor rate, latency, peak memory, and proposal
   yield after every independent screen;
5. confirm by test that serialized outputs always contain
   `evidence_tier=null`, `success_probability=null`, and
   `route_closure_authorized=false`;
6. compare against deterministic/template retrieval under the same bounded
   search budget before deciding whether the learned backend adds useful route
   breadth.

Until these gates pass, deterministic templates, exact retrieval, forward
verification, and evidence-bearing route records remain the production L2
backbone.
