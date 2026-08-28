# Phase 1 product + L1

This application owns one scientific model and two explicit pipelines.

```bash
forge doctor phase1-training-smoke
forge experiment run phase1-training-smoke --profile smoke --resume

forge doctor phase1-sampling
forge experiment run phase1-sampling --profile smoke --resume
forge experiment reproduce phase1-sampling --profile smoke
```

The development-only tree-aware Ugi Transformer challenger has a separate local qualification:

```bash
forge experiment plan phase1-ugi-tree-transformer-challenger-smoke-v1 \
  --profile smoke --backend local
forge experiment run phase1-ugi-tree-transformer-challenger-smoke-v1 \
  --profile smoke --backend local
forge experiment reproduce phase1-ugi-tree-transformer-challenger-smoke-v1 \
  --profile smoke --backend local
```

Its full training contract is pinned in
`configs/model/phase1_ugi_tree_transformer_challenger_seed0_v1.json`; the matched development
ablation ladder is pinned separately. Neither file authorizes paid compute or constitutes evidence
that the challenger improves on v0.

The completed development checkpoints and frozen v0 reference have one matched molecular
calibration sequence:

```bash
forge doctor phase1-ugi-tree-transformer-calibration-h100-preflight-v1
# Paid Modal execution requires a separate explicit approval.
forge experiment run phase1-ugi-tree-transformer-calibration-h100-preflight-v1 \
  --profile smoke --backend modal

# After the preflight verifies, launch these five independent full jobs in parallel:
forge experiment run phase1-ugi-tree-calibration-dense-h100-v1 --profile full --backend modal
forge experiment run phase1-ugi-tree-calibration-relations-h100-v1 --profile full --backend modal
forge experiment run phase1-ugi-tree-calibration-consistency-h100-v1 --profile full --backend modal
forge experiment run phase1-ugi-tree-calibration-masking-h100-v1 --profile full --backend modal
forge experiment run phase1-ugi-v0-calibration-h100-v1 --profile full --backend modal

forge experiment adjudicate-tree-transformer-calibration \
  runs/phase1-ugi-tree-calibration-dense-h100-v1/<run-id> \
  runs/phase1-ugi-tree-calibration-relations-h100-v1/<run-id> \
  runs/phase1-ugi-tree-calibration-consistency-h100-v1/<run-id> \
  runs/phase1-ugi-tree-calibration-masking-h100-v1/<run-id> \
  runs/phase1-ugi-v0-calibration-h100-v1/<run-id>
```

All five jobs use the same 512 ordered calibration programs, flow and decoder seeds, eight reverse
steps, no repair or retry, and the same exact-L1, realism, local-chemistry and role-morphology
assessors. The adjudicator uses 10,000 paired program-index bootstrap resamples and may select a
model checkpoint, never a molecule. Held-component evidence is excluded from this decision.

`training/` implements cache preparation and resumable joint/closure training. `sampling/`
implements restartable, sharded terminal generation and exact Ugi-L1 admission. `stages.py` is the
thin orchestration adapter; it must not define chemistry or model policy.

The production training specification uses the same APIs with an explicit L4 resource envelope and
is never launched by smoke tests. Sampling does not retrain, retry/repair, call a route planner, call
a potency oracle, or lock candidates.
