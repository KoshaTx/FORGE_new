# Experiments

Each active experiment is a small application: code, immutable JSON specifications, and local
configuration live together. `catalog.py` is the only allow-list consumed by the CLI.

- `_runtime/` executes DAGs and knows no FORGE stage implementations.
- `phase1/product_l1/` owns the current training and sampling pipelines.
- `phase1/multireaction/` owns source-adjudicated repeated-program data plus the bounded,
  checkpoint-separated multi-family training/sampling smoke, fail-closed overfit qualification and
  full-corpus shared Ugi/aza-Michael/reductive-amination representation and model-integration gates.
  It also owns the nonlaunching matched-production design preflight and the execution-only exact
  A100-40GB versus H100 production-accelerator qualification, plus the matched train-only
  finite-component catalogue baseline. The BL/LX repair study is a separate calibration-only
  two-model by two-decoder diagnosis; it never reads heldout products. The production alternatives
  are both checked in before the source freeze; only the accelerator selected by the matched
  benchmark is launched.
- `phase1/synthesis_guidance/` owns the bounded matched-guidance workflow.
- `phase1/hela_potency/` owns the single-source LNPDB biological-data contract and the
  single authorized potency diagnostic.
- `archive/` keeps historical producers runnable without putting them on the supported interface.

Run `forge experiment list` to see supported specifications. New experiments must be registered in
`catalog.py`; do not add a top-level script or a workflow module under `forge/`.
