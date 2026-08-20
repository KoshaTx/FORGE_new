# Experiments

Each active experiment is a small application: code, immutable JSON specifications, and local
configuration live together. `catalog.py` is the only allow-list consumed by the CLI.

- `_runtime/` executes DAGs and knows no FORGE stage implementations.
- `phase1/product_l1/` owns the current training and sampling pipelines.
- `phase1/synthesis_guidance/` owns the bounded matched-guidance workflow.
- `phase1/hela_potency/` owns the single authorized potency diagnostic.
- `archive/` keeps historical producers runnable without putting them on the supported interface.

Run `forge experiment list` to see supported specifications. New experiments must be registered in
`catalog.py`; do not add a top-level script or a workflow module under `forge/`.
