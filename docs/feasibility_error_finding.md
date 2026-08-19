# `FeasibilityError` cannot move to `core`

Recorded because it was proposed as a free win and is not one.

## The proposal

A classification pass reported that three `corpus/` modules import `FeasibilityError` from
`product/defog_feasibility.py` "purely as an exception type", so relocating it to `core` would
remove three cross-package edges at no cost.

## What is actually true

`FeasibilityError` is the shared error vocabulary of the entire sparse-representation layer:

- **9 modules raise it, across 151 raise statements** — `ugi_adapter_features`,
  `v5_sparse_representation`, `ugi_chemistry_interface`, `lipid_context`, `lipid_support_skeleton`,
  `sparse_topology_feasibility`, `lipid_morphology_audit`, `canonical_representation_audit` and the
  defining module itself.
- **2 modules catch it**, plus one test. `phase1_prelaunch_audit` wraps `_kekulized_molecule` and
  records a failure category on it; `ugi_generated_components` catches it alongside `ValueError` and
  `RuntimeError`; `tests/test_lipid_support_skeleton.py` asserts on it with `pytest.raises`.

So raise sites and catch sites live in *different modules*, and the class object's identity is what
couples them.

## Why moving it breaks things

`defog_feasibility.py` is hash-pinned and cannot be edited, so it will keep defining and raising its
own `FeasibilityError`. Defining another in `core` produces **two unrelated classes with the same
name**. Any raise/catch pair that ends up split across the two silently stops matching — and both
catchers use the exception to classify a failure rather than to abort, so the symptom is not a crash.
It is an audit quietly recording different counts.

The alternative, re-exporting the frozen class from `core`, is worse: it would make `core` import
`product`, and `core` must depend on nothing.

## Disposition

Leave it. The edge is structural, not incidental, and it cannot be removed while the defining module
is frozen. It would be resolved by unfreezing `defog_feasibility` — which requires re-running the
M0-06 artifact that pins it — and that is a scientific decision, not a refactor.

The general lesson: "imported purely as an exception type" is not a property of the importing module.
It is a property of the whole raise/catch graph, and it has to be checked across the package before
an exception class is treated as movable.
