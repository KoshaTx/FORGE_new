# Reaction schemes across the 22-family assembly vocabulary

Figure 5 illustrates aldehyde Ugi 3-CR, aza-Michael acrylate and reductive amination.
The ten companion vector panels cover the remaining 19 families. These are reaction schemes
from existing TRAIN reference records, not model-generated samples or experimental outcomes.

From the repository root:

```sh
.venv/bin/python results/phase1/compose_lipid_iclr22_reaction_figures_v1/extract.py
.venv/bin/python paper/v1_iclr22/figures/all_family_reactions/render.py
```

`reaction_records.json` and `verification.json` in that results directory contain the source
pins, 22 exact assembly round trips, 34 registered transformation events, and intermediate
continuity checks. `chemistry_review.json` records the independent chemistry review.

The renderer uses ordinary RDKit molecules. `schematic.py` verifies exact registry replay and
replaces only unchanged, singly attached hydrocarbon fragments with arrow-local R labels. It
retains the registered reactive core, heteroatoms, rings, connecting paths and subsequent-stage
reaction sites. Every retained atom, hydrogen count, bond order and formal charge is checked;
each hidden R group and its attachment must be identical across the arrow. Unqualified
abstractions fall back to complete molecules. Repeated events are labeled; one event per repeated
stage is displayed. In the thiol-yne example the two separately indexed tail labels are identical.

`render.json` records source/code hashes, registry pins, per-arrow audits and vector-output
hashes. SVG and PDF preserve editable paths. PNGs are previews. Rebuilding uses RDKit and
`rsvg-convert`; the portable Overleaf package uses only the generated PDFs. Final page inspection
is recorded in `paper/v1_iclr22/visual_review.json`.
