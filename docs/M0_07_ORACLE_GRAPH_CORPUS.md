# M0-07 graph-corpus and pretraining leakage gate

## Decision

The graph-oracle lane may tensorize the reconciled AGILE structures and the
frozen R0 pretraining ledger. It may not train from the raw AGILE table, the
unfiltered R0 file, the 12,276 virtual candidates, or the stale graph-support
profile in the upstream training manifest.

This gate does not freeze an oracle. It freezes the molecular identities and
support that the graph implementations are allowed to consume.

## Identity policy

RDKit 2025.09.6 sanitizes each connected molecule and produces canonical
nonisomeric SMILES. The identity preserves formal charge, protonation, and
tautomer state. No fragments are stripped. Stereoisomer rows collapse because
the M0-07 oracle representation deliberately excludes stereochemistry.

The retained training ledger contains only:

- a SHA-256 graph ID derived from constitutional SMILES;
- constitutional SMILES;
- atom count;
- undirected bond count;
- directed edge count.

It contains no assay, endpoint, study, provenance, component, or virtual
candidate field.

## Leakage result

The 15,433 eligible R0 rows contain 15,229 unique constitutions. All 1,100
reconciled AGILE target constitutions occur in R0 through 1,220 rows. Global
identity exclusion removes them before pretraining. The remaining 14,213 rows
collapse to 14,129 unique constitutions, with zero exact constitutional or
standard-InChI connectivity overlap with AGILE.

Provenance-only filtering is unsafe. Twenty LNPDB-only R0 rows reproduce AGILE
target constitutions without carrying `agile_measured1200` provenance. This
regression is tested explicitly.

The separate isomer-collapse ledger records all 184 duplicated constitutional
groups. One retained acid/anion pair shares standard-InChI connectivity but is
kept distinct because formal charge is an explicit graph feature.

## Actual encoder support

The retained R0 ledger contains 866,591 atoms and 1,726,776 directed edges
across 14,129 graphs. Molecules range from 14 to 282 atoms. Of these graphs,
9,602 have at most 64 atoms, 12,779 have at most 96 atoms, and 1,350 exceed 96
atoms. No truncation is authorized.

The observed element vocabulary is C, F, N, O, P, S, and Si. Atomic formal
charges are -1, 0, and +1. Bond types include single, double, triple, and
aromatic. The tensorizer must support all of them and fail on an unknown
feature rather than silently map it to a generic value.

## Manifest discrepancy

The pinned upstream manifest claims a maximum of 138 atoms, only C/N/O/P/S,
and no charged records. The pinned R0 instead reaches 282 atoms, includes F and
Si, and contains 509 net-charged records. Its stereocenter fraction also
matches none of the audited explicit or potential stereo definitions.

Therefore, the implementation derives graph support from the audited molecules
and records the manifest mismatch. It does not inherit the manifest profile.

## Next gate

Implement a pure-PyTorch sparse tensorizer and profile one train-only fold
before launching the full matrix. The first supervised lanes are:

1. whole-graph D-MPNN;
2. whole-graph edge-aware GIN;
3. Ugi-component-role-aware D-MPNN.

The first pretrained comparison uses whole-graph, label-free R0
self-supervision followed by frozen linear and MLP heads. Component IDs are
prohibited, and no artificial linker slot is introduced.
