# Backup candidate dossier — mean-ranked variant of the v6 rule

Twelve candidates that the frozen v6 rule does not reach, produced by re-running the frozen
selector with one change: the pool is ordered by the ensemble mean rather than by lcb90.
Eligibility, the physicochemical envelope, route completeness, the Tanimoto 0.95 ceiling, the
four-per-amine-head cap, the novelty tie-break and the canonical-SMILES tie-break are all
unchanged. The lcb90 run reproduces the frozen forty exactly (40/40), which is what makes this
comparison attributable to the statistic alone.

**This is a shortlist artifact, not a panel lock.** Nothing here is selected or frozen.

Supplier counts are from the 3 August 2026 procurement snapshot and are not live availability.

| ID | mean | lcb90 | novel | steps | product |
|---|---|---|---|---|---|
| N01 | 11.89 | 6.58 | no | 4 | `CCCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCC)NCCCN(CCCC)CCCC` |
| N02 | 9.62 | 3.41 | yes | 4 | `CCCCCCCCCC=CCCCCC(=O)OCCCCCCCC(NCCCN(CC)CC)C(=O)NCCCCCCCCCCC` |
| N03 | 9.36 | 3.14 | yes | 4 | `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCC)NCCN1CCCC1` |
| N04 | 8.78 | 3.47 | yes | 4 | `CCCCCCCCCCC=CCC=CCCC(=O)OCCCCCC(NCCCN1CCCC1)C(=O)NCCCCCCCCCCCCCC` |
| N05 | 8.25 | 2.03 | yes | 4 | `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCC)NN1CCOCC1` |
| N06 | 8.18 | 1.97 | yes | 4 | `CCCCCCCCCCCCCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCC)NCCCN(CCCC)CCCC` |
| N07 | 8.18 | 1.97 | yes | 4 | `CCCCCCCCC=CC(=O)OCCCC(NCCN1CCCCC1)C(=O)NCCCCCCCCCCCCC` |
| N08 | 7.96 | 1.75 | yes | 4 | `CCCCCCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCCCC)NCCN1CCCCC1` |
| N09 | 7.96 | 1.74 | yes | 4 | `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCOC(=O)C=CCCCCCCCC)NN1CCCCC1` |
| N10 | 7.59 | 1.38 | yes | 4 | `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCCC)NN1CCOCC1` |
| N11 | 7.58 | 1.36 | yes | 4 | `CCCCCCCCCCCCCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCC)NCCCN1CCOCC1` |
| N12 | 7.54 | 1.32 | yes | 4 | `CCCCCCCC=CCCCCCCCCNC(=O)C(CCOC(=O)CCCCCCCCCCCCCCCCC)NCCCN1CCOCC1` |

## Routes

### N01  (predicted mean 11.89, 4 steps)

Product: `CCCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCC)NCCCN(CCCC)CCCC`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `CCCCN(CCCC)CCCN` | purchase | catalogue | catalogue, 62 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCC(=O)OCCCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCC(=O)O` (113), `OCCCCCCCCO` (82) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCC=CCCCCCCCCN` (19) |

### N02  (predicted mean 9.62, 4 steps)

Product: `CCCCCCCCCC=CCCCCC(=O)OCCCCCCCC(NCCCN(CC)CC)C(=O)NCCCCCCCCCCC`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `CCN(CC)CCCN` | purchase | catalogue | catalogue, 46 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCC=CCCCCC(=O)OCCCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCC=CCCCCC(=O)O` (7), `OCCCCCCCCO` (82) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCN` (44) |

### N03  (predicted mean 9.36, 4 steps)

Product: `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCC)NCCN1CCCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCN1CCCC1` | purchase | catalogue | catalogue, 82 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCC(=O)OCCCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCC(=O)O` (107), `OCCCCCCCCO` (82) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCC=CCCCCCCCCN` (3) |

### N04  (predicted mean 8.78, 4 steps)

Product: `CCCCCCCCCCC=CCC=CCCC(=O)OCCCCCC(NCCCN1CCCC1)C(=O)NCCCCCCCCCCCCCC`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCCN1CCCC1` | purchase | catalogue | catalogue, 75 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCC=CCC=CCCC(=O)OCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCC=CCC=CCCC(=O)O` (2), `OCCCCCCO` (93) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCCCCN` (68) |

### N05  (predicted mean 8.25, 4 steps)

Product: `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCC)NN1CCOCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NN1CCOCC1` | purchase | catalogue | catalogue, 75 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCC(=O)OCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCC(=O)O` (101), `OCCCCO` (68) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCC=CCCCCCCCCN` (3) |

### N06  (predicted mean 8.18, 4 steps)

Product: `CCCCCCCCCCCCCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCC)NCCCN(CCCC)CCCC`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `CCCCN(CCCC)CCCN` | purchase | catalogue | catalogue, 62 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCC(=O)OCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCC(=O)O` (101), `OCCCCO` (68) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCCCCCCCCCCN` (38) |

### N07  (predicted mean 8.18, 4 steps)

Product: `CCCCCCCCC=CC(=O)OCCCC(NCCN1CCCCC1)C(=O)NCCCCCCCCCCCCC`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCN1CCCCC1` | purchase | catalogue | catalogue, 75 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCC=CC(=O)OCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCC=CC(=O)O` (18), `OCCCCO` (68) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCCCN` (52) |

### N08  (predicted mean 7.96, 4 steps)

Product: `CCCCCCCCCCCCCNC(=O)C(CCCOC(=O)CCCCCCCCCCCC)NCCN1CCCCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCN1CCCCC1` | purchase | catalogue | catalogue, 75 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCCCC(=O)OCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCCCC(=O)O` (88), `OCCCCO` (68) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCCCN` (52) |

### N09  (predicted mean 7.96, 4 steps)

Product: `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCOC(=O)C=CCCCCCCCC)NN1CCCCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NN1CCCCC1` | purchase | catalogue | catalogue, 67 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCC=CC(=O)OCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCC=CC(=O)O` (18), `OCCCCCCO` (93) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCC=CCCCCCCCCN` (3) |

### N10  (predicted mean 7.59, 4 steps)

Product: `CCCCCCCC=CCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCCC)NN1CCOCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NN1CCOCC1` | purchase | catalogue | catalogue, 75 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCC(=O)OCCCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCC(=O)O` (101), `OCCCCCCCCO` (82) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCC=CCCCCCCCCN` (3) |

### N11  (predicted mean 7.58, 4 steps)

Product: `CCCCCCCCCCCCCCCCCCCCNC(=O)C(CCCCCCCOC(=O)CCCCCCCCC)NCCCN1CCOCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCCN1CCOCC1` | purchase | catalogue | catalogue, 78 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCC(=O)OCCCCCCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCC(=O)O` (107), `OCCCCCCCCO` (82) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCCCCCCCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCCCCCCCCCCCCCCN` (38) |

### N12  (predicted mean 7.54, 4 steps)

Product: `CCCCCCCC=CCCCCCCCCNC(=O)C(CCOC(=O)CCCCCCCCCCCCCCCCC)NCCCN1CCOCC1`

| Role | Component | Action | Route | Starting materials (suppliers) |
|---|---|---|---|---|
| amine_head | `NCCCN1CCOCC1` | purchase | catalogue | catalogue, 78 suppliers |
| oxoester_aldehyde_body_tail | `CCCCCCCCCCCCCCCCCC(=O)OCCC=O` | synthesise | AGILE Tail A | `CCCCCCCCCCCCCCCCCC(=O)O` (158), `OCCCO` (108) |
| isocyanide_tail | `[C-]#[N+]CCCCCCCCC=CCCCCCCC` | synthesise | isocyanide from primary amine | `CCCCCCCC=CCCCCCCCCN` (3) |

