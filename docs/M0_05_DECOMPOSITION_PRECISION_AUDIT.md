# M0-05 Source-Grounded Decomposition Adjudication

Status: **automated source-evidence gate complete**

## Decision

Human chemist review is not a prerequisite for exact Ugi supervision. FORGE instead uses a
conservative literature-grounded chemistry adjudicator that:

- inspects primary articles and supplementary reaction schemes;
- hashes every source asset and records chemistry-page and procedure locators;
- verifies exact structures, component roles, reactive sites and forward reconstruction;
- separates source execution from deterministic transform consistency;
- abstains when source identity, procedure or reactive-site evidence is unresolved.

The earlier 319-case blinded packet remains frozen as an optional external audit. Its pending
annotations do not block training.

## Evidence classes

| Basis | Permitted use |
|---|---|
| Exact source-reported execution | Exact L1 or L2 supervision for the claim actually reported |
| Exact executed series member | Exact L2 supervision with general-procedure provenance |
| Exact reported experimental failure | Negative supervision |
| Deterministic transform consistency | Separately labeled L1 consistency objective only |
| Executed analogue or family precedent | Retrieval, template ranking and uncertainty only |
| Source conflict or missing procedure | Abstain |

Missing evidence is never a negative outcome. A rejected evidence claim does not mean the chemistry
is impossible.

## Supplementary-material workflow

The repository skill `$forge-adjudicate-chemistry-evidence` requires agents to:

1. render and visually inspect cited chemistry pages and neighboring pages;
2. follow every "prepared similarly" reference until the complete procedure is recovered;
3. extract exact reactants, products, source labels, conditions, workup, purification, yield and
   analytical evidence;
4. preserve discrepancies between schemes, prose, database records and normalized structures;
5. load reaction SMARTS and role policies from the frozen registry;
6. enumerate reactive sites and collapse symmetry-equivalent assignments;
7. emit a stable disposition and structured reason codes.

The deterministic gate then verifies all source hashes and machine-readable chemistry invariants.

## Frozen result

`make m0-05-adjudicate` produced 13,521 evidence records after applying the
blocking AGILE/LANTERN identity reconciliation:

| Stratum | Disposition | Count |
|---|---|---:|
| AGILE measured single-compound Ugi products | `admit_exact` | 1,100 |
| AGILE B4 cis/trans mixture executions | `abstain_mixture_single_graph` | 100 |
| AGILE virtual Ugi products | `admit_transform_consistency` | 12,276 |
| Source-extracted L2 route instances | `admit_exact` | 44 |
| Unresolved source identity | `abstain` | 1 |
| Reported experimental negatives | exact negative | 0 |

The separate L2 abstention is the exact upstream route to AGILE aldehyde B5.
LANTERN resolves the measured product as the pure-trans compound, but the
supplementary route still contains a chain-length and naming conflict. The
product identity is therefore resolved for the oracle while its exact upstream
route remains out of L2 training.

Four supplementary PDFs and 27 chemistry pages are hash-pinned in the result. The AGILE supplement
contains the executed esterification and oxidation routes for aldehyde tails and the
amine-to-formamide-to-isocyanide schemes needed for the primary Ugi component set.

## Claim boundary

The reconciled 1,100 single-compound records support exact, source-reported Ugi
L1 supervision under the frozen reactive-site and forward-reconstruction
policy. The 100 B4 mixture measurements remain valid nominal experimental
records but do not define a single product graph. The 12,276 virtual products
teach only transform consistency. They are not individual experimental
synthesis outcomes and do not provide L2 or L3 closure.

Non-Ugi R0 decompositions do not enter the joint L1 objective merely because they graph-round-trip.
They require separate source qualification under the same evidence policy.

## Artifacts and reproduction

- `results/m0_05_source_adjudication/result.json`
- `results/m0_05_source_adjudication/evidence_ledger.csv.gz`
- `configs/corpus/m0_05_source_evidence_adjudication.json`
- `.agents/skills/forge-adjudicate-chemistry-evidence/SKILL.md`

```bash
make m0-05-adjudicate
```

The command fails cleanly when a source supplement is absent, has the wrong hash, lacks visual
chemistry-page review, or violates any frozen expected count.

## Optional external packet

The earlier artifacts remain available:

- `results/m0_05/review_packet.csv`
- `results/m0_05/review_packet.html.gz`
- `results/m0_05/answer_key.csv.gz`
- `results/m0_05/sampling_manifest.json`

They may be scored later as an orthogonal external audit. They are not the admission authority.
