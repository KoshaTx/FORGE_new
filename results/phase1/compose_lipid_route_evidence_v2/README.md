# Bounded exact precursor evidence, 24 September 2026

One exact, source-executed AEMA component preparation is newly admitted at L2. It applies to the same
AEMA identity in 47 saved exact development requests. Its two source-route leaves still lack admitted
current terminal evidence. No current complete component or product dossier is promoted.

The five highest-priority component identities occur in 279 exact development requests spanning 16
families. The bounded vendor work identifies primary supplier candidates for all five. None of those
279 products has **all roles** covered by these five identities, even before availability and assembly
context qualification. The remaining six families are outside this five-component effort.

## Exact L2 admission

The existing user-supplied [Zhou 2016 SI](https://www.pnas.org/doi/suppl/10.1073/pnas.1520756113/suppl_file/pnas.1520756113.sapp.pdf),
DOI 10.1073/pnas.1520756113, reports AEMA preparation on physical PDF page 19. The complete preparation,
the exact AEMA drawing and monomer 1H NMR controls (Figures S1-S2), and neighboring pages were rendered
and visually inspected. The input PDF is SHA-256
`1b969c58dd4bfaca0a9bb6b3a558ac55d8b3bba31a08805d1eeae55367639987`.

The reported pair is 2-hydroxyethyl methacrylate plus acryloyl chloride. The unchanged registered
O-acylation transform returns one exact AEMA product with one reactive handle per component. All
nonleaving heavy atoms have one-to-one transform provenance, and elemental, hydrogen and charge inventories
balance with the registry's HCl byproduct. Both generic inverse candidates are retained; the SI
explicitly identifies which precursor pair was executed. The alternative reverse disconnection is
not mislabeled as source-executed.

The O-acylation transform's historical slot name `aminoalcohol_head` does not imply HEMA is an amine.
HEMA passes that unchanged slot's hydroxyl and competing-handle predicates. Only the transform and
its predicates are reused; Han source conditions, two-arm architecture, and execution evidence are
not transferred. No planner template or vendored registry was broadened.

The SI contains mass/mole discrepancies for HEMA and triethylamine. They are preserved in
[source_review.json](source_review.json). The exact identity and execution claim is admitted under
the repository's existing policy for disclosed nonidentity discrepancies. Numeric stoichiometry and
yield supervision is withheld; the reported 85% is not a recomputed or validated yield. The NMR
control contains BHT; no independent purity estimate is inferred.

The existing recursive assessor consumes this exact step and returns `missing_knowledge` with both
precursor leaves present and unassessed. A terminal snapshot and final-product context receipt are
not fabricated to make that partial tree complete.

## Vendor evidence and the access boundary

| Exact component | Development requests | Families | Primary supplier candidate | Current L3 |
| --- | ---: | ---: | --- | --- |
| N,N-dimethyl-1,3-propanediamine | 77 | 14 | [Sigma D145009](https://www.sigmaaldrich.com/US/en/product/aldrich/d145009), 99% | Unknown; fresh capture unavailable |
| 1-(3-aminopropyl)pyrrolidine | 72 | 14 | [TCI A2270](https://www.tcichemicals.com/US/en/p/A2270), >98% | Unknown; fresh capture unavailable |
| tert-butyl isocyanoacetate | 64 | 1 | [TCI D6366](https://www.tcichemicals.com/US/en/p/D6366), >98% | Unknown; US fulfillment also unresolved |
| AEMA | 47 | 1 | [ChemScene CS-0621230](https://www.chemscene.com/product/69040-48-8.html), ≥98% | Unknown; item-specific US availability absent |
| 1-(2-aminoethyl)piperidine | 33 | 14 | [TCI A1026](https://www.tcichemicals.com/US/en/p/A1026), >97% | Unknown; fresh capture unavailable |

Counts overlap across components. These are identity/specification retrieval candidates, not accepted
current terminals. The direct read-only requests returned HTTP 403 for all four TCI pages and
ChemScene; Sigma timed out. The live-browser fallback reported no available browser. Cached web
retrieval supplied the vendor fields, but no exact stock-observation timestamp or source HTML bytes.
The transcription records cache-age labels and does not pretend its own hash is a vendor snapshot
hash. Accessing cached material today does not reset availability expiry.

TCI D6366 requires contact for both US warehouses. Its Japan stock and generic shipping footnote
cannot establish applicable US fulfillment, particularly given frozen storage and the footnote's
ice-shipment exclusion. ChemScene's transport temperature and unrelated recommended-product stock
labels do not establish AEMA availability. No vendor was contacted and no transaction was initiated.

The concrete next evidence step is an accessible, read-only US supplier capture with exact item,
constitution/form, purity, stock or applicable shipping time, observation date, and source bytes.
For AEMA, either that direct terminal or both exact source-route leaves must close. Afterward the
receipts must be bound to each full registered assembly context, and all other roles must still close.

## All-family development impact

This is the original fixed 1,408-request development set, with 1,288 exact L1 results. It is not a
new sampling run or held-out quality estimate. All current all-role L2/L3/complete counts remain
`null` (unassessed), with **zero admitted current complete dossiers**. A zero in the new-L2 column
means this bounded five-component effort added no exact step to that family's requests.

| Family | Exact L1 / requests | Requests with new exact L2 step | Requests with selected vendor identity candidate |
| --- | ---: | ---: | ---: |
| a3_amine_aldehyde_alkyne | 64/64 | 0 | 0 |
| acid_epoxide_diester_multistep | 64/64 | 0 | 0 |
| aema_aza_thiol_addition | 47/64 | 47 | 47 |
| aldehyde_ugi3 | 64/64 | 0 | 16 |
| aldehyde_ugi4 | 64/64 | 0 | 7 |
| alpha_isocyanoester_dihydroimidazole | 64/64 | 0 | 64 |
| amine_alkylation | 57/64 | 0 | 5 |
| amine_epoxide_opening | 62/64 | 0 | 6 |
| aryl_reductive_amination | 40/64 | 0 | 9 |
| aza_michael_acrylamide | 63/64 | 0 | 8 |
| aza_michael_acrylate | 60/64 | 0 | 4 |
| disulfide_michael | 63/64 | 0 | 7 |
| epoxide_opening_o_acylation | 61/64 | 0 | 19 |
| iphos_ring_opening | 64/64 | 0 | 0 |
| ketone_isocyanide_amide | 64/64 | 0 | 64 |
| ketone_ugi4 | 58/64 | 0 | 3 |
| maleate_addition | 32/64 | 0 | 3 |
| o_esterification | 63/64 | 0 | 0 |
| passerini_3cr | 64/64 | 0 | 0 |
| preassembled_thiol_yne_tail_amidation | 44/64 | 0 | 0 |
| reductive_amination | 62/64 | 0 | 8 |
| thiolactone_aminolysis_michael | 64/64 | 0 | 9 |

## Reproduce and inspect

```sh
.venv/bin/python -m results.phase1.compose_lipid_route_evidence_v2.adjudicate
PATH="$PWD/.venv/bin:$PATH" make test-one UV_RUN= TEST=tests/test_compose_lipid_route_evidence_v2.py
PATH="$PWD/.venv/bin:$PATH" make test-one UV_RUN= TEST=tests/test_m0_05_source_evidence_adjudication.py
```

- [adjudication.json](adjudication.json): input pins, source review, seven mechanical checks, the
  repository adjudicator's L2 record-policy result, two alternative inverse candidates, typed partial route assessment,
  five vendor abstentions, context reuse rules and all 22 family rows.
- [impact.json](impact.json): compact all-family counts.
- [product_evidence_ledger.json](product_evidence_ledger.json): all requests and every recovered role.
- [sources/retrieval.json](sources/retrieval.json): timestamped direct HTTP failure receipts.
- [validation.json](validation.json): 8 focused and 10 existing adjudicator checks passed; Black/Ruff
  passed. Parent integration owns repository-wide vendor verification.

Missing evidence is not synthetic impossibility. No paper, frozen input, existing registry,
decoder, training checkpoint, or selection rule was edited.
