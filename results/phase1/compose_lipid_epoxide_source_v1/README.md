# Original epoxide inputs authenticated; chemistry still pending

Run from the repository root:

```sh
.venv/bin/python results/phase1/compose_lipid_epoxide_source_v1/audit_original_inputs.py
.venv/bin/python results/phase1/compose_lipid_epoxide_source_v1/audit_component_contract.py
```

The extraction requires a fresh output directory and independently authenticates the full
preparation population before selecting eligible original task lines. Its receipt is
`../compose_lipid_epoxide_original_tasks_v1/result.json`.

All 3,765 eligible rows have original task bytes. Their complete precursor structures agree with
the global component catalogue, and their declared epoxide occupancy agrees with both exported
quantities and the structural N-H inventory. Occupancy is 1, 2, 3, 4, 5 or 6; the formal family
default of six is not substituted for the original per-record quantity. `component-contract-audit.json`
preserves the checks for every selected record and input SHA-256 hashes.

Every selected task declares Anderson source subseries PMID 20080679, DOI 10.1073/pnas.0910603106.
The available Han source registry supplies structural queries for this input audit only. Its
separate source program does not establish the Anderson products, conditions or regioselectivity.
The primary Anderson supplement or equivalent exact primary controls are still needed. Earlier
access failures and the precedent-only disposition remain in `../compose_lipid_v8_epoxide_source_v1/`.

No epoxide reaction program, experimental outcome or training row is admitted by these checks.
