# STAAR source packet and qualified sequential program

Primary source: Peña et al., [Communications Chemistry 2025](https://doi.org/10.1038/s42004-025-01516-z),
PMID 40234552. `acquisition.json` and `figures-acquisition.json` record actual downloaded bytes.
The publisher supplement is `si.pdf`; the article and Figures 1–3 are retained separately.

Figures 1–3 and SI pages 3–5 were visually inspected. `control_transcriptions.json` records two
independent final-product drawings, their complete reactants, source locators, conditions and
conflicting source labels. The source combines Bj and Ck before adding Ai. Mechanistically,
thiolactone aminolysis must expose the thiol before its addition to the acrylate; reagent addition
order is a different statement.

```bash
uv run python results/phase1/compose_lipid_v8_staar_source_v1/check_transcriptions.py
```

This checks additive element/H/charge inventories and theoretical protonated masses for A4B2C3
and A4B2C8 (CP-LC-0729). It produces neutral formulas C39H77N3O4S and C43H85N3O4S. Formula and
mass agreement alone do not establish connectivity or reaction selectivity. The original
transcription receipt remains a record of source preparation, before executable qualification.

`stage_contract.json`, `build_registry.py` and `adjudication.json` now qualify a separate registry
overlay for thiolactone aminolysis followed by thiol-acrylate addition. Both independent controls
pass complete forward/inverse replay, unique intermediate/product checks and full inventory
balance. The source-derived scope retains neutral amine heads with a surviving basic-site
candidate, N-acyl homocysteine thiolactones and acrylate esters. It does not establish pKa,
experimental selectivity or membership in the original reagent banks.

The complete v8 enforced preparation population contains 9,314 products after 52 prior validation
products were excluded. All 9,314 have unique complete programs; none is lost to component
constraints or label conflicts. Known protected precursors exclude 2,792; the other 6,522 still
await global precursor protection, holdout qualification and final TRAIN admission.
`../compose_lipid_v8_staar_program_v1/result.json` and its complete ledger are independently
replayed by:

```bash
uv run python -m experiments.phase1.multireaction.compose_lipid_sequential \
  --verify results/phase1/compose_lipid_v8_staar_program_v1/result.json
```

The program result SHA-256 is
`21eec94e2f56de024c22501705f33332a5fbc081729a611ae79ce0b89bc4a17a`.
No training rows are admitted.

Keep the B2/B3 prose-versus-drawing discrepancy visible. Do not promote the permanently cationic
comparison or the separately modified ester/linker derivatives into the ordinary ionizable
STAAR program. Published biological outcomes were not imported as training labels.
