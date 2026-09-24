# Aldehyde Ugi four-component source qualification

The primary source is Li, Raji, Gordon et al., Nature Materials (2024),
DOI [10.1038/s41563-024-01867-3](https://doi.org/10.1038/s41563-024-01867-3).
`acquisition.json` records the public publisher URLs, retrieval time and exact asset hashes.
The independently transcribed control is lipid 119-23: the SI's named product, procedure,
formula and drawn structure agree with the four-component schematic in main Figure 1.
`control_transcriptions.json` records the locators, conditions and interpretation limits.

The SI was visually inspected at PDF pages 8–10 and 16–17, including the product drawing;
adjacent PDF pages 7 and 15 were also inspected. No L2 route or biological measurement is
admitted here. The source's earlier L2 compound-label inconsistencies are preserved.

`build_registry.py` derives a new overlay from pinned Ugi-3, Passerini and A3 registry queries.
The source drawing supplies the new acid-carbonyl-to-amine bond. Original registries remain
unchanged. The overlay records net water loss and a mechanistic atom-origin convention for
the acid oxygen; it does not claim isotope tracing or a sequence of isolated intermediates.
The production checker preserves unfiltered forward outcomes, complete inverse uniqueness,
reactive-site witnesses and complete element/hydrogen/charge balance.

The final production receipt is
`results/phase1/compose_lipid_v8_ugi4_program_v1/result.json`, SHA-256
`ae5fbd1fe263e3dd0e6904e1ad4c6c605614da05541407b3d133f5b154e1c5e3`.
It records **8,163 exact programs among 8,450 preparation records**. The earlier protection
layer already excluded 24 products. Known protected precursors occur in 4,314 exact records;
3,849 clear the currently known exclusions. None is admitted to training.

`exclusion_diagnostic.json` and `excluded_programs.jsonl.gz` retain all 287 rejected inverses.
Each inferred amine has three matching handles, outside the inherited registry multiplicity
of one or two. This is unsupported multiplicity under the qualified contract, not a claim
that the products are chemically invalid or that forward attachment is necessarily ambiguous.
The diagnostic never relaxes that bound or changes a program exclusion.

`control_check_development.json` is an earlier development check. Its registry/builder pins
precede final formatting and the monocarboxylic-acid restriction. It is retained as history;
the final source controls and implementation pins are those in the production receipt.

The separate `prior_dictionary_concordance.json` compares the older v5 dictionary against
the previously qualified seven-family precursor ledger. Only the 43 Michael amine labels
per Michael family are recognized and concordant; other tested roles remain absent. This
does not resolve evaluation identities, admit aliases or establish precursor disjointness.

Reproduction from the repository root, with the installed environment:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.phase1.multireaction.compose_lipid_condensation_event \
  build --config configs/multireaction/compose_lipid_v8_ugi4_program_v1.json \
  --output results/phase1/compose_lipid_v8_ugi4_replay

PYTHONPATH=. .venv/bin/python -m experiments.phase1.multireaction.compose_lipid_condensation_event \
  verify --result results/phase1/compose_lipid_v8_ugi4_program_v1/result.json

PYTHONPATH=. .venv/bin/python results/phase1/compose_lipid_v8_ugi4_source_v1/inspect_exclusions.py
PYTHONPATH=. .venv/bin/python results/phase1/compose_lipid_v8_ugi4_source_v1/check_prior_dictionary.py
```

Independent verification repeats the exact control and every preparation decision. These
receipts establish computed transform consistency within the declared scope. Corpus-wide
chemical precision, original experimental bank membership, synthesis success, complete L2/L3
dossiers and global holdout qualification remain unestablished. Ketone Ugi chemistry is not
qualified by this aldehyde control. No training or paid remote job was launched.
