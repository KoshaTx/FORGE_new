# Legacy code retirement

Both the three-family and 22-family studies remain supported. Age, filename, and absence from
one paper contract do not establish that code is obsolete.

## Survey commands

```sh
make code-survey-supported
# Equivalent, with a chosen report location:
PYTHONPATH=.:tools:paper python -m forge_maintenance survey \
  --scope supported-studies --output build/supported_studies_code_survey.json
```

The combined survey roots every maintained application and repository tool, including direct
workflows absent from the catalog. Executable scripts and source snapshots under `results/` are
historical roots too. It reports tests/reference code and historical consumers
separately. It follows imports, relative imports, re-exports, literal dynamic loads, documented
paths, and historical path moves. Source/document hashes and the exact pin-declaration digest
identify the inspected state. Compare fingerprints before reusing a report. Exit 2 reports
unresolved references requiring review; it is not authorization to delete an unreached module.

The existing single-contract `survey --contract ...` and `make code-survey` remain available.
Their historical receipt is not a combined inventory. The combined target writes under `build/`
and does not replace that receipt.

## Decisions

- `keep_active`: reachable from maintained workflows, CLI, tools, or current documentation.
- `keep_reference`: used by tests or behavioral reference implementations.
- `keep_historical`: used by historical code, manuscripts, or source pins.
- `review_unreached`: no discovered root; manual symbol and side-effect review is still required.
- `blocked`: unresolved source identity or unbounded dependency ambiguity prevents retirement.

Roles are recorded independently, even when a blocking condition takes precedence. The scanner
is conservative: all direct applications and historical producers are roots. It does not execute
modules or decide which scientific workflows should cease to be supported. The existing pin
collector tolerates unreadable/missing artifacts, so its output cannot establish absence of use.
Unknown external callers, generated import names, and reflection require manual adjudication.

For each removal, retain exact pre-edit source bytes, list incoming references and replacements,
and record affected tests. Compare the set of resolvable `(original_path, sha256)` identities
before and after. A new loss blocks the change; counts alone cannot prove preservation.
Never rewrite a result or weaken an evidence gate to accommodate cleanup.

## This cleanup and migration

The implementation and verification records are in
[`../results/maintenance/legacy_cleanup_v1/`](../results/maintenance/legacy_cleanup_v1/).
`retirements.json` lists the exact private facade names removed and their owning-module imports.
These helpers were not called through their facades by repository consumers. Their implementations
remain available in the listed owners. Public sampler functions, catalog IDs, stage IDs, and
registered stage functions are unchanged.

Five duplicated hashing/atomic-write bodies now delegate to `forge.core`. The corpus hash adapters
retain their domain-specific missing-input errors and `chunk_size` argument. The provenance hash
adapter retains its historical `chunk` keyword. Evidence writers retain their `content` keyword,
bytes, and 0644 publication permissions.

## Deferred work

- The legacy Ugi joint sampler remains an independent equivalence oracle and historical audit input.
- The provenance resolver compatibility module remains used by historical consumers.
- The sampler diameter-helper export remains used by an executable historical audit in `results/`.
- Text/JSON writers with distinct serialization or durability behavior require a separate contract
  comparison before consolidation.
- Whole-module retirement and historical relocations require bounded dynamic-reference review and
  runnable reproduction, not merely archived source bytes. No module is deleted by this survey.
- Existing unresolved historical identities remain visible. They block their affected retirements;
  they do not authorize relaxing source verification or block unrelated safe helper consolidation.
