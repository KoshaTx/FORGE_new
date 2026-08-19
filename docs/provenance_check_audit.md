# Provenance-check audit: the 50 `_pin` implementations

The `_pin` helper validates one declared `{"path", "sha256"}` input before a stage reads it. It is
written out longhand in 50 modules, and **the copies do not check the same things**. Which
guarantee a pinned input actually got depended on which module happened to read it.

This matters more than ordinary duplication. These checks are what make a result attributable to
the bytes it came from — the whole basis of the paper's evidence chain. A weaker copy does not
fail loudly; it silently accepts an input the strict version would reject.

## What the copies check

Six distinct check-sets across 50 definitions:

| Modules | `exact_key_set` | `rehashes_file` | `containment` | `symlink` | `is_file` |
|---:|:--:|:--:|:--:|:--:|:--:|
| 34 | yes | yes | yes | yes | yes |
| 5 | yes | yes | yes | — | — |
| 4 | yes | yes | — | yes | yes |
| 3 | yes | yes | yes | — | yes |
| 2 | — | yes | yes | yes | yes |
| 2 | — | yes | yes | — | — |

Every copy re-hashes the file, so no module is trusting a digest it never computed. That is the
one thing they all get right. **None validates that the declared digest is well formed** — a
64-character lowercase hex string — which is why `core.resolve_pin` adds it.

## The gaps, named

**No containment check — 4 modules.** These resolve the declared path without confirming it lands
inside the repository, so a pin naming `../../elsewhere/file` would be read and accepted as long as
its hash matched. Combined with a missing symlink check this is the weaker half of the pair that
`core.resolve_pin` documents.

- `bio/ugi_morphology_high_potency_challenger.py`
- `bio/ugi_morphology_high_potency_proposal.py`
- `product/ugi_graded_route_readiness_guidance.py`
- `value/ugi_synthesis_guidance_failure_audit.py`

**No symlink check — 10 modules.** A symlink can satisfy its own recorded hash while the bytes
actually live somewhere else entirely, so the pin verifies and the provenance record is wrong.

- `product/ugi_current_source_zero_guidance_requalification.py`
- `product/ugi_grouped_smc_schedule_qualification.py`
- `product/ugi_grouped_zero_guidance_identity.py`
- `product/ugi_production_zero_guidance_seam_v2.py`
- `product/ugi_selected_v2_full_equivalence.py`
- `product/ugi_selected_v2_pool_singleton_equivalence.py`
- `product/ugi_source_guidance_prerequisite_requalification.py`
- `route/ugi3_source_qualified_cumulative_inputs.py`
- `value/ugi3_fresh_pool_route_coverage_v6.py`
- `value/ugi_route_completion_utility_qualification.py`

**No key-set check — 4 modules.** They accept a record carrying extra keys beside `path` and
`sha256`, so a malformed or over-specified pin passes rather than being rejected.

- `product/ugi_bounded_hybrid_route_cascade.py`
- `product/ugi_grouped_smc_schedule_qualification.py`
- `product/ugi_grouped_zero_guidance_identity.py`
- `product/ugi_route_aware_panel_feasibility.py`

**No existence check — 7 modules**, overlapping the symlink list.

## How to read this

None of these is a live exploit. There is no adversary here, the repository is not multi-tenant, and
nothing suggests a pin has actually been subverted. What it does mean is that **16 of 50 provenance
checks are weaker than the strictest one in the same codebase**, and a reviewer asking "was this
input verified?" would get a different answer per module.

## Consequence for the migration

Migrating these onto `core.resolve_pin` is not a like-for-like swap. The strict version may reject
an input a weak copy currently accepts, and **that failure would be correct** — it would mean the
module had been validating loosely all along.

So this group is deliberately excluded from the mechanical batches. It wants its own pass:
migrate, run, and treat every new failure as a finding to investigate rather than a regression to
suppress. Doing it inside a bulk refactor would bury exactly the signal worth having.
