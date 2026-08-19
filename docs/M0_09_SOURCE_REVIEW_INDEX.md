# M0-09 Source Review Index

## Outcome

The M0-09 source acquisition pipeline now provides machine-readable main text
for all 26 PMC-linked LNPDB publication families. Complete local source packages
are available for 19 families. These packages contain 67 supplementary review
assets that were discovered from official JATS records or publisher pages,
downloaded, format-validated, and hash-verified.

Seven publication families still lack an acquired supplement. Their main text is
available, but they remain explicitly marked `supplement_unavailable`. No paper
is marked route-reviewed merely because its files were downloaded.

Chemistry review is a separate, validated stage. Two source packages have now
been reviewed: the first building-block lipid source and AGILE. See
`docs/M0_09_LNPDB_SUBCOMPONENT_REVIEWS.md` for the first source-level result
and `docs/M0_09_UGI3_PRECURSOR_CAPABILITY.md` for the Ugi-3 capability result.
Both keep upstream component synthesis separate from linker-tail and final
lipid assembly.

## Component-level source availability

The review index links source-package status back to all 726 normalized LNPDB
components.

| Role | Any complete source package | Only incomplete PMC package | Non-PMC or unlinked source |
|---|---:|---:|---:|
| Head | 330 | 21 | 57 |
| Linker | 6 | 2 | 27 |
| Tail 1 | 79 | 80 | 79 |
| Tail 2 | 8 | 2 | 35 |
| **Total** | **423** | **105** | **198** |

These counts measure source availability, not route recovery. A complete source
package may contain a general procedure for a family rather than an exact route
for every component. Conversely, one recovered general procedure may support
many structurally distinct components after its scope is validated.

## Modeling implication

LNPDB supplies the lipid-specific chemistry specialization layer. Its structures
identify relevant heads, linkers, tails, disconnections, and library synthesis
patterns. Its papers supply recoverable reaction procedures, conditions,
examples, and failures.

Structural diversity is larger than explicit route-label diversity. The 12,837
LNPDB lipid structures arise from a much smaller number of publication families
and recurring assembly reactions. The route system therefore combines:

1. General reaction competence from a cleaned, split-safe reaction corpus.
2. Lipid-specific route specialization and retrieval from LNPDB sources.
3. Exact forward checks and explicit uncertainty at every proposed step.
4. Current procurement evidence for commercial precursors.
5. Prospective experimental outcomes that can update route confidence.

This design does not reduce generation to building-block enumeration. The
discrete flow model proposes the complete lipid graph. The route layer
constructs and evaluates a complete synthesis program with L1 assembly, L2
precursor synthesis, and L3 procurement closure.

For the current paper, source review is prioritized by contribution to the
AGILE-type Ugi 3-CR precursor envelope. The goal is an auditable basis of
reaction families with known scope and gaps, not a claim that every LNPDB
publication contains a distinct useful transformation. Observed source examples
and the later bounded virtual-space census remain separate artifacts.

## Evidence boundary

The current index proves that source files are local and hash-verified. It does
not yet prove that:

- every attachment contains lipid synthesis details;
- every LNPDB component has an exact source-located route;
- a general procedure covers every structure assigned to its family;
- a proposed route is experimentally executable;
- synthesis-aware guidance improves generation.

Those claims require chemistry review, structure-linked extraction, route-family
scope checks, held-out route evaluation, and prospective synthesis.

## Reproduction

```bash
make m0-09-pmc-sources
make m0-09-publisher-sources
make m0-09-source-review-index
make m0-09-paper-route-reviews
```

The tracked artifacts contain source URLs, hashes, byte counts, status labels,
and software provenance. Downloaded source files remain in the ignored
`data/source_cache/` tree.
