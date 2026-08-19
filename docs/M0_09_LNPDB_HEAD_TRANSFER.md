# M0-09 LNPDB head-transfer census

## Purpose

This census broadens head-structure evidence beyond the 20 measured AGILE
amines without turning the whole-lipid generator into a component enumerator.

FORGE generates the complete lipid, including the head, at atom-and-bond
resolution. The LNPDB head ledger is used after or during generation to
measure structural precedent, qualify a Ugi-reactive amine site, prioritize
procurement and route work, and define evaluation strata. It is not a decoder
vocabulary and does not bound generated heads to observed identities.

A generated head may be absent from AGILE and LNPDB. It can still enter the
supported Ugi space when:

- the complete product is within the declared molecular support;
- a specific amine site passes the qualified Ugi handle policy;
- the exact components reconstruct the product through the frozen transform;
- the head terminates through current procurement or a complete upstream
  route;
- the final assembly and all other component routes pass the evidence policy.

## Frozen census

The LNPDB component source ledger contains 408 head entries. Of these, 396
parse as connected molecular graphs and 12 remain invalid source structures.
The parsed structures were compared with the 20 unique reconciled AGILE heads,
the qualified AGILE Ugi amine SMARTS, symmetry-distinct reactive-site
multiplicity, and the primary ring-topology gate.

| Disposition | Heads |
| --- | ---: |
| Exact AGILE head identity | 20 |
| Bounded LNPDB Ugi head-transfer candidate | 259 |
| Outside primary ring topology | 62 |
| No qualified Ugi N-H amine | 55 |
| Invalid source structure | 12 |

The 259 transfer candidates contain:

| Ring topology | Heads |
| --- | ---: |
| Acyclic | 130 |
| One 5-membered ring | 22 |
| One 6-membered ring | 107 |

Thirty-six candidates contain two symmetry-distinct qualified amine sites and
therefore require explicit reactive-site resolution. None of the 259 currently
has assessed procurement or upstream route evidence in the LNPDB component
ledger.

## Admission boundary

No head is admitted to primary candidate lock by this census. LNPDB occurrence
is structural and biological provenance for the source lipid, not evidence
that the isolated head:

- is commercially available;
- has a complete upstream route;
- is compatible with AGILE Ugi conditions;
- transfers the source lipid's activity to a new product;
- falls within an authorized biological-oracle domain.

The queue ranks candidates by source-lipid prevalence and source breadth only
to make subsequent review reproducible. Generator demand and scientific value
must determine which candidates receive actual route or procurement work.
Broad mining of all 259 heads is not authorized by default.

## Ring boundary

The queue uses a topology policy, not an exact ring-fragment vocabulary. A
novel generated 5- or 6-membered head heterocycle may qualify even when its
identity is absent from the queue. Macrocycles, fused systems, spiro systems,
bridged systems, and rings outside the head remain outside automatic primary
Ugi support and require a separate evidence-based support amendment.

## Reproduction

```bash
make m0-09-lnpdb-head-transfer
```

Outputs:

- `results/m0_09/lnpdb_head_transfer.json`
- `results/m0_09/lnpdb_head_transfer_ledger.csv.gz`

The component ledger, qualified reaction registry, reconciled AGILE dataset,
ring-support artifact, and configuration are hash-pinned.
