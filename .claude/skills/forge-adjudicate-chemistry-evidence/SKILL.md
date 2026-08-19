---
name: forge-adjudicate-chemistry-evidence
description: Inspect primary articles, supplementary-information PDFs, patents, and supplier procedures for FORGE lipid chemistry. Use when extracting or adjudicating L1 final assemblies, L2 head or tail syntheses, transferable hydrophobic motifs, reaction schemes, conditions, outcomes, reactive sites, or source support for model supervision.
---

# FORGE Chemistry Evidence Adjudication

Treat the supplementary material as the primary chemistry evidence surface. Do not infer an
experimental route from a database structure, a deterministic transform, or a family precedent.

## Inspect the source

1. Resolve the primary article and every cited supplement or referenced preparation.
2. Hash each local asset. Record DOI, PMID or patent identifier, filename, page, scheme, compound
   label, section and procedure locator.
3. Render the cited chemistry pages and neighboring pages to images. Inspect schemes visually.
   Use text extraction only to locate content or transcribe prose.
4. If the source says "prepared similarly" or "using the method above," follow the reference chain
   until the complete procedure is recovered. Abstain if any required link is unavailable.

## Extract the reaction actually reported

Record:

- exact reactant and product structures, source labels and attachment atoms;
- reaction role and route level, keeping L1 final assembly separate from L2 component synthesis;
- reagents, catalyst, equivalents, solvent, temperature, time and atmosphere;
- workup, purification, yield and product-specific analytical evidence;
- successful, failed, mixed or unreported outcome;
- discrepancies between the scheme, prose, analytical label, database and normalized structure.

Do not inherit a source lipid's biological label when transferring only a tail motif. Do not turn a
source alcohol route into an executed aldehyde route unless the exact handle conversion is reported.

## Verify structures and reactive sites

Load chemistry definitions from the hash-pinned FORGE registry. Never retype reaction SMARTS.

1. Canonicalize source and target structures while retaining the source-reported form.
2. Enumerate all required-handle matches and all allowed reactive-site combinations.
3. Collapse symmetry-equivalent site assignments.
4. Apply the frozen forward transform and require exact target reconstruction, atom and mass
   conservation, correct component roles and no forbidden handle.
5. For multifunctional components, require a unique constitutional site class or an exact
   source-resolved attachment. Otherwise abstain.

## Grade evidence without inflation

Use one evidence basis:

- `exact_executed_characterized`
- `exact_executed_series_member`
- `exact_reported_library_execution`
- `exact_failed`
- `analogue_executed`
- `family_precedent`
- `computed_transform_consistency`
- `source_conflict`

An exact source route may supervise the exact L1 or L2 claim it reports. An analogue or family
precedent may support retrieval, template ranking or uncertainty, but never exact-route labels.
Computed forward compatibility may supervise transform consistency, but never an experimental
success outcome.

## Decide conservatively

Emit exactly one disposition:

- `admit_exact`: exact source execution, exact identity, complete locator and all mechanical checks;
- `admit_transform_consistency`: exact deterministic reconstruction with qualified components, but
  no claim of experimental execution for that exact product;
- `admit_negative`: exact reported experimental failure;
- `precedent_only`: executed analogue or reaction-family support;
- `abstain`: evidence, identity, procedure or reactive-site ambiguity remains;
- `reject_claim`: the proposed evidence claim conflicts with the source or deterministic chemistry.

Missing evidence is not a negative result. Rejection means the evidence claim is false, not that the
chemistry is impossible.

Only `admit_exact` and `admit_transform_consistency` may enter their explicitly labeled L1 objectives.
Only exact source-executed component routes may enter exact L2 supervision. Keep every other record
available for retrieval, ranking, uncertainty or later prospective resolution.

## Persist the audit

Write stable, machine-readable evidence records with source hashes and structured reason codes. Run
the repository adjudicator and its adversarial tests. Atom-order and canonical-SMILES permutations
must not change dispositions. Preserve the earlier blinded packet as an optional external audit, not
as a training blocker.
