# M0-08 Biological Endpoint Decision Package

Status: **endpoint not locked; intramuscular reporter delivery followed by
functional editing is the current lowest-risk option**

## Decision question

Should the first prospective FORGE campaign culminate in intramuscular delivery
and editing, intravenous liver delivery and editing, or intramuscular mRNA
vaccination?

The endpoint changes the biological bridge, candidate-advancement rules,
formulation assessment, animal design, and interpretation of the AGILE oracle.
It does not change the synthesis-grounded whole-molecule architecture.

## Evidence ledger

| Claim | Evidence class | Source |
|---|---|---|
| AGILE connects HeLa screening to intramuscular mouse expression and reports a HeLa-to-IM correlation of approximately 0.78 for its selected panel | Source-reported | Xu et al., Nature Communications 2024, DOI `10.1038/s41467-024-50619-z` |
| JC_2023 uses a native Ugi-3 library and advances a muscle-selective lipid through IM expression, muscle editing, and vaccine studies | Source-reported | Chen et al., PNAS 2023, DOI `10.1073/pnas.2309472120` |
| Miao 2019 links a 1,080-lipid isocyanide-mediated 3-CR library to APC activation, antigen-specific immunity, and antitumor efficacy | Source-reported | Miao et al., Nature Biotechnology 2019, DOI `10.1038/s41587-019-0247-3` |
| LiON identified pulmonary candidates using an AI-guided screen and validated reporter mRNA delivery in mouse and ferret lungs without requiring genome editing or disease correction | Source-reported | Witten et al., Nature Biotechnology 2025, DOI `10.1038/s41587-024-02490-y` |
| A Bowen Li combinatorial LNP study progressed from reporter delivery to Cre and Cas9 reporter editing in mouse lung without a disease-correction experiment | Source-reported | Li et al., Nature Biotechnology 2023, DOI `10.1038/s41587-023-01679-x` |
| AGILE evaluated HeLa-to-IM translation for 15 selected lipids and reported a PCC of 0.78; it did not report a HeLa-to-IV-liver correlation | Source-reported | Xu et al., Nature Communications 2024, DOI `10.1038/s41467-024-50619-z` |
| A historical lipidoid study found HeLa informative for hepatocellular siRNA delivery within matched library and formulation conditions, but primary hepatocytes were more predictive and formulation affected the relationship | Source-reported | Whitehead et al., Nature Communications 2012, DOI `10.1038/ncomms1899` |
| Similar in vitro activity and liver accumulation can still yield different functional hepatocyte delivery because lipid chemistry affects in vivo identity and cell-level routing | Source-reported | Johnson et al., Molecular Pharmaceutics 2022, DOI `10.1021/acs.molpharmaceut.2c00442` |
| The current AGILE single-structure oracle contains HeLa and RAW 264.7 labels, not liver-editing or vaccine-immunogenicity labels | Computed and source-verified | `results/m0_07/agile_label_reconciliation.json` |
| The vendored LiON-derived R0 records are structure-only pretraining data and do not supply an admitted liver endpoint oracle | Computed | `data/vendor/r0_observed_real_structures.csv`; `data/vendor/training_corpus_manifest_v1.json` |
| The laboratory has a mature vaccine capability | User-reported | Project direction provided during M0-08 |
| A liver editing target, vaccine antigen, animal capacity, benchmark, and formulation protocol are committed | Not yet established | Requires explicit PI or laboratory confirmation |

## Reassessed venue burden

FORGE is an AI and synthesis-platform paper. The in vivo package must establish
that prospective, synthesis-grounded designs form LNPs and deliver a functional
RNA cargo. It does not need to reproduce the evidence burden of a standalone
disease-correction or vaccine-mechanism paper unless the manuscript makes those
claims.

The relevant precedent supports a staged package:

1. show reporter delivery and organ distribution for the in vivo panel;
2. advance one or more leads to a functional editing or antigen-specific
   endpoint;
3. include a declared benchmark and basic tolerability;
4. treat disease correction, challenge, extensive mechanism, repeat dosing,
   and large-animal validation as strengthening experiments rather than
   universal requirements.

## Comparison

| Criterion | IM delivery and editing | Liver delivery and editing | IM vaccination |
|---|---|---|---|
| Alignment to AGILE HeLa supervision | Strongest available, but based on 15 selected lipids | Indirect; no AGILE HeLa-to-IV-liver validation | Better for IM expression, but not immunity |
| Endpoint-specific bridge | Muscle-relevant expression | Primary hepatocyte or equivalent liver bridge | Muscle or APC expression |
| Minimum functional endpoint | Cre reporter conversion or reporter-locus/endogenous editing | Sequence-verified reporter-locus or endogenous editing | A predeclared antigen-specific immune response |
| Minimum in vivo sequence | IM reporter delivery, then editing in lead candidates | IV reporter delivery and biodistribution, then editing in leads | IM reporter delivery, then antigen-specific immunity |
| Additional work not intrinsically required | Disease correction and exhaustive mechanism | Disease correction, NHP, repeat dosing, exhaustive toxicology | Challenge, both adaptive arms, long durability |
| Main scientific risk | Selected-panel correlation may not transfer to novel components | General HeLa oracle may not transfer to functional hepatocyte delivery | Reporter delivery may not predict adaptive immunity |

## Current leading option, not a lock

The current lowest-risk option is **intramuscular reporter delivery followed by
functional editing**. This uses AGILE's demonstrated translation direction,
retains the compact reporter-to-function sequence, and avoids the
adaptive-immunity burden of a vaccine claim. It also remains separate from
COMPOSE-Lipid's pulmonary focus.

The AGILE correlation is not a universal guarantee. It was measured on 15
selected lipids, not on generated components under prospective holdout.
Therefore, FORGE still requires a muscle-relevant bridge, applicability-aware
selection, and prospective evaluation across distinct chemotypes.

Liver remains scientifically attractive and operationally familiar, but the
current AGILE oracle does not justify liver-specific biological tilting. A
liver endpoint should be chosen only after one of two gates is met:

1. a liver-labelled external corpus, such as the LiON source collection, is
   audited for assay, formulation, source, and chemistry compatibility and
   improves held-family liver prediction under non-leaking splits; or
2. a prospective primary-hepatocyte or equivalent bridge is used to advance
   synthesized candidates without calling AGILE a liver oracle.

IM vaccination remains a credible alternative because the laboratory reports a
mature vaccine capability. If selected, the minimum claim-bearing package needs
IM delivery plus at least one predeclared antigen-specific immune endpoint. A
challenge study is not automatically required for a synthesis-platform paper.

## Minimum IM editing package

Before in vivo advancement:

1. use the AGILE oracle as general-transfection guidance within its declared
   applicability domain;
2. measure reporter expression in a muscle-relevant system under the intended
   formulation;
3. prospectively measure particle size, polydispersity, encapsulation, and
   formulation robustness;
4. retain candidates across structural and component novelty bins.

In vivo:

- test IM reporter delivery and major-organ distribution for the selected
  FORGE panel against a declared benchmark;
- advance at least one lead to Cre reporter conversion or sequence-verified
  reporter-locus or endogenous editing in muscle;
- record basic local and systemic tolerability;
- show that at least two structurally distinct FORGE lipids deliver reporter
  mRNA, with editing demonstrated for at least one lead.

## Minimum liver package if selected

Before in vivo advancement:

1. use AGILE HeLa and RAW 264.7 models only as general-transfection guidance;
2. measure expression or editing in a hepatocyte-relevant system under the
   intended formulation;
3. prospectively measure particle size, polydispersity, encapsulation, and
   formulation robustness;
4. retain candidates across declared oracle-applicability and structural
   novelty bins.

In vivo:

- test reporter delivery and major-organ biodistribution for the selected
  FORGE panel against a declared benchmark;
- advance at least one lead to sequence-verified editing at a reporter locus or
  an endogenous target;
- record basic tolerability appropriate to the selected dose and schedule;
- show that at least two structurally distinct FORGE lipids deliver the
  reporter, with editing demonstrated for at least one lead.

A dose-response series, endogenous or disease-relevant editing, repeat dosing,
expanded histology, and disease correction strengthen the package but are not
all mandatory for the core synthesis-grounded platform claim.

## Minimum vaccine package if selected

- demonstrate IM reporter delivery and distribution under the intended
  formulation;
- measure at least one predeclared antigen-specific immune endpoint appropriate
  to the vaccine goal, either humoral or cellular;
- include a declared benchmark and basic tolerability;
- treat the second adaptive-immunity arm, extended durability, and challenge or
  therapeutic efficacy as strengthening experiments.

IM reporter expression alone supports an IM delivery claim, not a vaccination
claim.

## Candidate-lock rules

The primary synthesis panel is locked before any prospective synthesis outcome
is known. The in vivo advancement rule is frozen before formulation and bridge
results are examined. Advancement balances:

- calibrated general-transfection guidance;
- synthesis-program closure and uncertainty;
- structural and component novelty;
- prospective formulation success;
- endpoint-specific bridge performance;
- diversity across chemotypes.

No unavailable apparent pKa, particle, or formulation value is imputed.

## What remains to lock

The final endpoint becomes locked only after the laboratory confirms:

1. the muscle or liver editing target, or vaccine antigen;
2. the endpoint-specific bridge assay;
3. the in vivo functional assay;
4. candidate and animal capacity;
5. benchmark LNP and formulation protocol;
6. minimum tolerability measurements.

This decision is intentionally deferred while M0 computation continues. Until
the human lock, software requires an explicit endpoint identifier and carries
no default.

## Software boundary

`forge.bio.endpoint` defines the generic endpoint contract.
`forge.bio.muscle`, `forge.bio.liver`, and `forge.bio.vaccine` define evidence
stubs. No model, generator, route planner, or candidate selector imports a
default endpoint.
