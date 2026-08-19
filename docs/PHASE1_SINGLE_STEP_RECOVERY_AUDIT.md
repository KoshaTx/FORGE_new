# Phase 1 single-step retrosynthesis recovery audit

## Purpose

This audit asks a narrow question: can an independently frozen proposal engine recover known
single-step precursor sets without seeing the route truth during proposal generation? It does not ask
whether a proposal is experimentally validated, operationally suitable, commercially closed, or ready
to enter FORGE's synthesis value.

The benchmark contains 120 public targets. Thirty-six targets have exact hidden precursor truth used
only after the proposal ledgers are frozen; 12 adversarial controls remain reserved for the full
operational qualification benchmark. Exact precursor-set recovery is scored at top 1, 5, 10 and 20.

## Frozen results

| Proposal engine | Top 1 | Top 5 | Held-family top 5 | Known-exact top 5 |
|---|---:|---:|---:|---:|
| AiZynthFinder 4.4.1 | 18/36 (50.0%) | 19/36 (52.8%) | 9/18 (50.0%) | 10/18 (55.6%) |
| Graph2Edits | 22/36 (61.1%) | 29/36 (80.6%) | 13/18 (72.2%) | 16/18 (88.9%) |
| Union | 22/36 (61.1%) | 29/36 (80.6%) | -- | -- |

Graph2Edits recovers every oxidation, esterification and formamide-dehydration truth in the benchmark.
AiZynthFinder recovers every oxidation and formamide-dehydration truth but no esterification truth.
Neither learned engine recovers the seven amine-formylation truths. Those seven transformations are
already represented in the frozen exact-source lipid route registry, so the miss identifies the value
of the hybrid architecture rather than a reason to discard the registry.

## Decision

Use Graph2Edits as the primary source-neutral proposal engine for the next adjudication experiment.
Retain AiZynthFinder as an independent diagnostic challenger. Preserve the exact-source lipid registry,
including amine formylation, as the evidence-backed route layer.

The learned engines generate route hypotheses only. Their scores and precursor sets do not by
themselves establish route evidence, synthesis probability, terminal-material closure, or permission
to influence generation. Before a proposed route can change the production evaluator it must pass:

1. exact forward reconstruction;
2. substrate-scope and reaction-family adjudication;
3. operational compatibility checks;
4. independent evidence qualification; and
5. L3 terminal-material or procurement closure.

Only if those checks increase nonuniform, independently supported route value on generated candidates
will FORGE repeat the single preregistered matched synthesis-guidance comparison. Promotion still
requires higher unique, diverse, route-closed yield per matched compute than post-generation routing.

## Frozen artifacts

- `results/phase1/aizynthfinder_single_step_recovery_benchmark_v1/score.json`
- `results/phase1/graph2edits_single_step_recovery_benchmark_v1/score.json`
- `results/phase1/hybrid_single_step_recovery_audit_v1/result.json`

The hybrid result is content-identified by
`1eee60b504f4bff12c1d7bf661ff7c891078211bb015107e7427005174b19520`.
