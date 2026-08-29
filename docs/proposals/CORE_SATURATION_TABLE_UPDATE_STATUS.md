# GEM table update under the core-saturation decoder: status

Tracking which tables can be regenerated now and which are blocked. Table numbers are the
current GEM build. "Blocked" means the underlying evaluation does not exist yet, not that the
work is hard.

## Ready now — evaluation complete, only artifact regeneration left

| # | Label | Source rows | What it needs |
|---|---|---|---|
| **T4** | `tab:production-comparison` | `production_comparison_transposed_rows` | Re-run the production adjudication over the 3 completed seeds. **Cells are currently inlined by hand** and must go back to a generated `\input` (see Caveats). |
| **T5** | `tab:production-seed-counts` | `production_seed_exact_counts_rows` | Same adjudication pass. |
| **T7** | `tab:architecture-ablations` | `architecture_ablation_rows` | Rebuild `seed_rows.jsonl` from the 3 mechanism runs, then `render-results-v1`. |

All three have every seed evaluated. No compute required.

## Blocked on the two training runs

| # | Label | Source rows | Blocker |
|---|---|---|---|
| **T2** | `tab:ugi-common-benchmark` | `common_ugi_benchmark_completed_rows` | FORGE row is seed 0 of 3. Internal arms (`shared_null_posthoc`, `fact_matched`, `fact_generous`) have ledgers in hand but need assessments run. |
| **T6** | `tab:lipid-realism` | `lipid_realism_rows` | Realism assessment over the new FORGE samples, all 3 seeds. |
| **T8** | `tab:catalogue-comparison` | `catalogue_comparison_transposed_rows` | FORGE rows for all three programs, 3 seeds. **Also currently inlined by hand.** |
| **T11** | `tab:baseline-seed-results` | `common_ugi_seed_rows` | Per-seed FORGE rows; needs all 3. |
| **T12** | `tab:baseline-decomposition` | `common_ugi_decomposition_rows` | Same assessment pass as T2. |

## Not affected by the decoder

| # | Label | Why |
|---|---|---|
| **T1** | `tab:gem-shared-programs` | Reports exact-L1 yield and transform verification per program from the production design. Regenerates with T4's adjudication; no separate work. |
| **T3** | `tab:route-dispositions` | Route cascade dispositions over a frozen, route-blinded shortlist. Reads `\ForgeRoute*` macros from the route artifacts, which the decoder does not touch. |
| **T9** | `tab:route-evidence-baselines` | Method-blind route evidence under the frozen union. FORGE's row changes only if its sample set changes; re-run alongside T2 for consistency. |
| **T10** | `tab:hela-property-guidance` | Potency guidance; frozen and closed. Unaffected. |
| **T13** | `tab:forge-generated-atlas` | Deterministic display-only structure atlas. |

## Caveats that will bite at assembly

1. **Two tables are hand-inlined.** T4 and T8 had their cells pasted in when they were transposed
   earlier today, so they no longer track their generated artifacts. Regenerating the artifacts
   will **not** update those tables. Either restore the `\input` with a transposed generated file,
   or re-copy 42 and 60 cells by hand. The former is correct.
2. **T2 will mix decoders across rows.** FORGE and its internal controls become core-saturation
   decoded; RGFN, DeFoG, GenMol/SAFE, the learned inventory selector and the finite catalogue
   oracle do not, because they are external or have no decoder. This needs one explicit sentence
   in the caption, not silence.
3. **T4 and T7 were never decoded the same way.** The production comparison ran unconstrained
   argmax; the mechanism study ran `strict_valence_topology_argmax`. Putting both under core
   saturation makes them comparable for the first time, which is worth stating.
4. **BL and LX stay unconstrained.** The constraint is Ugi-specific and left the other two
   programs byte-identical. Say so rather than implying one decoder throughout.
5. **n=3 spreads are wide.** Mechanism arms show seed standard deviations of 84 to 192 per 1,000.
   The gains are not separable from seed noise at three seeds; report the observed range and the
   preserved ordering, not a mean implying precision.

## Claims affected

- Shared-vs-Ugi-only collapses from 3.1x to 1.10x. The GEM prose states this through a macro and
  survives regeneration; the **ICLR** manuscript editorialises it at line 809 and needs softening.
- The cyclic control inverts, from costing 38.43 points to −0.3. Numerically self-updating, but a
  reader will ask why the control shows nothing; it deserves a clause.
- Conditioning versus post-hoc filtering **survives** at roughly 8x and is the contribution the
  paper actually states.
