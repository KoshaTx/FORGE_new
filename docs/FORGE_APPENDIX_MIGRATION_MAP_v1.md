# FORGE appendix migration map v1

Editorial control document. Not part of the manuscript.

Source material:

- `manuscript/FORGE_ICLR2027_submission.md` — Appendix A--J, lines 708--966; Tables S1--S16, lines 967--1491
- `manuscript/FORGE_Nature_Biotechnology_working_draft.md` — Methods lines 400--878; the same
  Tables S1--S16, lines 1125--1652; chemistry dossier, lines 1653--1682
- `scripts/phase1_build_manuscript_supplement_v1.py` — emits S1--S16 between `SUPPLEMENT:BEGIN`
  and `SUPPLEMENT:END`, marked do not edit by hand

Target: Appendix A--H in `manuscript/FORGE_ICLR2027_paper.tex`.

Decisions: **KEEP** as is; **UPDATE** content changed, structure survives; **REGEN** rebuild from
artifacts via the supplement script; **HISTORICAL** survives only as labelled provenance;
**DROP** does not survive.

---

## Prose appendix, old ICLR draft

| Old material | Decision | New location | Reason |
|---|---|---|---|
| A Notation, 25 symbols | UPDATE | A.1 | Drop `s(x)`, `q_{1-\alpha}(s)`, `LCB_{1-\alpha}`, which are retired conformal symbols. Add `E(G)`, `S_act`, `\Delta_sem`, `\pi_\gamma`, `\rho(c)`. Notation moves inside A rather than holding its own letter. |
| B.1 Prop 3.1, exact decomposition | UPDATE | G.1 | Still true and still load-bearing for Eq. (5)--(6). Restate against the current deterministic role map; the old proof leans on `o` as part of the sampled state, which is the withdrawn origin-variable formulation. |
| B.2 Prop 3.2, sampler support invariant | KEEP | G.4 | Valid unchanged. This is the appendix G.3 stub written out already; use the old proof rather than rewriting it. |
| B.3 Prop 4.1, split conformal | DROP | --- | Supports only the retired stratified-conformal ranking. |
| B.4 Prop 4.2, support preservation | UPDATE | G.3 | Same result as the new Proposition 1. **Ship one proposition, not two.** Keep the new statement, which is written against `\pi_\gamma` and `\epsilon`, and drop the duplicate. |
| C Model, training, declared support | KEEP | A.5, A.6 | Contains the production provenance that matters: refit once from random init, 5,100 fixed steps, no early stopping, no post-refit checkpoint selection, so no held-out quantity chose the shipped weights; architecture selected earlier on a separate 1,024-program draw under prespecified hard gates; best development calibration-loss checkpoint was step 500 and was not shipped. |
| D Data, splits, reaction adapter | KEEP | A.1, A.2 | The split-before-component-extraction rule is a real leakage control and is not stated anywhere in the current draft. |
| E Algorithms | RELOCATE | F.4, A.8 | Routing and the two tail disconnections belong in F; training and sampling pseudocode in A.8. |
| F Extended results | UPDATE | D.3 | Figure placeholders; rebuild around the current 38-descriptor contract. |
| G Blinded chemistry dossier | KEEP | F.8 | The strongest anti-post-hoc evidence in the paper and absent from the current skeleton. Shuffled codes, predicted activity and calibrated bound and stratum absent from the packet, code-to-identifier map held separately under a recorded seed. |
| H Frozen reporting skeleton | KEEP | H.2 | A blank prespecified 12-row table. Preserve as a prespecified reporting table so the prospective work reads as prospective. |
| I Extended comparison, incl. Remark 6.1 | KEEP | Related work or F.1 | Remark 6.1, that a catalogue action space cannot emit a component outside the closure of its block set, is the sharpest statement of the paper's difference from SynFlowNet-style methods. The honest counterweight paragraph must travel with it. |
| J Tables S1--S16 | see below | --- | --- |

---

## Machine-generated tables S1--S16

Regenerate through the supplement script; do not retype.

| Table | Decision | New location | Reason |
|---|---|---|---|
| S1 Evidence tiers | HISTORICAL | B.6 | Tier machinery is retired as authority. |
| S2 The forty locked candidates | HISTORICAL | F.9 | Keep the candidate table; the `LCB90` column is provenance, not a ranking quantity. |
| S3 Generated designs to locked panel | KEEP | D.3 or F | The funnel: 32,768 drawn, 30,979 valid and exactly reconstructed, 30,180 screened, 26,235 distinct admitted, 1,367 eligible, 40 selected. No repair, resampling or replacement. |
| S4 Novelty against each reference set | KEEP | A.1 | Already matches the current ledger at 83.7% and 86.3%. Adds a third row not currently in the paper: 25,982 of 26,235, or 99.0%, carry at least one unmeasured component. |
| S5 Component space | KEEP | A.1 or D | Measured against enumerated against corpus against scoreable, per role. The aldehyde role expands 14.8-fold over the measured set, 11 to 163. |
| S6 Panel composition | HISTORICAL | F.9 | Tied to the retired high/low arm construction. |
| S7 Synthetic burden | KEEP | F.4 | 49 catalogue materials, 53 purchased, 67 prepared, 13 candidates at two steps, 27 at four. |
| S8 The selection rule, as frozen | HISTORICAL | F.9 | **Answers the deferred candidate-selection-hygiene question**: ranking statistic, arm construction, diversity rule of pairwise product Tanimoto at most 0.95 and at most four per amine head, deterministic tie-break on canonical SMILES. It documents the rule that produced Library 0; it is not the prospective rule. |
| S9 Conformal calibration and tiers | HISTORICAL | B.6 | Also explains *why* the method was retired: the quantile takes four discrete values across the pool, so within a stratum the ranking is identical to ranking by point prediction, and the head-generalization stratum is calibrated on 42 designs. Worth keeping as the reason, not as the method. |
| S10 Every component used | REGEN | F.4 | Live. |
| S11 Synthesis route per candidate | REGEN | F.4 | Live and load-bearing. |
| S12 Bill of materials | REGEN | F.4 | Live. |
| S13 The routing algorithm | KEEP | F.4 | Pseudocode including both tail disconnections and depth and step limits. |
| S14 Architecture and parameter budget | KEEP | A.5 | 1,159,916 parameters; 4-layer bidirectional GRU at hidden 192; dropout 0.15; per-block parameter shares; 8 reverse transitions. |
| S15 State space and declared support | KEEP | A.3 | All seven categorical vocabularies and all eight support bounds. Directly fills the A.3 stub. |
| S16 Training and checkpoint selection | KEEP | A.6 | AdamW, batch 128, lr 1e-4, weight decay 5e-5, clip 1.0, deterministic float32 on one L4, flow time uniform on [0.02, 0.98], 5,100 fixed updates. |

---

## Nature Biotechnology draft

**Correction to an earlier assessment.** The prospective Methods are largely `[TBD]`
placeholders, not written protocols. They supply structure and a few genuine prespecification
commitments, not procedures.

| Old material | Decision | New location | Reason |
|---|---|---|---|
| Methods: datasets, standardization, Ugi records, expanded support, representation, model, training | UPDATE | A | Overlaps the ICLR appendix; take whichever statement is more complete. |
| Methods: generative baselines and molecular evaluation | UPDATE | D | Rebuild on the 38-descriptor contract and Amendment 15. |
| Methods: activity prediction, comparator models | KEEP | B.1 | This is the predictor-baseline material that belongs beside the support definition. |
| Methods: predictive support and conformal ranking | SPLIT | B.2--B.4 live, B.6 historical | Support survives; conformal ranking does not. |
| Methods: support-aware generation | KEEP | B, and Section 3 | Live. |
| Methods: component routing, route recovery | KEEP | F.4, F.5 | Live. |
| Methods: synthesis-guidance diagnostic | UPDATE | F | Check against the closed synthesis-guidance decision before reuse. |
| Methods: prospective selection and lock; chemistry review | KEEP | F.8, F.9 | The blinding and lock protocol. |
| Prospective experimental methods, 15 subsections | KEEP AS SCAFFOLD | H | Section headings are the right skeleton. Bodies are `[TBD]`; do not fill with generic laboratory boilerplate. |
| Statistical analysis | KEEP | H | Genuinely written and worth preserving verbatim in substance: candidate is the inferential unit, animal is the unit in vivo, technical measurements from one candidate are not independent biological replicates, predicted-versus-measured association is an evaluation of the frozen ranking strategy rather than an unbiased model comparison because the panel was selected with the primary predictor, and no prospective result modifies the generator, activity model, route procedure or selection rule. |
| Data and code availability | UPDATE | E | Do not claim availability until it exists. |
| Chemistry dossier | KEEP | F.8 | Three reproduced pages plus the code-to-identifier map held separately. |

---

## Things the old material has that the current skeleton lacks entirely

1. The split-before-component-extraction leakage rule.
2. Remark 6.1, catalogue unreachability.
3. The blinded chemistry review protocol.
4. The frozen prospective reporting skeleton.
5. Architecture selection under prespecified hard gates on a separate 1,024-program draw, and
   the fact that likelihood did not select the production model.
6. The 99.0% unmeasured-component row.
7. The frozen candidate-selection rule, which is the deferred selection-hygiene question,
   already answered in frozen form for Library 0.

## Things that must not be resurrected

Stratified conformal ranking as the active method; Proposition 4.1; evidence tiers as ranking
authority; Table S9 framed as an active guarantee; LCB90 as a current ranking statistic. Every
one of these appears in the old draft as live methodology and must be relabelled before reuse.

## Numbering conflict to resolve

The old draft numbers its propositions 3.1, 3.2, 4.1, 4.2 against old section numbers that no
longer exist. Renumber to G.1--G.4 and delete the duplicate support-preservation statement.
