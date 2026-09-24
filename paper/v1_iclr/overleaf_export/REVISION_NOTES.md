# Mathematical revision and experimental extension

## Substantive mathematical corrections

- **Verification scope:** the reported L1 evaluator uses the registered chemistry. Exact replay does not establish agreement with every sampled depth, role-position or morphology field. Forward products, recovered witnesses and catalogue reachability now use the chemistry index `c`; generation remains conditioned on the full program `P`.
- **Object types:** graphs, padded states, clean endpoints, numerical states, precursor inputs, assembly traces and training atom-origin annotations are distinguished. Comparisons between recovered input/trace pairs and exact precursor preimages use the input projection.
- **Representation:** completeness concerns the declared finite atom/bond vocabularies, maximum atom count and closure budget. The theorem proves both well-defined decoding and surjectivity for the unrestricted serialization. Program masks and production child-count restrictions remain additional constraints.
- **Statistical guarantee:** the population theorem applies to a reference coordinate-denoising objective with positive program-dependent weights. The reported three-family training used role-balanced reductions, auxiliary losses, PCGrad and clipped times. The theorem makes no optimality claim for that full procedure.
- **Production objective:** the detailed loss now includes role/core, repeat, child-count, junction and topology-conditioned chemistry terms. Training-time clipping creates endpoint masses at 0.02 and 0.98. These details are separated from the reference objective.
- **Flow law:** the conditional path is a product of coordinate mixtures. The proof establishes its forward equation and the posterior-averaged joint generator, with finite-time uniqueness and convergence of marginal laws as time approaches one. It does not assume a pathwise value at the singular endpoint.
- **Numerical sampling:** the description includes the 1e-8 denominator clamp, 0.999 outgoing-mass cap and fresh neural prediction at time one followed by constrained readout. Uncapped endpoint sampling averages to the coordinate Euler kernel; the nonlinear cap and simultaneous updates require separate approximation analysis.
- **Error bounds:** decoder contraction applies to the distribution after terminal readout. No unproved discretization or learned-readout error bound is asserted.
- **Novelty metrics:** component-novel yield counts distinct verified products with an unambiguous recovered input. Held-component recovery counts qualifying attempts, including duplicates. Role-indexed constitution sets replace an ill-typed tuple/set intersection. Neither metric establishes strong catalogue escape.
- **Sampling budgets:** the binomial result counts accepted attempts under independent, identically distributed outcomes. It includes zero budgets and acceptance probabilities zero and one, and distinguishes unequal fixed strata and unique-product counts.
- **Routes:** termination requires finite expansion, depth and terminating local calls. Dossier completeness is an explicit admission invariant including a unique verified root, tree consistency, upstream evidence/replay and terminal availability; it is not a synthesis-success guarantee.

## Editorial and organizational changes

The essential population theorem is in the main method section. Supporting definitions and proofs follow their dependencies in Appendix C. The appendix no longer repeats the main conditioning equations. The inference algorithm and overview figure use the same state, verifier and output notation. Existing reference labels and the set of historical empirical result macros were retained.

## Historical implementation correspondence

`mathematical_review.json` records the checked run artifacts and source hashes. All three final evaluation results and their training-result, design and checkpoint pins matched the recorded hashes. The complete seed-0 evaluation source fingerprint matched commit `ac9ef87be4c8e5c14c477bc194a333344e621d3e`.

No exact retained-commit match was found for the seed-1/seed-2 evaluation fingerprint among the 36 historical commits checked. A complete training source manifest was not recovered. The implementation descriptions use the matched historical source and the hash-verified per-seed settings; current development code is not presented as proof of what generated the historical results. No training or production evaluation was rerun. The proof review is informal, not machine-checked formal verification.

## Author-supplied in vivo extension

The author confirmed that FORGE-1, FORGE-2 and FORGE-3 came from the historical three-family model and supplied their structure drawings. The main experimental figure contains those drawings, the unaltered mouse image and a vector plot of the four reported ROI values. Original images, transcribed values and hashes are preserved in `experimental/in_vivo/`.

The Results, abstract, introduction, discussion and experimental methods now acknowledge this exploratory experiment. The comparison is descriptive; no replicate count, error bars, statistical significance, organ targeting, tolerability or therapeutic efficacy is inferred. The PNAS synthesis reference is Chen et al. (2023), DOI 10.1073/pnas.2309472120.

Before submission, the experimental record still needs biological replicate counts, animal/site allocation, signal units and background correction, the final administration/imaging protocol, formulation and analytical details, and candidate/checkpoint identifiers. The supplied draft states intravenous administration, but this route is omitted from manuscript prose pending confirmation of the treatment/site assignments. No new experiments were initiated.

## Figure presentation

Figure 1 is centered across the text width. Figure 2 contains the generated lipid and recovered
precursor columns; the reaction-coordinate column is omitted. Figure 3 uses blue FORGE bars, a
gray MC3 bar and a dashed red MC3 reference line with a legend. All four ROI values are unchanged.
The generated two-dimensional examples and atlas use monochrome vector drawings rendered from
their saved structures. The assembly schemes use the same thin gray bonds and dark atom labels.
The atlas pairs each generated graph and its canonical SMILES with exact L1 building blocks,
replacing the decorated 3D column. The original experimental drawings and mouse image are retained. `figures/chemical_drawings/rendering_record.json` records the redraw inputs and outputs.

## Disclosure statements

The AI-use statement now includes the research, software, analysis and writing roles declared by the
author, with author responsibility for final outputs. The ethics statement acknowledges the three
synthesized lipids and exploratory mouse measurements already included in the paper. The author confirmed approval by
Children's Hospital of Philadelphia and requested omission of the protocol number. The reproducibility statement uses
FORGE terminology and points to the new artifact-provenance subsection, including the historical
source limitations and the bulk artifacts absent from the standalone manuscript archive.

## Build validation

The final build and export checks are recorded in `iclr_validation.json`. Mathematical corrections change definitions and guarantee scope, not the stored historical computational measurements. Page-budget editing remains deferred at the author's request.
