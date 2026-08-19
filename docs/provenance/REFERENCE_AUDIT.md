# FORGE manuscript reference audit

Last verified: 2026-08-01

This ledger records primary-source checks for recent papers that carry the
field-positioning or data-curation argument in the working manuscript. It is a
drafting control, not a substitute for the final reference manager export.

| Manuscript ref. | Verified primary source | Citation facts checked | Claim boundary used in FORGE |
|---|---|---|---|
| 3 | [Li et al., Nature Biotechnology (2023)](https://www.nature.com/articles/s41587-023-01679-x) | *Nature Biotechnology* 41, 1410–1415; pulmonary mRNA delivery and genome editing | Experimental combinatorial discovery platform; not evidence of open-ended molecule–route generation |
| 4 | [Li et al., Nature Materials (2024)](https://www.nature.com/articles/s41563-024-01867-3) | *Nature Materials* 23, 1002–1008 | Machine-learning ranking over a chemistry-defined candidate universe |
| 5 | [Witten et al., Nature Biotechnology (2025)](https://www.nature.com/articles/s41587-024-02490-y) | *Nature Biotechnology* 43, 1790–1799; >9,000 activity measurements; 1.6 million in-silico lipids; mouse and ferret pulmonary evaluation | Strong predictive-design comparator; its final candidate space remains synthesis-class defined |
| 6 | [Xu et al., Nature Communications (2024)](https://www.nature.com/articles/s41467-024-50619-z) | *Nature Communications* 15, 6305; 1,200 measured Ugi products in HeLa and RAW 264.7 cells | Primary source for AGILE chemistry and labels; predictive rather than open-ended product–route generation |
| 7 | [Wang et al., Nature Communications (2024)](https://www.nature.com/articles/s41467-024-55072-6) | *Nature Communications* 15, 10804; apparent-pKa and delivery models; two-stage virtual screening with prospective synthesis | AI-guided generation/screening comparator; do not imply recursive precursor-route grounding |
| 8 | [Zhou et al., Nature Biotechnology (2026)](https://www.nature.com/articles/s41587-026-03109-0) | Published 28 April 2026; MOLEA; cartilage-selective delivery and in-vivo editing | Biological and multiobjective optimization comparator; not a matched test of synthesis-guided versus post-hoc generation |
| 9 | [Xu et al., Cell (2026)](https://doi.org/10.1016/j.cell.2026.01.012) | *Cell* 189, 1620–1635.e25; >1,700 autonomously synthesized/tested lipids; LUMI-6 lung epithelial editing | Autonomous Ugi-library experimentation comparator; does not eliminate FORGE's upstream-route question |
| 10 | [Maganti et al., OpenReview (2026)](https://openreview.net/forum?id=6RFQqfjD06) | Synthesis-constrained discrete diffusion for ionizable-lipid generation | Historical fixed-scaffold generative baseline; do not present as recursive complete-route generation |
| 11 | [Ou et al., NeurIPS workshop / arXiv (2024)](https://arxiv.org/abs/2412.00928) | Generative synthesis-DAG approach using synthetically accessible building blocks | Synthesis-native comparator; building-block accessibility is not identical to complete precursor-tree evidence |
| 12 | [Mehradfar et al., Communications AI & Computing (2026)](https://www.nature.com/articles/s44488-026-00007-x) | Published 14 July 2026; volume 1, article 2; 1,100 curated HeLa structures; reports 235 SMILES-label mismatches and 100 duplicate/isomer-related entries in the prior release | Independent motivation for source reconciliation and strong fingerprint/descriptor baselines; not evidence that any oracle is reliable outside its applicability domain |

## Drafting controls

- Cite the primary paper for scientific claims; use repositories only for code,
  data or immutable artifact provenance.
- Keep reported candidate counts, assay endpoints and in-vivo claims attached
  to the specific study that measured them.
- Do not use predictive performance from a random or scaffold split as evidence
  of prospective biological performance outside the measured assay.
- Do not describe a finite reaction-enumerated library as open-ended generation.
- Recheck all 2026 bibliographic metadata at submission because pagination,
  issue assignment and author display can change after online publication.
