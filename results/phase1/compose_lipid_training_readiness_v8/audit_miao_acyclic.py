"""Persist the inspected Miao acyclic-branch evidence boundary without inventing controls."""

from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/phase1/compose_lipid_all_family_sources_v1/remaining/31570898"


def main():
    assets = {
        name: pin(ROOT, SOURCE / relative)
        for name, relative in {
            "supplement": "miao-si.pdf",
            "main_figure_1": "fig1.png",
            "main_figure_3": "fig3.png",
            "supplement_printed_page_3": "pages/page-04.png",
            "supplement_printed_page_4": "pages/page-05.png",
        }.items()
    }
    dump(
        HERE / "miao-acyclic-source-review.json",
        {
            "schema_version": "forge.chemistry_source_scope_review.v1",
            "date": "2026-09-21",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "assets": assets,
            "inputs": {
                "source_family_claims": pin(
                    ROOT,
                    ROOT
                    / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json",
                ),
            },
            "doi": "10.1038/s41587-019-0247-3",
            "pmid": "31570898",
            "family": "ketone_isocyanide_amide",
            "evidence_basis": "family_precedent",
            "disposition": "abstain",
            "observations": [
                {
                    "locator": "Main Fig. 1b",
                    "observation": "The generic primary-amine scheme draws imine, acyclic amide "
                    "and dihydroimidazole alternatives separated by 'or'. It does not by itself "
                    "assign a unique product to each secondary-amine library tuple.",
                },
                {
                    "locator": "Main Fig. 1c",
                    "observation": "The source library includes both primary and secondary amine "
                    "drawings. Their inclusion is library evidence, not isolated-product identity "
                    "evidence for every secondary-amine combination.",
                },
                {
                    "locator": "Main Fig. 3a; SI Note 2, printed pages 3-4, lines 113-121",
                    "observation": "The local secondary amine discussed here is in the proposed "
                    "cyclization intermediate. This sentence does not establish the separate "
                    "secondary-amine-reactant acyclic branch claimed in the generator metadata.",
                },
                {
                    "locator": "SI Note 1, named product headings; printed page 3",
                    "observation": "The inspected isolated-product characterization concerns "
                    "dihydroimidazole products. It does not supply an independently identified "
                    "acyclic product control for the proposed corpus executor.",
                },
            ],
            "remaining_evidence": [
                "An independently identified acyclic product and complete precursor tuple, with "
                "a source procedure locator and applicable branch-selection evidence.",
                "A qualified registry-backed atom/bond transformation and explicit carbonyl "
                "oxygen accounting, tested against that independent source control.",
            ],
            "conclusion": "The local supplement is available and has been inspected. Missing "
            "branch-specific control evidence, not a missing PDF, prevents this review from "
            "qualifying the proposed acyclic executor. No assertion of chemical impossibility "
            "or experimental failure is made.",
            "training_admitted": False,
            "experimental_execution_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
