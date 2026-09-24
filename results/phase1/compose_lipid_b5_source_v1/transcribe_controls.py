"""Independent B5 SI structure transcriptions; no corpus products supply these controls."""

from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.repeated_components import element_inventory
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SI = (
    ROOT
    / "results/phase1/compose_lipid_all_family_sources_v1/remaining/39502027/ADHM-14-0-s001.pdf"
)
CORE = "CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)O"
TAIL6 = "OCC(OC(=O)C(CCCCCC)CCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCC"
TAIL11 = "O=C(O)CCCC(=O)OCC(CCCCCC)CCCCCCCC"
TAIL18 = "O=C(O)CC(=O)OCC(CCCCCC)CCCCCCCC"
HEAD9A = "CN(C)CCC(=O)O"
ACID23 = (
    "CC(C)(COC(=O)CC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)O"
)
ACID27 = "CC(C)(COC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)O"


def main():
    controls = [
        dict(
            label="I71",
            profile="I7",
            pages=[7, 10, 11, 12],
            components={"vitamin_b5_core": CORE, "tail6_alcohol": TAIL6, "head_acid": HEAD9A},
            expected_stage_products=[
                "CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)OCC(OC(=O)C(CCCCCC)CCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCC"
            ],
            expected_product="CC(C)(COC(=O)CCN(C)C)[C@@H](O)C(=O)NCCC(=O)OCC(OC(=O)C(CCCCCC)CCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCC",
            expected_formula="C49H92N2O10",
            reported_m_plus_h=869.6825,
            observed_m_plus_h=869.6839,
            yield_percent=46,
        ),
        dict(
            label="I81",
            profile="I8",
            pages=[16, 17, 18, 19],
            components={
                "vitamin_b5_core": CORE,
                "r2_tail_alcohol": "CCCCCCCCCCCO",
                "head_acid": HEAD9A,
                "tail_acid": TAIL11,
            },
            expected_stage_products=[
                "CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)OCCCCCCCCCCC",
                "CC(C)(COC(=O)CCN(C)C)[C@@H](O)C(=O)NCCC(=O)OCCCCCCCCCCC",
            ],
            expected_product="CC(C)(COC(=O)CCN(C)C)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)OCCCCCCCCCCC",
            expected_formula="C46H86N2O9",
            reported_m_plus_h=811.6406,
            observed_m_plus_h=811.6538,
            yield_percent=87,
        ),
        dict(
            label="I91",
            profile="I9_hetero_amine",
            pages=[25, 26, 27, 28, 29, 30],
            components={
                "vitamin_b5_core": CORE,
                "tail_acid_1": TAIL18,
                "tail_acid_2": TAIL11,
                "head_nucleophile": "CN(C)CCN",
            },
            expected_stage_products=[ACID23],
            expected_product="CC(C)(COC(=O)CC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)NCCN(C)C",
            expected_formula="C53H99N3O10",
            reported_m_plus_h=938.7403,
            observed_m_plus_h=938.7571,
            yield_percent=82,
        ),
        dict(
            label="I93",
            profile="I9_hetero_alcohol",
            pages=[25, 26, 27, 28, 29, 31],
            components={
                "vitamin_b5_core": CORE,
                "tail_acid_1": TAIL18,
                "tail_acid_2": TAIL11,
                "head_nucleophile": "CN(C)CCO",
            },
            expected_stage_products=[ACID23],
            expected_product="CC(C)(COC(=O)CC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)OCCN(C)C",
            expected_formula="C53H98N2O11",
            reported_m_plus_h=939.7243,
            observed_m_plus_h=939.7275,
            yield_percent=90,
        ),
        dict(
            label="I95",
            profile="I9_homo_amine",
            pages=[25, 26, 27, 32, 33, 34],
            components={
                "vitamin_b5_core": CORE,
                "tail_acid_1": TAIL11,
                "tail_acid_2": TAIL11,
                "head_nucleophile": "CN(C)CCN",
            },
            expected_stage_products=[ACID27],
            expected_product="CC(C)(COC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)NCCN(C)C",
            expected_formula="C55H103N3O10",
            reported_m_plus_h=966.7716,
            observed_m_plus_h=966.7734,
            yield_percent=80,
        ),
        dict(
            label="I97",
            profile="I9_homo_alcohol",
            pages=[25, 26, 27, 32, 33, 35, 36],
            components={
                "vitamin_b5_core": CORE,
                "tail_acid_1": TAIL11,
                "tail_acid_2": TAIL11,
                "head_nucleophile": "CN(C)CCO",
            },
            expected_stage_products=[ACID27],
            expected_product="CC(C)(COC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)[C@@H](OC(=O)CCCC(=O)OCC(CCCCCC)CCCCCCCC)C(=O)NCCC(=O)OCCN(C)C",
            expected_formula="C55H102N2O11",
            reported_m_plus_h=967.7556,
            observed_m_plus_h=967.7586,
            yield_percent=85,
        ),
    ]
    for c in controls:
        m = Chem.MolFromSmiles(c["expected_product"])
        assert rdMolDescriptors.CalcMolFormula(m) == c["expected_formula"], c["label"]
        assert (
            Chem.FindMolChiralCenters(Chem.MolFromSmiles(c["components"]["vitamin_b5_core"]))[0][1]
            == "R"
        )
        c["computed_m_plus_h"] = rdMolDescriptors.CalcExactMolWt(m) + 1.007276466621
        assert abs(c["computed_m_plus_h"] - c["reported_m_plus_h"]) < 0.00015, c["label"]
        c["observed_mass_error_ppm"] = (
            (c["observed_m_plus_h"] - c["computed_m_plus_h"]) / c["computed_m_plus_h"] * 1e6
        )
        left = sum((element_inventory(s) for s in c["components"].values()), Counter())
        right = element_inventory(c["expected_product"])
        n = len(c["components"]) - 1
        right.update({"H": 2 * n, "O": n})
        assert left == right, c["label"]
        c.update(
            net_water_equivalents=n,
            full_inventory_balance=True,
            source_asset="si",
            source_locator={"pdf_pages": c.pop("pages"), "compound": c["label"]},
            evidence_basis="computed_transform_consistency",
            expected_computed_consistency_pass=True,
            expected_stage_widths=[1] * (len(c["expected_stage_products"]) + 2),
            experimental_execution_admitted=False,
        )
        c["expected_stage_products"].append(c["expected_product"])
    dump(
        HERE / "control-transcriptions.json",
        {
            "schema_version": "forge.b5_source_control_transcriptions.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "assets": {"si": pin(ROOT, SI), "article": pin(ROOT, SI.parent / "article.xml")},
            "doi": "10.1002/adhm.202403366",
            "pmid": "39502027",
            "controls": controls,
            "source_stereochemistry": "D-pantothenic R retained in transcriptions; model constitution explicitly stereo-free; no stereochemical prediction claim.",
            "stage_interpretation": {
                "I7_I8_first": "Net composite of source acetal protection, carboxyl ester coupling, and acetal removal, ending at independently drawn isolated diol 8 or 14. Protecting groups/reagents are not corpus components; no direct unprotected selective esterification claim.",
                "I9_first": "Net composite of source Fm protection, site-resolved two-tail installation and Fm removal, ending at independently drawn isolated acid 23 or 27. Hetero-tail primary then secondary attachment is retained; homo-tail pair is identical. No simultaneous heterotail or isolated free-acid mono-adduct claim.",
            },
            "source_discrepancies": [
                {
                    "locator": "SI PDF p.6 vs p.25 Fig.S3 and p.32 compound26",
                    "issue": "General paragraph assigns RT4 to homo-pair; detailed scheme and procedure use tail11 (RT3/glutarate).",
                    "policy": "Use explicit compound11 identity from structure and detailed procedure; retain paragraph conflict.",
                },
                {
                    "locator": "SI p.6 vs p.25 Fig.S3 vs p.32 compound26",
                    "issue": "Homo-tail general time 2h, scheme 0 C/14h, detailed procedure room temperature/14h following5 C addition.",
                    "policy": "No unique exact condition label; computed connectivity only.",
                },
                {
                    "locator": "SI p.35 I97 heading vs drawing/procedure/formula pp.35-36",
                    "issue": "Heading repeats amide name but drawing and alcohol25a/EDCI procedure and N2O11 formula support ester.",
                    "policy": "Source conflict retained; computed ester connectivity only.",
                },
                {
                    "locator": "SI pp.19,30 I81/I91 HRMS",
                    "issue": "Reported found values differ from calculated [M+H]+ by more than10ppm.",
                    "policy": "Record numeric discrepancy; do not claim unambiguous analytical validation or exact-execution admission.",
                },
                {
                    "locator": "SI pp.18,20 source intermediate NMR listings",
                    "issue": "Some reported resonances resemble acetonide despite drawn deprotected intermediate; no independent spectral assignment adjudication performed.",
                    "policy": "Drawing/product-specific net formula and source site locators support computed controls only.",
                },
            ],
            "training_admitted": False,
        },
    )


if __name__ == "__main__":
    main()
