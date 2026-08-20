from __future__ import annotations

from rdkit import Chem

from forge.design.flow.ugi_adapter_features import CORE_POSITION_TO_INDEX, ORIGIN_TO_INDEX
from forge.design.corpus.ugi_generated_components import precursor_components_from_product_semantics


def test_precursor_components_restore_exact_ugi_handles() -> None:
    product = Chem.MolFromSmiles(
        "[CH3:10][CH2:11][NH:4][C:3](=[O:5])[CH:2]([CH3:20])" "[NH:1][CH2:30][CH3:31]"
    )
    assert product is not None
    origin_by_map = {
        1: "amine_head",
        30: "amine_head",
        31: "amine_head",
        2: "oxoester_aldehyde_body_tail",
        20: "oxoester_aldehyde_body_tail",
        3: "isocyanide_tail",
        4: "isocyanide_tail",
        10: "isocyanide_tail",
        11: "isocyanide_tail",
        5: "assembly_introduced",
    }
    origins = []
    positions = []
    for atom in product.GetAtoms():
        map_number = atom.GetAtomMapNum()
        origins.append(ORIGIN_TO_INDEX[origin_by_map[map_number]])
        positions.append(
            CORE_POSITION_TO_INDEX[f"map_{map_number}"]
            if map_number in {1, 2, 3, 4}
            else (
                CORE_POSITION_TO_INDEX["template_introduced_0"]
                if map_number == 5
                else CORE_POSITION_TO_INDEX["not_core"]
            )
        )
        atom.SetAtomMapNum(0)
    observed = precursor_components_from_product_semantics(product, origins, positions)
    assert observed == {
        "amine_head": "CCN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]CC",
    }
