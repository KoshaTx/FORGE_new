from __future__ import annotations

import json
from pathlib import Path

from forge.model.defog_feasibility import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/shared_synthesis_program_representation_v1.json"
RESULT = REPO / "results/phase1/shared_synthesis_program_representation_v1/result.json"
EXPERIMENT = REPO / "experiments/phase1/multireaction/shared_representation.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_full_shared_representation_census_is_lossless_and_hash_attributed() -> None:
    result = _json(RESULT)
    assert result["status"] == "pass"
    assert result["gates"] == {
        "all_declared_records_attempted_and_represented": True,
        "all_declared_core_positions_observed": True,
        "aromatic_atoms_and_bonds_preserved": True,
        "fixed_ugi_core_is_explicit": True,
        "full_closure_support_preserved": True,
        "full_heavy_atom_support_preserved": True,
        "no_component_identity_fields": True,
    }
    assert result["summary"] == {
        "programs": 3,
        "records_attempted": 113_150,
        "records_represented": 113_150,
        "atom_rows": 4_472_034,
        "representation_sha256": (
            "ef10c20f443ee8a272b916a6da76556cdbaca9e53d58cae75abb5c6069301534"
        ),
        "component_identifiers_used": False,
        "fragment_tokens_used": False,
        "biological_labels_used": False,
    }
    assert result["support"] == {
        "declared_maximum_closures": 3,
        "declared_maximum_heavy_atoms": 194,
        "observed_maximum_closures": 3,
        "observed_maximum_heavy_atoms": 194,
    }
    assert result["programs"]["ugi_3cr_agile"]["records_represented"] == 112_386
    assert result["programs"]["bl_2023_repeated_aza_michael"]["records_represented"] == 610
    assert result["programs"]["lx_2024_repeated_reductive_amination"]["records_represented"] == 154
    assert all(value["errors"] == [] for value in result["programs"].values())


def test_representation_config_and_experiment_repeat_exact_input_pins() -> None:
    config = _json(CONFIG)
    result = _json(RESULT)
    experiment = _json(EXPERIMENT)
    stage = experiment["stages"][0]

    assert result["config"]["sha256"] == sha256_file(CONFIG)
    assert stage["config"] == {
        "path": str(CONFIG.relative_to(REPO)),
        "sha256": sha256_file(CONFIG),
    }
    assert config["inputs"] == stage["inputs"]
    assert {
        label: {"path": receipt["path"], "sha256": receipt["sha256"]}
        for label, receipt in result["inputs"].items()
    } == config["inputs"]
    assert "production" in " ".join(experiment["nonclaims"]).lower()
