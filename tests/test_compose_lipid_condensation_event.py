"""Condensation receipts preserve exclusions and reject rehashed forged evidence."""

import copy
import gzip
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_condensation_event import control  # noqa: F401

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_condensation_event as module
from forge.corpus.library_splits import FrozenIdentityFolds, constitution_id

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch, control):  # noqa: F811
    adapter, row, kwargs = control
    source = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_ugi4_source_v1/adjudication.json").read_text()
    )
    family = source["family"]
    order = ["amine", "aldehyde", "carboxylic_acid", "isocyanide"]
    config = {"family": family, "maximum_outcomes": 256, "inputs": {}}
    records = []
    for label, product in [("good", row["expected_product"]), ("no_inverse", "CCO")]:
        records.append(
            {
                "source": {
                    "target_id": label,
                    "constitution": product,
                    "primary_metadata": {
                        "precursor_ids": [label + role for role in order],
                        "design_lane": "retained_fixture_lane",
                    },
                },
                "preparation": {
                    "eligible_for_program_preparation": True,
                    "constitution_id": constitution_id(product),
                },
            }
        )
    guards = FrozenIdentityFolds()
    guards.add(constitution_id(row["components"]["amine_head"]), "heldout", "test guard")
    reader = SimpleNamespace(
        result={"summary": {"by_family": {family: {"eligible_for_program_preparation": 2}}}},
        iter_preparation_records=lambda **_: iter(records),
    )
    loaded = (config, source, adapter, reader, guards, order, kwargs)
    monkeypatch.setattr(module, "_load", lambda *_: loaded)
    for name in module.IMPLEMENTATION:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, loaded, records


def test_every_record_and_known_precursor_exclusion_is_retained(workspace):
    root, *_ = workspace
    result = module.run_condensation_event(root, Path("config.json"), Path("out"))
    assert result["summary"]["rows"] == 2
    assert result["summary"]["computed_consistency_pass"] == 1
    assert result["summary"]["historical_protected_precursor"] == 1
    assert result["summary"]["eligible_after_known_exclusions"] == 0
    assert result["summary"]["training_admitted"] == 0
    assert not result["summary"]["training_ready"]
    assert module.verify_condensation_event(root, Path("out/result.json")) == result


@pytest.mark.parametrize("mutation", ["drop", "unprotect", "water", "site"])
def test_independent_replay_rejects_forged_ledger_even_after_rehash(workspace, mutation):
    root, *_ = workspace
    result = module.run_condensation_event(root, Path("config.json"), Path("out"))
    ledger = root / result["artifacts"]["programs.jsonl.gz"]["path"]
    with gzip.open(ledger, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if mutation == "drop":
        rows.pop()
    elif mutation == "unprotect":
        rows[0]["historical_protected_precursor"] = False
    elif mutation == "water":
        rows[0]["net_byproducts"] = {}
    else:
        rows[0]["site_witnesses"] = []
    with gzip.open(ledger, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["programs.jsonl.gz"]["sha256"] = str(sha256_file(ledger))
    (root / "out/result.json").write_text(json.dumps(result))
    with pytest.raises(ComposeLipidError, match="independently replayed"):
        module.verify_condensation_event(root, Path("out/result.json"))


@pytest.mark.parametrize("mutation", ["prior_excluded", "duplicate", "lost_row", "missing_label"])
def test_bad_preparation_boundary_leaves_no_published_result(workspace, mutation):
    root, loaded, records = workspace
    if mutation == "prior_excluded":
        records[0]["preparation"]["eligible_for_program_preparation"] = False
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "lost_row":
        records.pop()
    else:
        records[0]["source"]["primary_metadata"]["precursor_ids"].pop()
    with pytest.raises(ComposeLipidError):
        module.run_condensation_event(root, Path("config.json"), Path("out"))
    assert not (root / "out").exists()


def test_source_cannot_override_registry_water_contract(tmp_path, monkeypatch):
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_ugi4_program_v1.json").read_text()
    )
    source = json.loads((ROOT / config["inputs"]["adjudication"]["path"]).read_text())
    source["source_contract"]["net_byproducts"]["H"] = 1
    monkeypatch.setattr(
        module.base, "_load", lambda *_: (config, source, None, None, None, None, {})
    )
    with pytest.raises(ComposeLipidError, match="byproduct inventories differ"):
        module._load(ROOT, Path("unused.json"))
