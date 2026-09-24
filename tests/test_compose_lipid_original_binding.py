"""Original task provenance must preserve the protected population and full recipe."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus import compose_lipid_original_binding as module


def example():
    prepared = {
        "eligible_for_program_preparation": True,
        "family": "fixture",
        "component_instances": [["head", "global-a", 1], ["tail", "global-b", 1]],
        "constitution_id": hashlib.sha256(b"CCO").hexdigest(),
    }
    construction = {
        "target_id": "product-id",
        "task_id": "task-a",
        "declared_precursor_identifiers": [
            {"roles": ["head"], "value": "original-a"},
            {"roles": ["tail"], "value": "original-b"},
        ],
    }
    source = {
        "family": "fixture",
        "task_id": "task-a",
        "head_id": "original-a",
        "head_smiles": "NCC",
        "tail_id": "original-b",
        "tail_smiles": "O=CC",
        "stage_order_fixed": True,
        "training_admissible": False,
    }
    binding = {
        "kind": "task",
        "stage_order_field": "stage_order_fixed",
        "role_fields": {
            "head": {"id": "head_id", "smiles": "head_smiles"},
            "tail": {"id": "tail_id", "smiles": "tail_smiles"},
        },
    }
    definitions = {
        "roles": {"head": 1, "tail": 1},
        "field_to_role": {"head_id": ["head"], "tail_id": ["tail"]},
    }
    return (
        prepared,
        construction,
        source,
        binding,
        definitions,
        {"global-a": "CCN", "global-b": "CC=O"},
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_only_requested_molecular_lines_are_decoded(tmp_path, compressed):
    path = tmp_path / ("tasks.jsonl.gz" if compressed else "tasks.jsonl")
    payload = b'not-json protected molecule\n{"task_id":"eligible"}\r\ninvalid unassigned\n'
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    selected = list(module.selected_lines(path, {2}))
    assert selected == [
        (2, {"task_id": "eligible"}, hashlib.sha256(b'{"task_id":"eligible"}').hexdigest())
    ]


@pytest.mark.parametrize("lines", [{0}, {-1}, {True}, {1.5}, {"1"}, {2}])
def test_invalid_or_absent_selected_line_fails(tmp_path, lines):
    path = tmp_path / "tasks.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ComposeLipidError):
        list(module.selected_lines(path, lines))


@pytest.mark.parametrize("field", ["path", "sha256", "line", "payload_sha256", "extra"])
def test_changed_reference_cannot_bind(field):
    source = {"source": "original/tasks", "sha256": "file-hash"}
    reference = {
        "path": source["source"],
        "sha256": "file-hash",
        "line": 2,
        "payload_sha256": "row-hash",
    }
    module.authenticate_reference(reference, source, 2, "row-hash")
    reference[field] = "changed"
    with pytest.raises(ComposeLipidError, match="binding differs"):
        module.authenticate_reference(reference, source, 2, "row-hash")


def test_complete_task_binding_does_not_claim_product_or_chemistry_verification():
    args = example()
    result = module.bind_original(*args)
    assert result["original_recipe_verified"]
    assert not result["original_product_graph_verified"]
    assert not result["source_training_admissible"]
    assert not result["upstream_atom_annotations_used_as_supervision"]
    args[0]["component_instances"].reverse()
    assert module.bind_original(*args) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_id", "different-id"),
        ("head_smiles", "CCO"),
        ("task_id", "wrong-task"),
        ("family", "wrong-family"),
        ("stage_order_fixed", False),
        ("stage_order_fixed", 1),
    ],
)
def test_source_disagreement_is_retained_without_admission(field, value):
    args = example()
    args[2][field] = value
    result = module.bind_original(*args)
    assert not result["original_recipe_verified"]
    assert result["disposition"] == "source_recipe_disagreement"


@pytest.mark.parametrize("change", ["quantity", "role", "missing", "duplicate"])
def test_incomplete_or_changed_recipe_is_not_verified(change):
    args = example()
    instances = args[0]["component_instances"]
    if change == "quantity":
        instances[0][2] = 2
    elif change == "role":
        instances[0][0] = "tail"
    elif change == "missing":
        instances.pop()
    else:
        instances.append(instances[0].copy())
    assert not module.bind_original(*args)["original_recipe_verified"]


@pytest.mark.parametrize("flag", [True, None, 0])
def test_source_admission_flag_must_be_explicit_false(flag):
    args = example()
    args[2]["training_admissible"] = flag
    with pytest.raises(ComposeLipidError, match="training-admission"):
        module.bind_original(*args)


@pytest.mark.parametrize("entry", ["task", "regional", "recipe"])
def test_protected_record_is_rejected_before_any_molecular_parsing(monkeypatch, entry):
    args = example()
    args[0]["eligible_for_program_preparation"] = False

    def forbidden(*unused):
        raise AssertionError("Protected molecule was parsed")

    monkeypatch.setattr(module, "constitutional_molecule", forbidden)
    with pytest.raises(ComposeLipidError, match="Protected or unassigned"):
        if entry == "task":
            module.bind_original(*args)
        elif entry == "regional":
            module.bind_regional(args[0], args[1], {}, {}, {})
        else:
            module.compare_recipe(args[0], args[1], [], {})


def test_context_product_identity_is_checked_after_fixed_recipe():
    args = example()
    source = args[2]
    source.update(
        precursor_ids=["original-a", "original-b"],
        constitutional_smiles="OCC",
        complete_subcomponents=[
            {
                "reaction_role": "head",
                "subcomponent_id": "original-a",
                "constitutional_smiles": "NCC",
            },
            {
                "reaction_role": "tail",
                "subcomponent_id": "original-b",
                "constitutional_smiles": "O=CC",
            },
        ],
    )
    args[3].update(kind="context", source_to_registry_role={"head": "head", "tail": "tail"})
    assert module.bind_original(*args)["original_product_graph_verified"]
    source["constitutional_smiles"] = "CCCO"
    result = module.bind_original(*args)
    assert not result["original_recipe_verified"]
    assert result["checks"]["complete_role_quantity_structure_agreement"]
    source["precursor_ids"].reverse()
    with pytest.raises(ComposeLipidError, match="precursor IDs disagree"):
        module.bind_original(*args)


def test_regional_id_and_catalog_agreement_does_not_verify_unavailable_program():
    prepared, construction, _, _, _, structures = example()
    source = {
        "training_admissible": False,
        "precursor_ids": ["original-a", "original-b"],
        "tuple_id": "task-a",
        "product_id": "product-id",
        "disposition": "fixture",
        "reported_released_tuple": False,
        "source_parent_precursor_ids": ["parent"],
    }
    catalog = {
        "original-a": {"role": "head", "smiles": "NCC"},
        "original-b": {"role": "tail", "smiles": "O=CC"},
    }
    result = module.bind_regional(prepared, construction, source, catalog, structures)
    assert result["original_recipe_verified"]
    assert not result["original_product_graph_verified"]
    assert not result["original_product_program_payload_verified"]
    source["product_id"] = "wrong"
    assert not module.bind_regional(prepared, construction, source, catalog, structures)[
        "original_recipe_verified"
    ]
    del catalog["original-a"]
    with pytest.raises(ComposeLipidError, match="Unresolved regional precursor"):
        module.bind_regional(prepared, construction, source, catalog, structures)


def test_public_runner_is_deterministic_and_keeps_preparation_only(tmp_path, monkeypatch):
    prepared, construction, source, binding, definitions, structures = example()
    family = "acid_epoxide_diester_multistep"
    prepared.update(
        family=family,
        target_id="product-id",
        construction_source_line=2,
        construction_basis="original_task",
    )
    source["family"] = family
    source_path = tmp_path / "task.jsonl"
    source_payload = json.dumps(source).encode()
    source_path.write_bytes(b"protected invalid-json molecular record\n" + source_payload + b"\n")
    source_pin = module.pin(tmp_path, source_path)
    reference = {
        "path": "upstream/task.jsonl",
        "sha256": source_pin["sha256"],
        "line": 2,
        "payload_sha256": hashlib.sha256(source_payload).hexdigest(),
    }
    construction.update(
        primary_family=family,
        construction_basis="original_task",
        component_instances=[
            {"role": r, "component_id": c, "quantity": q}
            for r, c, q in prepared["component_instances"]
        ],
        provenance={"task": reference},
    )
    constructions = tmp_path / "constructions.jsonl"
    constructions.write_text(
        "protected invalid-json molecular record\n" + json.dumps(construction) + "\n"
    )
    precursors = tmp_path / "precursors.jsonl.gz"
    precursors.write_bytes(
        gzip.compress(
            "".join(
                json.dumps({"component_id": k, "constitution": v}) + "\n"
                for k, v in structures.items()
            ).encode()
        )
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    config = {
        "inputs": {},
        "families": {family: binding, "aldehyde_ugi4": {"regional_source": "regional"}},
    }
    sources = {
        reference["path"]: {
            "source": reference["path"],
            "local": source_path,
            "pin": source_pin,
            "sha256": source_pin["sha256"],
        },
        "regional": {"local": empty},
    }
    loaded = (
        config,
        {"constructions": constructions, "ugi4_catalog": empty},
        SimpleNamespace(preparation={"product-id": prepared}, precursors=precursors),
        sources,
        {family: definitions},
    )
    monkeypatch.setattr(module, "_load", lambda *args: loaded)
    monkeypatch.setattr(module, "IMPLEMENTATION", ())
    results = [
        module.run_original_binding(tmp_path, config_path, tmp_path / name)
        for name in ("first", "second")
    ]
    assert results[0]["summary"] == {
        family: {"rows": 1, "original_recipe_verified": 1, "verified_recipe_rows": 1}
    }
    assert (
        results[0]["artifacts"]["bindings.jsonl.gz"]["sha256"]
        == results[1]["artifacts"]["bindings.jsonl.gz"]["sha256"]
    )
    record = json.loads(gzip.decompress((tmp_path / "first/bindings.jsonl.gz").read_bytes()))
    assert record["component_instances"] == prepared["component_instances"]
    assert not record["training_admitted"]
    assert not record["experimental_execution_admitted"]
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.run_original_binding(tmp_path, config_path, tmp_path / "first")
