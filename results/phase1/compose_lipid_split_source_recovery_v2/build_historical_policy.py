"""Extract the original v8 morphology role contract as data, without executing it."""

import ast
import base64
import hashlib
import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    tree_path = ROOT / "results/phase1/compose_lipid_v8_source_recovery_v1/commit-tree.json"
    tree = json.loads(tree_path.read_text())
    if tree["truncated"] is not False:
        raise ValueError("Incomplete original source tree")
    blobs = {entry["path"]: entry for entry in tree["tree"]}
    assets = {}
    for source, response in (
        ("scripts/build_post_instruction_generator_splits_v8.py", "old-split-driver-blob.json"),
        ("src/compose_lipid/data/generator_splits.py", "old-split-module-blob.json"),
    ):
        path = HERE / "upstream" / source
        raw = path.read_bytes()
        response_path = HERE / response
        payload = json.loads(response_path.read_text())
        git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if (
            git_hash != blobs[source]["sha"]
            or len(raw) != blobs[source]["size"]
            or payload["sha"] != git_hash
            or base64.b64decode(payload["content"]) != raw
        ):
            raise ValueError("Historical split source authentication failed")
        assets[source] = {
            "file": pin(ROOT, path),
            "response": pin(ROOT, response_path),
            "git_blob_sha1": git_hash,
        }
    current_path = ROOT / "data/vendor/compose_lipid_frozen_split_policy_v1.json"
    current = json.loads(current_path.read_text())
    recovery = json.loads(
        resolve_pin(current["source_recovery"], ROOT, label="corrected split source").read_text()
    )
    current_driver = next(
        a
        for a in recovery["assets"]
        if a["upstream_path"] == "scripts/build_post_instruction_generator_splits_v8_1.py"
    )
    current_tree = ast.parse(
        resolve_pin(
            {k: current_driver[k] for k in ("path", "sha256")}, ROOT, label="corrected driver"
        ).read_text()
    )
    old_tree = ast.parse(
        (HERE / "upstream/scripts/build_post_instruction_generator_splits_v8.py").read_text()
    )
    old_module = ast.parse(
        (HERE / "upstream/src/compose_lipid/data/generator_splits.py").read_text()
    )

    def function(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def executable_body(tree):
        body = function(tree, "molecule_profile").body
        if (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return [ast.dump(node, include_attributes=False) for node in body]

    if executable_body(old_tree) != executable_body(current_tree):
        raise ValueError(
            "Historical molecular descriptor differs from the reproduced corrected descriptor"
        )
    fields = next(
        ast.literal_eval(n.value)
        for n in old_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TOPOLOGY_FIELDS" for t in n.targets)
    )
    if sorted(fields) != current["metadata_fields"]:
        raise ValueError("Historical morphology context fields differ")
    definitions = {}
    for branch in function(old_tree, "virtual_instances").body:
        if not isinstance(branch, ast.If):
            continue
        test = branch.test
        if (
            not isinstance(test, ast.Compare)
            or not isinstance(test.left, ast.Name)
            or test.left.id != "family"
            or len(test.comparators) != 1
        ):
            raise ValueError("Unsupported historical family dispatch")
        values = ast.literal_eval(test.comparators[0])
        families = [values] if isinstance(values, str) else sorted(values)
        bindings = {
            n.targets[0].id: n.value
            for n in branch.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }
        returned = next(n.value for n in branch.body if isinstance(n, ast.Return))
        role_rules = []
        if "roles" in bindings:
            role_rules = [
                {"role": role, "quantity": {"constant": 1}}
                for role in ast.literal_eval(bindings["roles"])
            ]
        else:
            for call in ast.walk(returned):
                if (
                    not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Name)
                    or call.func.id != "instance"
                ):
                    continue
                quantity = call.args[2] if len(call.args) == 3 else ast.Constant(value=1)
                if isinstance(quantity, ast.Name):
                    quantity = bindings[quantity.id]
                if isinstance(quantity, ast.Constant) and type(quantity.value) is int:
                    rule = {"constant": quantity.value}
                elif (
                    isinstance(quantity, ast.Subscript)
                    and isinstance(quantity.value, ast.Name)
                    and quantity.value.id == "metadata"
                ):
                    rule = {"metadata": ast.literal_eval(quantity.slice)}
                else:
                    raise ValueError("Unsupported historical role multiplicity")
                role_rules.append({"role": ast.literal_eval(call.args[0]), "quantity": rule})
        if not role_rules:
            raise ValueError("Historical role inventory was not extracted")
        for family in families:
            if family in definitions:
                raise ValueError("Duplicate historical family contract")
            definitions[family] = sorted(role_rules, key=lambda r: r["role"])

    def labels(tree):
        return [
            n.args[0].elts[0].value
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "digest"
            and n.args
            and isinstance(n.args[0], ast.List)
            and n.args[0].elts
            and isinstance(n.args[0].elts[0], ast.Constant)
        ]

    driver_labels = labels(function(old_tree, "build"))
    module_labels = labels(function(old_module, "structural_group_signature"))
    if len(module_labels) != 1:
        raise ValueError("Historical structural group label is ambiguous")
    policy = {
        "schema_version": "forge.compose_lipid_historical_morphology_policy.v1",
        "derivation": pin(ROOT, Path(__file__).resolve()),
        "source_tree": pin(ROOT, tree_path),
        "source_assets": assets,
        "corrected_policy": pin(ROOT, current_path),
        "profile_executable_ast_identical": True,
        "metadata_fields": sorted(fields),
        "source_anchor_roles": "supplied_complete_source_instances_requiring_original_signature_replay",
        "virtual_role_descriptors": definitions,
        "signature_labels": {
            "core": next(label for label in driver_labels if "family_core" in label),
            "regional": next(label for label in driver_labels if "regional_topology" in label),
            "structural": module_labels[0],
        },
        "scope": "Historical partition descriptors only; never modify actual complete component inventories or quantities.",
        "training_admitted": False,
    }
    dump(ROOT / "data/vendor/compose_lipid_historical_morphology_policy_v1.json", policy)


if __name__ == "__main__":
    main()
