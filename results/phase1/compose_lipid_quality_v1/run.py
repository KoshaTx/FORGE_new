"""Reproducible TRAIN-only development quality baseline and paired reassessment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from forge.core.hashing import sha256_file
from forge.model.common_lipid_realism import DESCRIPTOR_NAMES, _descriptor_vector
from forge.model.compose_lipid_component_diversity import accepted_components
from forge.model.compose_lipid_quality import (
    StructuralSupport,
    canonical_molecule,
    descriptor_comparison,
    fingerprint_distribution,
    identity_diversity,
    representation_diagnostics,
    source_role_mapping,
)
from results.phase1.compose_lipid_component_decoder_v1.contracts import load_all, matching

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CACHE = HERE.with_name("compose_lipid_mapped_preparation_v3") / "cache"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def rank(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def status_summary(assessments):
    return {
        "molecules": len(assessments),
        "unknown_neighborhood": sum(bool(a["unknown_atom_environments"]) for a in assessments),
        "unknown_ring_system": sum(bool(a["unknown_ring_systems"]) for a in assessments),
        "all_features_observed": sum(a["all_local_features_observed"] is True for a in assessments),
        "chemically_rejected": 0,
        "interpretation": "unknown support is not a chemical rejection or a quality pass",
    }


def load_references(db, policy, pins):
    saved = HERE / "reference.json"
    input_names = [
        "membership",
        "manifest",
        "cohort",
        "smiles_bytes",
        "smiles_offsets",
        "catalogue",
        "catalogue_receipt",
        "r0",
        "r0_splits",
        "policy",
        "quality_code",
        "runner",
    ]
    reference_pins = {name: pins[name] for name in input_names}
    if saved.exists():
        cached = read(saved)
        if cached["inputs"] == reference_pins:
            print("Reusing exact-pin TRAIN reference", flush=True)
            return cached
        raise ValueError("Reference inputs changed; preserve old outputs and use a new directory")
    seed = policy["seed"]
    census = defaultdict(lambda: defaultdict(set))
    query = """select distinct s.family,json_extract(j.value,'$[0]'),
               json_extract(j.value,'$[1]') from sources s,
               json_each(s.component_instances) j order by 1,2,3"""
    for family, role, identity in db.execute(query):
        census[family][role].add(identity)
    needed = {identity for roles in census.values() for ids in roles.values() for identity in ids}
    components = {}
    with gzip.open(ROOT / pins["catalogue"]["path"], "rt") as handle:
        for line in handle:
            row = json.loads(line)
            # Membership selection occurs before accessing a structure field.
            if row["component_id"] not in needed:
                continue
            parsed = canonical_molecule(row["constitution"])
            if parsed is None:
                raise ValueError("TRAIN precursor is invalid")
            canonical = parsed[0]
            if row["component_id"] in components and components[row["component_id"]] != canonical:
                raise ValueError("Global component identity maps to conflicting constitutions")
            components[row["component_id"]] = canonical
    if needed - set(components):
        raise ValueError(
            f"TRAIN catalogue lacks {len(needed - set(components))} precursor identities"
        )
    print(f"Resolved {len(components)} distinct TRAIN precursors", flush=True)
    offsets = np.load(ROOT / pins["smiles_offsets"]["path"], mmap_mode="r")
    values = np.load(ROOT / pins["smiles_bytes"]["path"], mmap_mode="r")
    products = {}
    controls = {}
    product_ids = {}
    for family in sorted(census):
        indices = [x[0] for x in db.execute("select idx from sources where family=?", (family,))]
        rng = random.Random(rank(family, seed))
        chosen = rng.sample(
            indices,
            min(
                len(indices),
                policy["product_reference_per_family"] + policy["product_control_per_family"],
            ),
        )
        structures = []
        for index in chosen:
            smiles = bytes(values[offsets[index] : offsets[index + 1]]).decode()
            parsed = canonical_molecule(smiles)
            if parsed is None or parsed[0] != smiles:
                raise ValueError("TRAIN product identity changed")
            structures.append(smiles)
        cut = policy["product_reference_per_family"]
        products[family], controls[family] = structures[:cut], structures[cut:]
        product_ids[family] = {"support": chosen[:cut], "control": chosen[cut:]}
    # Only the conservative TRAIN intersection is eligible; no TEST or CAL structure
    # is parsed, fingerprinted, copied into output, or used for development.
    with (ROOT / pins["r0_splits"]["path"]).open() as handle:
        rows = list(csv.DictReader(handle))
    fold_fields = [key for key in rows[0] if key.endswith("_fold")]
    eligible = {
        row["r0_structure_id"]: row["source_study_group_id"]
        for row in rows
        if all(row[key] == "R0_train" for key in fold_fields)
    }
    empirical = []
    empirical_counts = Counter(eligible_split_ids=len(eligible))
    with gzip.open(ROOT / pins["r0"]["path"], "rt") as handle:
        for row in csv.DictReader(handle):
            identity = row["r0_structure_id"]
            if identity not in eligible:
                continue
            parsed = canonical_molecule(row["canonical_constitutional_smiles"])
            if parsed is None or len(Chem.GetMolFrags(parsed[1])) != 1:
                empirical_counts["invalid_or_disconnected"] += 1
                continue
            smiles, mol = parsed
            constitution = hashlib.sha256(smiles.encode()).hexdigest()
            overlap = db.execute(
                "select family from sources where constitution_id=?", (constitution,)
            ).fetchone()
            empirical.append(
                {
                    "id": identity,
                    "group": eligible[identity],
                    "smiles": smiles,
                    "compose_train_overlap_family": overlap[0] if overlap else None,
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                    "elements": sorted({atom.GetSymbol() for atom in mol.GetAtoms()}),
                }
            )
            empirical_counts["connected"] += 1
            empirical_counts["compose_train_exact_overlap"] += bool(overlap)
    by_group = defaultdict(list)
    for row in empirical:
        by_group[row["group"]].append(row)
    for group in by_group:
        by_group[group].sort(key=lambda row: rank(row["id"], seed))
    # Split each source group before balanced sampling. Taking the second block
    # of a single balanced ordering would leave controls dominated by large groups.
    balanced_partitions = []
    for parity in (0, 1):
        partition = {group: items[parity::2] for group, items in by_group.items()}
        balanced = []
        for index in range(max(map(len, partition.values()), default=0)):
            for group in sorted(partition, key=lambda value: rank(value, seed)):
                if index < len(partition[group]):
                    balanced.append(partition[group][index])
        balanced_partitions.append(balanced[: policy["empirical_reference_limit"]])
    result = {
        "schema_version": "forge.compose_lipid_quality_reference.v1",
        "inputs": reference_pins,
        "seed": seed,
        "component_smiles_by_identity": components,
        "component_ids_by_family_role": {
            f: {r: sorted(ids) for r, ids in roles.items()} for f, roles in census.items()
        },
        "product_support": products,
        "product_controls": controls,
        "product_indices": product_ids,
        "empirical_reference": balanced_partitions[0],
        "empirical_controls": balanced_partitions[1],
        "empirical_selected_source_groups": [
            dict(Counter(row["group"] for row in partition)) for partition in balanced_partitions
        ],
        "empirical_census": dict(empirical_counts),
        "empirical_source_groups": {group: len(items) for group, items in by_group.items()},
        "family_matched_empirical_status": "unassessed; exact family labels unavailable",
        "heldout_structures_used": False,
    }
    write(saved, result)
    return result


def support_tables(reference, policy):
    product_support, component_support = {}, {}
    positive_audit = {}
    for family, structures in reference["product_support"].items():
        support = StructuralSupport()
        for smiles in structures:
            support.add(smiles)
        product_support[family] = support
        audit = {
            "products": status_summary(
                [
                    support.assess(canonical_molecule(s)[1])
                    for s in reference["product_controls"][family]
                ]
            ),
            "components": {},
        }
        component_support[family] = {}
        for role, ids in reference["component_ids_by_family_role"][family].items():
            support = StructuralSupport()
            fit = StructuralSupport()
            controls = []
            source_smiles = sorted(
                {reference["component_smiles_by_identity"][identity] for identity in ids}
            )
            for smiles in source_smiles:
                support.add(smiles)
                # Identity-based global assignment prevents a shared component
                # appearing in both support and holdback through different roles.
                held = (
                    int(rank(smiles, policy["seed"]), 16) / 16**64
                    < policy["component_control_fraction"]
                )
                if held:
                    controls.append(smiles)
                else:
                    fit.add(smiles)
            component_support[family][role] = support
            audit["components"][role] = {
                "source_catalogue_ids": len(ids),
                "source_unique_constitutions": len(source_smiles),
                "fit_unique": len(fit.identities),
                "positive_holdback": status_summary(
                    [fit.assess(canonical_molecule(s)[1]) for s in controls]
                ),
                "full_source_self_coverage": status_summary(
                    [support.assess(canonical_molecule(smiles)[1]) for smiles in source_smiles]
                ),
            }
        positive_audit[family] = audit
    return product_support, component_support, positive_audit


def assess_panel(
    rows, branch, product_support, component_support, reference, policy, db, executors, layouts
):
    detail = []
    by_family = defaultdict(list)
    elements = {
        a["symbol"] for a in read(ROOT / policy["allowed_elements_from"])["atom_vocabulary"]
    }
    empirical = reference["empirical_reference"]
    empirical_smiles = [row["smiles"] for row in empirical]
    empirical_vectors = np.asarray(
        [_descriptor_vector(canonical_molecule(s)[1]) for s in empirical_smiles]
    )
    for row in rows:
        family = row["family"]
        saved = (
            row[branch]
            if branch != "selected"
            else {"smiles": row["selected_smiles"], "check": row["selected_check"]}
        )
        exact = saved.get("check", {}).get("exact") is True
        item = {
            "index": row["index"],
            "family": family,
            "exact_l1": exact,
            "smiles": saved.get("smiles"),
            "structural_qualification": "unassessed",
        }
        parsed = canonical_molecule(saved.get("smiles"))
        item["parse_valid"] = parsed is not None
        item["connected"] = parsed is not None and len(Chem.GetMolFrags(parsed[1])) == 1
        if parsed is not None:
            smiles, mol = parsed
            item.update(
                {
                    "canonical_smiles": smiles,
                    **representation_diagnostics(
                        mol,
                        elements=elements,
                        maximum_heavy_atoms=policy["maximum_heavy_atoms"],
                        maximum_cycles=policy["maximum_independent_cycles"],
                    ),
                    "product_support": product_support[family].assess(mol),
                    "train_product": db.execute(
                        "select 1 from sources where constitution_id=?",
                        (hashlib.sha256(smiles.encode()).hexdigest(),),
                    ).fetchone()
                    is not None,
                    "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                    "descriptors": dict(
                        zip(DESCRIPTOR_NAMES, _descriptor_vector(mol).tolist(), strict=True)
                    ),
                }
            )
        if exact:
            if not item["connected"]:
                raise ValueError("An exact source assessment claims an invalid/disconnected graph")
            parts = accepted_components(saved["check"])
            role_mapping = source_role_mapping(
                parts, saved["check"], matching(executors, layouts[row["index"]])
            )
            item["source_role_mapping"] = role_mapping
            item["components"] = {}
            for role, smiles in parts.items():
                parsed_component = canonical_molecule(smiles)
                if parsed_component is None:
                    raise ValueError("An accepted precursor does not sanitize")
                source_role = role_mapping[role]
                support = component_support[family].get(source_role, StructuralSupport())
                item["components"][role] = {
                    "smiles": parsed_component[0],
                    "source_role": source_role,
                    "source_component_identity_seen": parsed_component[0] in support.identities,
                    **support.assess(parsed_component[1]),
                }
            item["all_component_features_observed"] = all(
                part["all_local_features_observed"] is True for part in item["components"].values()
            )
        detail.append(item)
        by_family[family].append(item)
    summary = {}
    for family, items in sorted(by_family.items()):
        n = len(items)
        connected = [row for row in items if row["connected"]]
        exact = [row for row in items if row["exact_l1"]]
        observed = [
            row
            for row in exact
            if row["all_component_features_observed"]
            and row["product_support"]["all_local_features_observed"]
        ]
        products = [row["canonical_smiles"] for row in connected]
        vectors = np.asarray(
            [[row["descriptors"][name] for name in DESCRIPTOR_NAMES] for row in connected]
        )
        role_diversity = {}
        for subset_name, subset in [("exact_l1", exact), ("all_local_features_observed", observed)]:
            roles = sorted({role for row in subset for role in row["components"]})
            role_diversity[subset_name] = {
                role: identity_diversity(
                    [
                        row["components"][role]["smiles"]
                        for row in subset
                        if role in row["components"]
                    ],
                    requests=n,
                )
                for role in roles
            }
        summary[family] = {
            "requests": n,
            "parse_valid": sum(row["parse_valid"] for row in items),
            "connected": len(connected),
            "exact_l1": len(exact),
            "within_declared_size_element_cycle_bounds": sum(
                row.get("within_declared_size_element_cycle_bounds", False) for row in items
            ),
            "products_above_194_heavy_atoms": sum(row.get("heavy_atoms", 0) > 194 for row in items),
            "exact_unknown_product_neighborhood": sum(
                bool(row["product_support"]["unknown_atom_environments"]) for row in exact
            ),
            "exact_unknown_product_ring_system": sum(
                bool(row["product_support"]["unknown_ring_systems"]) for row in exact
            ),
            "exact_unknown_component_neighborhood": sum(
                any(bool(p["unknown_atom_environments"]) for p in row["components"].values())
                for row in exact
            ),
            "exact_unknown_component_ring_system": sum(
                any(bool(p["unknown_ring_systems"]) for p in row["components"].values())
                for row in exact
            ),
            "exact_unassessed_component_role": sum(
                any(p["status"].startswith("unassessed") for p in row["components"].values())
                for row in exact
            ),
            "exact_all_local_features_observed": len(observed),
            "structurally_qualified_exact_count": None,
            "structural_qualification_status": "unassessed_no_complete_calibrated_chemistry_policy",
            "train_novel_connected_products": sum(not row["train_product"] for row in connected),
            "product_diversity": identity_diversity(products, requests=n),
            "exact_product_diversity": identity_diversity(
                [r["canonical_smiles"] for r in exact], requests=n
            ),
            "local_support_product_diversity": identity_diversity(
                [r["canonical_smiles"] for r in observed], requests=n
            ),
            "scaffold_diversity_connected": identity_diversity(
                [r["murcko_scaffold"] for r in connected], requests=n
            ),
            "component_diversity": role_diversity,
            "broad_observed_train_fingerprint": fingerprint_distribution(
                products, empirical_smiles, requests=n, neighbors=policy["fingerprint_neighbors"]
            ),
            "broad_observed_train_descriptors": descriptor_comparison(
                vectors, empirical_vectors, DESCRIPTOR_NAMES
            ),
            "family_empirical_realism": {
                "status": "unassessed",
                "reason": "no qualified exact family attribution for empirical R0 TRAIN reference",
            },
            "l2": "unassessed_by_this_module",
            "l3": "unassessed_by_this_module",
        }
    return summary, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    policy = read(HERE / "policy.json")
    paths = {
        "policy": HERE / "policy.json",
        "quality_code": ROOT / "forge/model/compose_lipid_quality.py",
        "runner": Path(__file__),
        "diversity_parser": ROOT / "forge/model/compose_lipid_component_diversity.py",
        "descriptor_code": ROOT / "forge/model/common_lipid_realism.py",
        "region_code": ROOT / "forge/model/lipid_context.py",
        "selected": HERE.with_name("compose_lipid_consistency90_v1")
        / "selected_pool_4/attempts.json",
        "selection_result": HERE.with_name("compose_lipid_consistency90_v1")
        / "selected_pool_4/result.json",
        "membership": CACHE / "sources.sqlite",
        "manifest": CACHE / "manifest.json",
        "smiles_bytes": CACHE / "smiles_bytes.npy",
        "smiles_offsets": CACHE / "smiles_offsets.npy",
        "cohort": HERE.with_name("compose_lipid_training_cohort_v1") / "cohort.json",
        "catalogue": ROOT
        / "data/source_cache/compose_lipid_supplement_2026-09-19/precursor_catalog.jsonl.gz",
        "catalogue_receipt": HERE.with_name("compose_lipid_program_origins_v1") / "request.json",
        "r0": ROOT / "results/m0_03/r0_constitutional.csv.gz",
        "r0_splits": ROOT / "data/splits/m0_03_constitutional/r0_fold_assignments.csv",
        "r0_receipt": ROOT / "configs/multireaction/common_lipid_realism_v1.json",
        "vocabulary": ROOT / policy["allowed_elements_from"],
        "layout_protocol": HERE.with_name("compose_lipid_component_decoder_v1")
        / "fresh/protocol.json",
        "layout_payload": HERE.with_name("compose_lipid_component_decoder_v1") / "fresh/input.pt",
        "role_contract_loader": HERE.with_name("compose_lipid_component_decoder_v1")
        / "contracts.py",
    }
    if args.paired:
        paths["paired"] = args.paired.resolve()
    pins = {name: pin(path) for name, path in paths.items()}
    manifest = read(paths["manifest"])
    cohort = read(paths["cohort"])
    assert manifest["by_family"] == cohort["by_family"]
    assert pins["membership"] == manifest["artifacts"]["sources.sqlite"]
    for key in ("smiles_bytes", "smiles_offsets"):
        assert pins[key] == manifest["artifacts"][key + ".npy"]
    assert pins["catalogue"] == read(paths["catalogue_receipt"])["inputs"]["catalogue"]
    assert pins["selected"] == read(paths["selection_result"])["inputs"]["attempts"]
    for key, name in [("r0", "r0_constitutional"), ("r0_splits", "r0_fold_assignments")]:
        assert pins[key] == read(paths["r0_receipt"])["inputs"][name]
    assert pins["layout_payload"] == read(paths["layout_protocol"])["inputs"]["payload"]
    torch.set_num_threads(1)
    layouts = torch.load(paths["layout_payload"], weights_only=False, map_location="cpu")["layouts"]
    with rdBase.BlockLogs():
        executors, contract_pins, _ = load_all()
    pins.update({f"source_contract:{name}": value for name, value in contract_pins.items()})
    with sqlite3.connect(paths["membership"].as_uri() + "?mode=ro", uri=True) as db:
        assert (
            dict(db.execute("select family,count(*) from sources group by family"))
            == cohort["by_family"]
        )
        reference = load_references(db, policy, pins)
        print("Building unique-component and product support", flush=True)
        product_support, component_support, controls = support_tables(reference, policy)
        write(HERE / "positive_controls.json", {"inputs": pins, "by_family": controls})
        print("Assessing selected development panel", flush=True)
        summary, details = assess_panel(
            read(paths["selected"]),
            "selected",
            product_support,
            component_support,
            reference,
            policy,
            db,
            executors,
            layouts,
        )
        panels = {"selected_pool_4": summary}
        write(HERE / "attempts.json", {"inputs": pins, "attempts": details})
        if args.paired:
            paired = read(paths["paired"])
            for branch in [
                key
                for key in paired[0]
                if isinstance(paired[0][key], dict) and "smiles" in paired[0][key]
            ]:
                print(f"Assessing paired branch {branch}", flush=True)
                panels[branch], detail = assess_panel(
                    paired,
                    branch,
                    product_support,
                    component_support,
                    reference,
                    policy,
                    db,
                    executors,
                    layouts,
                )
                write(HERE / f"paired_{branch}_attempts.json", {"inputs": pins, "attempts": detail})
    empirical = reference["empirical_reference"]
    empirical_controls = reference["empirical_controls"]
    observed_comparison = fingerprint_distribution(
        [r["smiles"] for r in empirical_controls],
        [r["smiles"] for r in empirical],
        requests=max(1, len(empirical_controls)),
        neighbors=policy["fingerprint_neighbors"],
    )
    result = {
        "schema_version": "forge.compose_lipid_quality_dashboard.v1",
        "inputs": pins,
        "reference": pin(HERE / "reference.json"),
        "policy": policy,
        "runtime_seconds": time.monotonic() - start,
        "rdkit_version": rdBase.rdkitVersion,
        "by_panel": panels,
        "empirical_census": reference["empirical_census"],
        "observed_train_vs_observed_train": observed_comparison,
        "nonclaims": [
            "Source support is not experimental execution or full chemical plausibility.",
            "Unknown environment/ring is not invalidity; observed features are not quality qualification.",
            "All requests retained. No structural quality threshold was selected from these results.",
            "Broad R0 TRAIN comparison is not family-matched or independent model generalization.",
            "Source controls are reaction-enumerated TRAIN support, not an empirical lipid validation set.",
            "No pKa, efficacy, delivery, synthesis-success or supplier-availability inference.",
        ],
    }
    write(HERE / ("paired_result.json" if args.paired else "result.json"), result)
    header = [
        "family",
        "requests",
        "exact_l1",
        "exact_unknown_component_neighborhood",
        "exact_unknown_component_ring_system",
        "exact_all_local_features_observed",
        "structurally_qualified_exact_count",
    ]
    with (HERE / "dashboard.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in header} | {"family": family}
            for family, row in summary.items()
        )
    print(
        json.dumps(
            {
                "elapsed_seconds": result["runtime_seconds"],
                "panels": list(panels),
                "empirical": result["empirical_census"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
