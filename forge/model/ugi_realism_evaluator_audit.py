"""Train-only diagnostic of descriptor resolution and drawing invariance.

This supplements frozen realism gates; fingerprint proximity is neither a new gate nor evidence
of biological activity, synthesis success, or realism independent of memorization.
"""

from __future__ import annotations

import html
import math
import random
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdDepictor, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_json
from forge.core.io import iter_csv, iter_jsonl, read_json_object, write_json
from forge.model.common_lipid_realism import UGI_PRECURSOR_ROLES
from forge.model.ugi_development_realism import ROLE_DESCRIPTOR_NAMES, ugi_role_descriptor_vector

CONFIG_SCHEMA = "forge.ugi_realism_evaluator_audit_config.v1"
RESULT_SCHEMA = "forge.ugi_realism_evaluator_audit.v1"
FINGERPRINTS = ("morgan_count_radius2", "atom_pair_count")


class EvaluatorAuditError(ValueError):
    """An audit input violates its explicit development-only contract."""


@dataclass(frozen=True)
class AuditProduct:
    identity: str
    smiles: str
    components: tuple[tuple[str, str], ...]
    source: str
    group: str
    origin: str
    attempt_index: int | None = None
    inherited_novelty: str = ""

    def roles(self) -> dict[str, str]:
        return dict(self.components)


def canonical_graph(smiles: str) -> str:
    """Constitution-only identity, independent of atom order and 2D coordinates."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise EvaluatorAuditError("expected one valid connected molecular graph")
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


@lru_cache(maxsize=8192)
def graph_fingerprint(smiles: str, metric: str) -> Any:
    """Sparse counts avoid fixed-length folding; hash fingerprints are not exact graph identity."""
    mol = Chem.MolFromSmiles(canonical_graph(smiles))
    if metric == "morgan_count_radius2":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=False)
    elif metric == "atom_pair_count":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=1, maxDistance=30, includeChirality=False, use2D=True
        )
    else:
        raise EvaluatorAuditError(f"unknown fingerprint metric: {metric}")
    return generator.GetSparseCountFingerprint(mol)


def fingerprint_distance(left: str, right: str, metric: str) -> float:
    return 1.0 - float(
        DataStructs.TanimotoSimilarity(
            graph_fingerprint(left, metric), graph_fingerprint(right, metric)
        )
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "scientific_question",
        "inputs",
        "expected_attempts",
        "seed",
        "bootstrap_replicates",
        "invariance_pairs",
        "positive_control_pairs",
        "policy",
        "sources",
    }
    if set(config) != required or config["schema_version"] != CONFIG_SCHEMA:
        raise EvaluatorAuditError("evaluator audit config schema or fields changed")
    if set(config["inputs"]) != {
        "assignments",
        "baseline",
        "treatment",
        "previous_review",
        "comparison",
        "baseline_assessment_index",
        "treatment_assessment_index",
        "baseline_common_result",
        "treatment_common_result",
    }:
        raise EvaluatorAuditError("evaluator audit input names changed")
    for field in (
        "expected_attempts",
        "seed",
        "bootstrap_replicates",
        "invariance_pairs",
        "positive_control_pairs",
    ):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvaluatorAuditError(f"{field} must be a positive integer")
    if config["policy"] != {
        "train_only": True,
        "new_model_or_generation_calls": False,
        "modify_frozen_gates": False,
        "automatic_visual_adjudication": False,
        "fingerprints": list(FINGERPRINTS),
        "reference_weighting": "unique components within each precursor role",
        "uncertainty": "paired attempt bootstrap; descriptive within one saved seed",
        "novelty_reference": "measured train and complete train catalogs separately",
    }:
        raise EvaluatorAuditError("evaluator audit policy changed")


def load_train_reference(
    path: Path,
) -> tuple[list[AuditProduct], dict[str, set[str]], dict[str, int]]:
    """Reject nontrain rows before accessing any structure field, including catalog construction."""
    rows: list[AuditProduct] = []
    catalogs = {role: set() for role in UGI_PRECURSOR_ROLES}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for record in iter_csv(path):
        if record["primary_product_fold"] != "train":
            counts["nontrain_rows_skipped_before_structure_access"] += 1
            continue
        counts["train_rows"] += 1
        if any(record[f"{role}_family_fold"] != "train" for role in UGI_PRECURSOR_ROLES):
            raise EvaluatorAuditError("train product has a nontrain role family")
        for role in UGI_PRECURSOR_ROLES:
            catalogs[role].add(record[f"{role}_smiles"])
        if record["is_source_adjudicated_measured_product"].lower() not in {"true", "1"}:
            counts["unmeasured_train_rows_used_only_for_novelty_catalog"] += 1
            continue
        smiles = canonical_graph(record["canonical_product_smiles"])
        if smiles in seen:
            raise EvaluatorAuditError("duplicate measured training constitutional product")
        seen.add(smiles)
        group = "|".join(record[f"{role}_family_id"] for role in UGI_PRECURSOR_ROLES)
        if not all(record[f"{role}_family_id"] for role in UGI_PRECURSOR_ROLES):
            raise EvaluatorAuditError("measured training row lacks its family group")
        rows.append(
            AuditProduct(
                identity=record["product_id"],
                smiles=smiles,
                components=tuple(
                    (role, canonical_graph(record[f"{role}_smiles"]))
                    for role in UGI_PRECURSOR_ROLES
                ),
                source=record["source_stratum"],
                group=group,
                origin="measured_train",
                inherited_novelty=record["component_novelty_class"],
            )
        )
    if not rows or len({row.identity for row in rows}) != len(rows):
        raise EvaluatorAuditError("empty or duplicate measured training identities")
    canonical_catalogs = {
        role: {canonical_graph(smiles) for smiles in values} for role, values in catalogs.items()
    }
    counts["measured_train_products"] = len(rows)
    return (
        sorted(rows, key=lambda row: row.identity),
        canonical_catalogs,
        dict(sorted(counts.items())),
    )


def load_saved_attempts(
    path: Path, arm: str, expected: int
) -> tuple[list[AuditProduct], list[dict]]:
    ledger = iter(iter_jsonl(path))
    header = next(ledger, {})
    if header != {"rows": expected, "schema_version": "forge.common_ugi_assessed_attempts.v1"}:
        raise EvaluatorAuditError(f"{arm}: assessed attempt header changed")
    products = []
    abstentions = []
    seen = set()
    for record in ledger:
        index = record["attempt_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index in seen:
            raise EvaluatorAuditError(f"{arm}: malformed or duplicate attempt index")
        seen.add(index)
        traces = record.get("exact_l1_traces", [])
        if not record.get("valid") or not record.get("exact_l1_program") or len(traces) != 1:
            abstentions.append({"attempt_index": index, "reason": "no_unique_valid_exact_l1_trace"})
            continue
        roles = traces[0]["components_by_role"]
        if set(roles) != set(UGI_PRECURSOR_ROLES):
            raise EvaluatorAuditError(f"{arm}: exact role trace schema changed")
        products.append(
            AuditProduct(
                identity=f"{arm}:{index}",
                smiles=canonical_graph(record["canonical_smiles"]),
                components=tuple(
                    (role, canonical_graph(roles[role])) for role in UGI_PRECURSOR_ROLES
                ),
                source=record["method_id"],
                group=f"saved_attempt:{index}",
                origin=arm,
                attempt_index=index,
                inherited_novelty=(
                    "held_component_exact_l1"
                    if record["held_component_exact_l1"]
                    else "not_held_component_exact_l1"
                ),
            )
        )
    if seen != set(range(expected)):
        raise EvaluatorAuditError(f"{arm}: attempt ledger is incomplete")
    return sorted(products, key=lambda row: row.attempt_index), abstentions


def descriptor_collisions(products: Sequence[AuditProduct]) -> dict[str, Any]:
    """Enumerate every collision class, deduplicating exact constitutions inside each class."""
    buckets: dict[tuple[float, ...], dict[str, list[AuditProduct]]] = defaultdict(dict)
    for row in products:
        signature = tuple(ugi_role_descriptor_vector(row.roles()).tolist())
        buckets[signature].setdefault(row.smiles, []).append(row)
    groups = []
    for signature, by_graph in sorted(buckets.items()):
        if len(by_graph) < 2:
            continue
        molecules = sorted(by_graph)
        exemplar_left, exemplar_right = molecules[:2]
        groups.append(
            {
                "descriptor_sha256": str(sha256_json(signature)),
                "descriptor_values": list(signature),
                "distinct_graphs": len(molecules),
                "members": [
                    {
                        "canonical_smiles": smiles,
                        "observations": [
                            {
                                "identity": row.identity,
                                "origin": row.origin,
                                "source": row.source,
                                "group": row.group,
                                "components_by_role": row.roles(),
                            }
                            for row in by_graph[smiles]
                        ],
                    }
                    for smiles in molecules
                ],
                "deterministic_first_pair_fingerprint_distances": {
                    metric: fingerprint_distance(exemplar_left, exemplar_right, metric)
                    for metric in FINGERPRINTS
                },
            }
        )
    return {
        "observations": len(products),
        "unique_products": len({row.smiles for row in products}),
        "distinct_descriptor_vectors": len(buckets),
        "collision_classes": len(groups),
        "unique_products_in_collision_classes": sum(item["distinct_graphs"] for item in groups),
        "groups": groups,
    }


def paired_summary(differences: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    if not differences:
        return {"status": "abstain_no_eligible_pairs", "paired_attempts": 0}
    values = np.asarray(differences, dtype=np.float64)
    if not np.isfinite(values).all():
        raise EvaluatorAuditError("nonfinite paired differences")
    rng = np.random.default_rng(seed)
    # Stream replicates rather than retaining a replicates-by-attempt allocation.
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(replicates)]
    )
    return {
        "status": "descriptive_only",
        "paired_attempts": len(values),
        "treatment_minus_baseline_mean_distance": float(values.mean()),
        "bootstrap_percentile_95_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        "negative_means_treatment_closer_to_measured_train": True,
    }


def _component_metrics(
    reference: Sequence[AuditProduct],
    arms: Mapping[str, Sequence[AuditProduct]],
    catalogs: Mapping[str, set[str]],
    *,
    seed: int,
    replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    component_sources: dict[str, dict[str, list[dict]]] = {
        role: defaultdict(list) for role in UGI_PRECURSOR_ROLES
    }
    for row in reference:
        for role, smiles in row.components:
            component_sources[role][smiles].append(
                {
                    "product_id": row.identity,
                    "source_stratum": row.source,
                    "family_group": row.group,
                }
            )
    reference_catalog = {role: sorted(items) for role, items in component_sources.items()}
    per_attempt = []
    measured_baselines = {}
    for role, smiles_list in reference_catalog.items():
        measured_baselines[role] = {}
        for metric in FINGERPRINTS:
            leave_self_out = (
                [
                    min(
                        fingerprint_distance(smiles, other, metric)
                        for other in smiles_list
                        if other != smiles
                    )
                    for smiles in smiles_list
                ]
                if len(smiles_list) > 1
                else []
            )
            measured_baselines[role][metric] = {
                "unique_components": len(smiles_list),
                "mean_leave_exact_self_out_nearest_distance": (
                    float(np.mean(leave_self_out)) if leave_self_out else None
                ),
                "status": "descriptive_only" if leave_self_out else "abstain_single_component",
            }
    for arm, products in sorted(arms.items()):
        for row in products:
            for role, smiles in row.components:
                novel_measured = smiles not in component_sources[role]
                novel_train = smiles not in catalogs[role]
                scores = {}
                for metric in FINGERPRINTS:
                    candidates = reference_catalog[role]
                    similarities = DataStructs.BulkTanimotoSimilarity(
                        graph_fingerprint(smiles, metric),
                        [graph_fingerprint(other, metric) for other in candidates],
                    )
                    best = max(similarities)
                    nearest = [
                        candidates[index]
                        for index, score in enumerate(similarities)
                        if abs(score - best) < 1e-12
                    ]
                    scores[metric] = {
                        "distance": 1.0 - float(best),
                        "nearest_measured_components": nearest,
                    }
                per_attempt.append(
                    {
                        "arm": arm,
                        "attempt_index": row.attempt_index,
                        "role": role,
                        "component_smiles": smiles,
                        "product_identity": row.identity,
                        "product_smiles": row.smiles,
                        "source_method": row.source,
                        "novel_vs_measured_train_role_catalog": novel_measured,
                        "novel_vs_all_train_role_catalog": novel_train,
                        "inherited_saved_ledger_novelty_label": row.inherited_novelty,
                        "scores": scores,
                    }
                )
    comparisons = {}
    for role in UGI_PRECURSOR_ROLES:
        by_arm = {
            arm: {
                row["attempt_index"]: row
                for row in per_attempt
                if row["arm"] == arm and row["role"] == role
            }
            for arm in arms
        }
        indices = sorted(set(by_arm["baseline"]) & set(by_arm["treatment"]))
        comparisons[role] = {}
        for stratum, field in (
            ("all_eligible", None),
            ("both_novel_vs_measured_train", "novel_vs_measured_train_role_catalog"),
            ("both_novel_vs_all_train", "novel_vs_all_train_role_catalog"),
        ):
            selected = [
                index
                for index in indices
                if field is None
                or (by_arm["baseline"][index][field] and by_arm["treatment"][index][field])
            ]
            comparisons[role][stratum] = {
                metric: paired_summary(
                    [
                        by_arm["treatment"][index]["scores"][metric]["distance"]
                        - by_arm["baseline"][index]["scores"][metric]["distance"]
                        for index in selected
                    ],
                    seed=seed,
                    replicates=replicates,
                )
                for metric in FINGERPRINTS
            }
        comparisons[role]["per_arm"] = {
            arm: {
                "eligible_attempts": len(rows),
                "unique_components": len({row["component_smiles"] for row in rows.values()}),
                "novel_vs_measured_train_attempts": sum(
                    row["novel_vs_measured_train_role_catalog"] for row in rows.values()
                ),
                "novel_vs_all_train_attempts": sum(
                    row["novel_vs_all_train_role_catalog"] for row in rows.values()
                ),
                "mean_distance": {
                    metric: (
                        float(np.mean([row["scores"][metric]["distance"] for row in rows.values()]))
                        if rows
                        else None
                    )
                    for metric in FINGERPRINTS
                },
            }
            for arm, rows in by_arm.items()
        }
    return {
        "reference_components_with_source_family_provenance": component_sources,
        "leave_exact_self_out_measured_baseline": measured_baselines,
        "comparisons": comparisons,
        "warning": "Nearest-reference distance rewards memorization; novel strata and unique-component "
        "leave-self-out baselines are descriptive, not a calibration of realism.",
        "source_scope": "source_stratum and component-family triples from the assignment ledger; "
        "no unavailable source-study identity is inferred",
    }, per_attempt


def drawing_variant(smiles: str, *, angle_degrees: float, reverse_atoms: bool) -> tuple[str, str]:
    """Change atom ordering and 2D rotation only; return SVG and recovered graph identity."""
    canonical = canonical_graph(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if reverse_atoms:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True, forceRDKit=True)
    conf = mol.GetConformer()
    theta = math.radians(angle_degrees)
    for index in range(mol.GetNumAtoms()):
        point = conf.GetAtomPosition(index)
        conf.SetAtomPosition(
            index,
            Point3D(
                point.x * math.cos(theta) - point.y * math.sin(theta),
                point.x * math.sin(theta) + point.y * math.cos(theta),
                0.0,
            ),
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(700, 280)
    drawer.drawOptions().includeMetadata = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    recovered = canonical_graph(Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False))
    if recovered != canonical:
        raise EvaluatorAuditError("drawing transformation changed constitutional identity")
    return drawer.GetDrawingText(), recovered


def _packet(
    reference: Sequence[AuditProduct],
    treatment: Sequence[AuditProduct],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    rng = random.Random(config["seed"])
    count = config["invariance_pairs"] + config["positive_control_pairs"]
    if len(reference) < count or len(treatment) < config["positive_control_pairs"]:
        raise EvaluatorAuditError("too few measured controls or eligible saved attempts for packet")
    selected = rng.sample(list(reference), count)
    generated = rng.sample(list(treatment), config["positive_control_pairs"])
    key_rows, sheet_rows, cards = [], [], []
    cases = [(measured, measured, True) for measured in selected[: config["invariance_pairs"]]]
    cases.extend(
        (measured, other, False)
        for measured, other in zip(selected[config["invariance_pairs"] :], generated, strict=True)
    )
    rng.shuffle(cases)
    for index, (measured, other, invariant) in enumerate(cases):
        styles = [(0.0, False), (117.0, True)]
        rng.shuffle(styles)
        slots = [(measured, *styles[0]), (other, *styles[1])]
        rng.shuffle(slots)
        pair_id = f"C{index + 1:02d}"
        slot_records, drawings = {}, []
        for label, (row, angle, reverse) in zip(("A", "B"), slots, strict=True):
            svg, recovered = drawing_variant(row.smiles, angle_degrees=angle, reverse_atoms=reverse)
            # Round-trip the exact role structures independently of product depiction.
            roundtrip_roles = {
                role: canonical_graph(
                    Chem.MolToSmiles(
                        Chem.RenumberAtoms(
                            Chem.MolFromSmiles(smiles),
                            list(reversed(range(Chem.MolFromSmiles(smiles).GetNumAtoms()))),
                        ),
                        canonical=False,
                        isomericSmiles=False,
                    )
                )
                for role, smiles in row.components
            }
            descriptor_identical = np.array_equal(
                ugi_role_descriptor_vector(row.roles()), ugi_role_descriptor_vector(roundtrip_roles)
            )
            fp_identical = all(
                fingerprint_distance(row.smiles, recovered, metric) == 0.0
                for metric in FINGERPRINTS
            )
            if not descriptor_identical or not fp_identical:
                raise EvaluatorAuditError("drawing invariance feature check failed")
            slot_records[label] = {
                "identity": row.identity,
                "origin": row.origin,
                "source": row.source,
                "family_group": row.group,
                "canonical_smiles": row.smiles,
                "components_by_role": row.roles(),
                "rotation_degrees": angle,
                "reverse_atom_order": reverse,
                "graph_identity_preserved": recovered == row.smiles,
                "descriptor_identity_preserved": bool(descriptor_identical),
                "fingerprint_identity_preserved": fp_identical,
            }
            drawings.append(f"<div><h3>{pair_id} {label}</h3>{svg[svg.index('<svg'):]} </div>")
        key_rows.append(
            {
                "pair_id": pair_id,
                "kind": "identical_graph_redrawing" if invariant else "measured_positive_control",
                "expected_identity": "same" if measured.smiles == other.smiles else "different",
                "realism_preference_ground_truth": "tie_by_identity" if invariant else None,
                "slots": slot_records,
            }
        )
        sheet_rows.append(
            {
                "pair_id": pair_id,
                "same_constitution": None,
                "overall_plausibility_preference": None,
                "A_contains_concerning_arrangement": None,
                "B_contains_concerning_arrangement": None,
                "confidence": None,
                "reason": "",
            }
        )
        cards.append("<section>" + "".join(drawings) + "</section>")
    sheet = {
        "schema_version": "forge.ugi_realism_invariance_review_sheet.v1",
        "status": "pending",
        "reviewer_id": None,
        "reviews": sheet_rows,
        "instructions": "Assess all pairs before opening the separate blinding key. Score chemical "
        "arrangements; distinguish drawing preference from structural concern. "
        "Use tie or unassessable when warranted. No synthesis or efficacy inference.",
        "allowed_preferences": ["A", "B", "tie", "unassessable"],
        "allowed_identity_labels": ["same", "different", "unassessable"],
    }
    key = {"schema_version": "forge.ugi_realism_invariance_blinding_key.v1", "pairs": key_rows}
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Structure review calibration</title>"
        "<style>body{font-family:sans-serif;margin:24px}section{display:flex;break-inside:avoid;"
        "border-bottom:1px solid #bbb}section>div{width:50%}svg{width:100%;height:auto}</style>"
        "</head><body><h1>Structure review calibration</h1><p>"
        + html.escape(sheet["instructions"])
        + "</p>"
        + "".join(cards)
        + "</body></html>"
    )
    return key, sheet, document


def run_ugi_realism_evaluator_audit(
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the pinned saved-output audit locally and publish a complete diagnostic directory."""
    repo = repo_root.resolve()
    config_file = config_path if config_path.is_absolute() else repo / config_path
    config = read_json_object(config_file)
    _validate_config(config)
    if output_dir.exists():
        raise EvaluatorAuditError(f"output directory already exists: {output_dir}")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    source_paths = {
        name: resolve_pin(pin, repo, label=name) for name, pin in config["sources"].items()
    }
    comparison = read_json_object(paths["comparison"])
    if comparison.get("program_pairing", {}).get("identical_coarse_programs") is not True:
        raise EvaluatorAuditError("saved comparison does not attest paired coarse programs")
    reference, catalogs, counts = load_train_reference(paths["assignments"])
    arms, abstentions = {}, {}
    for arm in ("baseline", "treatment"):
        arms[arm], abstentions[arm] = load_saved_attempts(
            paths[arm], arm, config["expected_attempts"]
        )
        original_arm = "amine_semantic" if arm == "baseline" else "all_role_semantic"
        index = read_json_object(paths[f"{arm}_assessment_index"])
        common = read_json_object(paths[f"{arm}_common_result"])
        links = (
            (comparison["methods"][original_arm]["assessment"], f"{arm}_assessment_index"),
            (index["assessment"]["common_assessment"], f"{arm}_common_result"),
            (common["assessed_attempts"], arm),
        )
        if any(record["sha256"] != config["inputs"][name]["sha256"] for record, name in links):
            raise EvaluatorAuditError(f"{arm}: comparison-to-ledger provenance chain changed")
        if any(row.source != common["method_id"] for row in arms[arm]):
            raise EvaluatorAuditError(f"{arm}: saved method identity changed")
    metrics, scored_rows = _component_metrics(
        reference,
        arms,
        catalogs,
        seed=config["seed"],
        replicates=config["bootstrap_replicates"],
    )
    collision_reports = {
        "measured_train": descriptor_collisions(reference),
        "baseline": descriptor_collisions(arms["baseline"]),
        "treatment": descriptor_collisions(arms["treatment"]),
        "combined": descriptor_collisions([*reference, *arms["baseline"], *arms["treatment"]]),
    }
    key, sheet, packet = _packet(reference, arms["treatment"], config)
    previous = read_json_object(paths["previous_review"])
    notes = [(row["pair_id"], row.get("reviewer_note", "")) for row in previous["reviews"]]
    hypotheses = {
        "status": "uncalibrated_hypotheses_only",
        "previous_reviewer_id": previous["reviewer_id"],
        "saturation_language": [pair for pair, note in notes if "saturat" in note.lower()],
        "silhouette_or_compact_language": [
            pair
            for pair, note in notes
            if "silhouette" in note.lower() or "compact" in note.lower()
        ],
        "interpretation": "Word occurrences motivate calibration; they do not establish bias, "
        "incorrect chemical judgment, or causal effects on gate decisions.",
    }
    source_root = Path(__file__).resolve().parents[2]
    sources = {
        name: pin_record(source_root / name, source_root)
        for name in (
            "forge/model/ugi_realism_evaluator_audit.py",
            "forge/model/ugi_development_realism.py",
            "forge/model/common_lipid_realism.py",
            "forge/core/hashing.py",
            "forge/core/io.py",
            "experiments/phase1/multireaction/ugi_realism_evaluator_audit.py",
        )
    }
    for name, path in source_paths.items():
        if name not in sources or sources[name]["sha256"] != config["sources"][name]["sha256"]:
            raise EvaluatorAuditError(f"executed source differs from declared source: {name}")
    result = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python_version": sys.version, "numpy_version": np.__version__},
        "status": "numerical_complete_reviewer_pending",
        "numerical_complete": True,
        "reviewer_status": "pending",
        "config": pin_record(config_file, repo),
        "inputs": {name: pin_record(path, repo) for name, path in paths.items()},
        "sources": sources,
        "seed": config["seed"],
        "rdkit_version": rdBase.rdkitVersion,
        "definitions": {
            "frozen_descriptor_names": list(ROLE_DESCRIPTOR_NAMES),
            "distance": "1 - RDKit sparse-count Tanimoto similarity; per role nearest unique "
            "source-adjudicated measured train component; equal requested-attempt weight",
            "fingerprints": list(FINGERPRINTS),
            "morgan_radius": 2,
            "atom_pair_distance_range": [1, 30],
            "atom_pair_long_distance_policy": "Paths beyond 30 bonds are absent from this RDKit "
            "fingerprint; complete molecular graphs are retained.",
            "stereochemistry": "removed",
            "fixed_length_folding": False,
            "uncertainty": config["policy"]["uncertainty"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "definitions_fixed_in_config_and_source_before_execution": True,
        },
        "train_reference": counts,
        "reference_selection_sha256": str(
            sha256_json(
                [
                    [row.identity, row.smiles, row.roles(), row.source, row.group]
                    for row in reference
                ]
            )
        ),
        "requested_attempts_per_arm": config["expected_attempts"],
        "abstentions": abstentions,
        "descriptor_collision_summary": {
            name: {field: value for field, value in report.items() if field != "groups"}
            for name, report in collision_reports.items()
        },
        "graph_sensitive_diagnostics": metrics,
        "visual_calibration": {
            "status": "reviewer_pending",
            "invariance_pairs": config["invariance_pairs"],
            "measured_positive_control_pairs": config["positive_control_pairs"],
            "all_machine_identity_checks_pass": True,
            "previous_review_hypotheses": hypotheses,
            "new_review_outcomes": None,
        },
        "calls": {"training": 0, "generation": 0, "synthesis": 0, "oracle": 0, "remote": 0},
        "policy": config["policy"],
        "nonclaims": [
            "No calibration or heldout structure was used, parsed, decomposed or fingerprinted.",
            "Fingerprint proximity measures training resemblance and can reward memorization.",
            "This diagnostic does not replace, relax, validate or promote any frozen realism gate.",
            "Paired intervals describe saved attempts from one seed, not across-training uncertainty.",
            "Novelty strata condition on realized outputs and do not identify a causal treatment effect.",
            "Sparse fingerprints and 36 descriptors can collide; neither proves graph equivalence.",
            "Measured controls establish source evidence, not universal superiority over generated graphs.",
            "The new visual packet has no reviewer outcomes and is not an independent human review.",
            "No candidate selection, synthesis-success or biological efficacy inference is made.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".realism-audit-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / "complete"
        staging.mkdir()
        write_json(staging / "descriptor_collisions.json", collision_reports)
        write_json(
            staging / "component_scores.json",
            {"schema_version": "forge.ugi_realism_component_scores.v1", "rows": scored_rows},
        )
        write_json(staging / "blinding_key.json", key)
        write_json(staging / "review_sheet.json", sheet)
        (staging / "review_packet.html").write_text(packet)
        result["artifacts"] = {
            path.name: artifact_record(path, logical_path=path.name)
            for path in sorted(staging.iterdir())
        }
        write_json(staging / "result.json", result)
        staging.rename(output_dir)
    return result
