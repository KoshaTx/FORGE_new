"""Deterministic molecule-level audit for route-readiness false negatives.

The report deliberately juxtaposes exact, family-projected, partial and unresolved
conservative-high products.  It visualizes the complete product and all three Ugi
components without promoting a planner proposal to route evidence.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

CONFIG_SCHEMA_VERSION = "phase1_ugi_high_potency_molecule_route_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_high_potency_molecule_route_audit_result.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_high_potency_molecule_route_audit_ledger.v1"


class MoleculeAuditError(ValueError):
    """Raised when a molecule-audit input violates its frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MoleculeAuditError(f"{label} must be an object")
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MoleculeAuditError(f"{label} must be a nonempty string")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _require_mapping(json.loads(path.read_text()), label=str(path))
    except OSError as exc:
        raise MoleculeAuditError(f"could not read {path}") from exc
    except json.JSONDecodeError as exc:
        raise MoleculeAuditError(f"{path} is not valid JSON") from exc


def _read_rescoring(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise MoleculeAuditError(f"could not read rescoring ledger {path}") from exc


def _read_readiness(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = _require_mapping(json.loads(line), label=f"readiness row {line_number}")
                except json.JSONDecodeError as exc:
                    raise MoleculeAuditError(
                        f"readiness ledger row {line_number} is not valid JSON"
                    ) from exc
                rows.append(row)
    except OSError as exc:
        raise MoleculeAuditError(f"could not read readiness ledger {path}") from exc
    return rows


def _float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MoleculeAuditError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise MoleculeAuditError(f"{label} must be finite")
    return number


def _integer(value: Any, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MoleculeAuditError(f"{label} must be an integer") from exc
    return number


def _molecule(smiles: str, *, label: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise MoleculeAuditError(f"{label} is not a valid molecule: {smiles}")
    return molecule


def _route_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        _require_nonempty(row.get("arm_id"), label="arm_id"),
        _integer(row.get("draw_index"), label="draw_index"),
        _require_nonempty(row.get("canonical_product"), label="canonical_product"),
    )


def _rank_key(row: dict[str, str]) -> tuple[str, int, str]:
    return (
        _require_nonempty(row.get("arm_id"), label="rescoring arm_id"),
        _integer(row.get("draw_index"), label="rescoring draw_index"),
        _require_nonempty(row.get("canonical_product"), label="rescoring canonical_product"),
    )


def _join_high_candidates(
    rescoring_rows: Iterable[dict[str, str]],
    readiness_rows: Iterable[dict[str, Any]],
    *,
    arm_id: str,
    require_high: bool,
) -> list[dict[str, Any]]:
    ranking: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rescoring_rows:
        if row.get("arm_id") != arm_id or row.get("oracle_scored") != "True":
            continue
        if require_high and row.get("conservative_high_potency") != "True":
            continue
        key = _rank_key(row)
        if key in ranking:
            raise MoleculeAuditError(f"duplicate rescoring key: {key}")
        ranking[key] = row

    joined: list[dict[str, Any]] = []
    seen_route_keys: set[tuple[str, int, str]] = set()
    for route in readiness_rows:
        if route.get("arm_id") != arm_id:
            continue
        key = _route_key(route)
        if key not in ranking:
            continue
        if key in seen_route_keys:
            raise MoleculeAuditError(f"duplicate route-readiness key: {key}")
        seen_route_keys.add(key)
        rank = ranking[key]
        components = route.get("components")
        if not isinstance(components, list) or len(components) != 3:
            raise MoleculeAuditError(f"route row {key} must contain exactly three Ugi components")
        roles = {component.get("role") for component in components if isinstance(component, dict)}
        expected_roles = {"amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"}
        if roles != expected_roles:
            raise MoleculeAuditError(f"route row {key} has malformed component roles: {roles}")
        joined.append(
            {
                **route,
                "potency_utility": _float(rank.get("potency_utility"), label="potency_utility"),
                "oracle_mean": _float(rank.get("oracle_mean"), label="oracle_mean"),
                "oracle_sd": _float(rank.get("oracle_sd"), label="oracle_sd"),
                "lcb90": _float(rank.get("lcb90"), label="lcb90"),
                "calibration_ecdf": _float(rank.get("calibration_ecdf"), label="calibration_ecdf"),
                "authority_tier": rank.get("authority_tier", ""),
                "unseen_roles": rank.get("unseen_roles", ""),
            }
        )

    missing = set(ranking) - seen_route_keys
    if missing:
        example = sorted(missing)[0]
        raise MoleculeAuditError(
            f"{len(missing)} selected rescoring rows lack route-readiness rows; first={example}"
        )

    best_by_product: dict[str, dict[str, Any]] = {}
    for row in joined:
        product = row["canonical_product"]
        incumbent = best_by_product.get(product)
        candidate_key = (-row["potency_utility"], row["draw_index"], product)
        if incumbent is None:
            best_by_product[product] = row
            continue
        incumbent_key = (
            -incumbent["potency_utility"],
            incumbent["draw_index"],
            incumbent["canonical_product"],
        )
        if candidate_key < incumbent_key:
            best_by_product[product] = row
    return list(best_by_product.values())


def _select_diverse(
    rows: list[dict[str, Any]],
    *,
    count: int,
    radius: int,
    fp_size: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise MoleculeAuditError("representatives_per_stratum must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (-row["potency_utility"], row["draw_index"], row["canonical_product"]),
    )
    if len(ordered) <= count:
        return ordered
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fingerprints = {
        row["canonical_product"]: generator.GetFingerprint(
            _molecule(row["canonical_product"], label="canonical_product")
        )
        for row in ordered
    }
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining and len(selected) < count:
        selected_fps = [fingerprints[row["canonical_product"]] for row in selected]

        def diversity_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
            fp = fingerprints[row["canonical_product"]]
            maximum_similarity = max(
                DataStructs.TanimotoSimilarity(fp, selected_fp) for selected_fp in selected_fps
            )
            return (
                1.0 - maximum_similarity,
                row["potency_utility"],
                -row["draw_index"],
                row["canonical_product"],
            )

        chosen = max(remaining, key=diversity_key)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _unresolved_component_diagnostics(
    candidates: list[dict[str, Any]],
    *,
    radius: int,
    fp_size: int,
) -> dict[str, Any]:
    reference_classes = {"exact_complete_current", "family_projected_all_current_leaves"}
    references: dict[str, set[str]] = {}
    unresolved: dict[str, list[str]] = {}
    role_combinations: Counter[tuple[str, ...]] = Counter()
    for row in candidates:
        missing_roles: list[str] = []
        for component in row["components"]:
            role = component["role"]
            readiness = component["route_readiness_class"]
            if readiness in reference_classes:
                references.setdefault(role, set()).add(component["canonical_smiles"])
            if readiness == "unresolved_or_unsupported":
                unresolved.setdefault(role, []).append(component["canonical_smiles"])
                missing_roles.append(role)
        if missing_roles:
            role_combinations[tuple(sorted(missing_roles))] += 1

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    output: dict[str, Any] = {}
    for role, occurrences in sorted(unresolved.items()):
        unique_components = sorted(set(occurrences))
        role_references = sorted(references.get(role, set()))
        if not role_references:
            raise MoleculeAuditError(
                f"unresolved role {role} has no route-ready reference components"
            )
        reference_records = []
        reference_fingerprints = {
            smiles: generator.GetFingerprint(_molecule(smiles, label=f"{role} reference"))
            for smiles in role_references
        }
        similarities: list[float] = []
        for smiles in unique_components:
            molecule = _molecule(smiles, label=f"{role} unresolved component")
            fingerprint = generator.GetFingerprint(molecule)
            heavy_atoms = molecule.GetNumHeavyAtoms()

            def reference_key(reference: str) -> tuple[float, int, str]:
                reference_molecule = _molecule(reference, label=f"{role} reference")
                similarity = DataStructs.TanimotoSimilarity(
                    fingerprint, reference_fingerprints[reference]
                )
                heavy_atom_delta = abs(heavy_atoms - reference_molecule.GetNumHeavyAtoms())
                return similarity, -heavy_atom_delta, reference

            nearest = max(role_references, key=reference_key)
            similarity = DataStructs.TanimotoSimilarity(
                fingerprint, reference_fingerprints[nearest]
            )
            nearest_molecule = _molecule(nearest, label=f"{role} nearest reference")
            similarities.append(similarity)
            reference_records.append(
                {
                    "unresolved_component": smiles,
                    "nearest_route_ready_component": nearest,
                    "morgan_tanimoto": similarity,
                    "unresolved_heavy_atoms": heavy_atoms,
                    "nearest_reference_heavy_atoms": nearest_molecule.GetNumHeavyAtoms(),
                }
            )
        output[role] = {
            "unresolved_occurrences": len(occurrences),
            "unresolved_unique_components": len(unique_components),
            "route_ready_reference_unique_components": len(role_references),
            "median_max_morgan_tanimoto": statistics.median(similarities),
            "minimum_max_morgan_tanimoto": min(similarities),
            "unique_components_at_or_above_0_8": sum(value >= 0.8 for value in similarities),
            "unique_components_at_or_above_0_9": sum(value >= 0.9 for value in similarities),
            "nearest_reference_records": reference_records,
        }
    return {
        "by_role": output,
        "unresolved_product_role_combinations": {
            "+".join(roles): count for roles, count in sorted(role_combinations.items())
        },
    }


def build_audit(
    *,
    config_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise MoleculeAuditError("molecule-audit config schema is unsupported")
    inputs = _require_mapping(config.get("inputs"), label="inputs")
    resolved: dict[str, Path] = {}
    receipts: dict[str, dict[str, str]] = {}
    for label in ("terminal_rescoring", "route_readiness"):
        record = _require_mapping(inputs.get(label), label=f"inputs.{label}")
        relative = Path(_require_nonempty(record.get("path"), label=f"inputs.{label}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise MoleculeAuditError(f"inputs.{label}.path must remain inside the repository")
        path = repo_root / relative
        actual = _sha256_file(path)
        expected = _require_nonempty(record.get("sha256"), label=f"inputs.{label}.sha256")
        if actual != expected:
            raise MoleculeAuditError(f"inputs.{label} SHA-256 mismatch")
        resolved[label] = path
        receipts[label] = {"path": relative.as_posix(), "sha256": actual}

    policy = _require_mapping(config.get("policy"), label="policy")
    arm_id = _require_nonempty(policy.get("arm_id"), label="policy.arm_id")
    require_high = policy.get("require_conservative_high_potency")
    if not isinstance(require_high, bool):
        raise MoleculeAuditError("require_conservative_high_potency must be boolean")
    representatives = _integer(
        policy.get("representatives_per_stratum"), label="representatives_per_stratum"
    )
    radius = _integer(policy.get("morgan_radius"), label="morgan_radius")
    fp_size = _integer(policy.get("morgan_fp_size"), label="morgan_fp_size")
    strata = policy.get("route_strata")
    if not isinstance(strata, list) or not strata or len(strata) != len(set(strata)):
        raise MoleculeAuditError("route_strata must be a nonempty unique list")

    candidates = _join_high_candidates(
        _read_rescoring(resolved["terminal_rescoring"]),
        _read_readiness(resolved["route_readiness"]),
        arm_id=arm_id,
        require_high=require_high,
    )
    counts = Counter(row["route_readiness_class"] for row in candidates)
    unexpected = set(counts) - set(strata)
    if unexpected:
        raise MoleculeAuditError(f"unexpected route-readiness strata: {sorted(unexpected)}")

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for stratum in strata:
        stratum_rows = [row for row in candidates if row["route_readiness_class"] == stratum]
        chosen = _select_diverse(
            stratum_rows,
            count=representatives,
            radius=radius,
            fp_size=fp_size,
        )
        for rank, row in enumerate(chosen, start=1):
            selected.append(
                {
                    **row,
                    "audit_stratum": stratum,
                    "audit_rank_within_stratum": rank,
                    "schema_version": LEDGER_SCHEMA_VERSION,
                }
            )
        selected_counts[stratum] = len(chosen)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "inputs": receipts,
        "policy": policy,
        "summary": {
            "eligible_unique_products": len(candidates),
            "eligible_unique_products_by_route_stratum": {
                stratum: counts.get(stratum, 0) for stratum in strata
            },
            "selected_representatives": len(selected),
            "selected_representatives_by_route_stratum": selected_counts,
            "unresolved_component_diagnostics": _unresolved_component_diagnostics(
                candidates,
                radius=radius,
                fp_size=fp_size,
            ),
        },
        "nonclaims": config.get("nonclaims", []),
    }
    result["result_sha256"] = _content_sha256(result)
    return result, selected


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_audit_artifacts(
    *,
    result: dict[str, Any],
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    result_payload = json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    ledger_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in selected
    )
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
        stream.write(ledger_payload)
    _atomic_write(output_dir / "result.json", result_payload)
    _atomic_write(output_dir / "selected_molecules.jsonl.gz", compressed.getvalue())


def _wrapped_lines(text: str, *, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    return [text[index : index + width] for index in range(0, len(text), width)]


def _molecule_png(smiles: str, *, width: int, height: int) -> bytes:
    molecule = _molecule(smiles, label="rendered molecule")
    rdMolDraw2D.PrepareMolForDrawing(molecule)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.minFontSize = 12
    options.maxFontSize = 22
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_pdf(
    *,
    result: dict[str, Any],
    selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - optional report dependency
        raise MoleculeAuditError(
            "PDF rendering requires the optional report dependencies: pip install -e '.[report]'"
        ) from exc

    colors = {
        "exact_complete_current": HexColor("#B9D8C2"),
        "family_projected_all_current_leaves": HexColor("#C7DCEF"),
        "family_projected_partial_current_leaves": HexColor("#F6D7A7"),
        "unresolved_or_unsupported": HexColor("#E7BBC2"),
    }
    role_names = {
        "amine_head": "Amine-derived head",
        "oxoester_aldehyde_body_tail": "Aldehyde-derived body/tail",
        "isocyanide_tail": "Isocyanide-derived tail",
    }
    role_order = tuple(role_names)
    page_width, page_height = landscape(letter)
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    document.setTitle("FORGE conservative-high molecule and route-readiness audit")

    document.setFillColor(HexColor("#284F7A"))
    document.rect(0, page_height - 92, page_width, 92, fill=1, stroke=0)
    document.setFillColor(white)
    document.setFont("Helvetica-Bold", 22)
    document.drawString(38, page_height - 50, "FORGE molecule-level route-readiness audit")
    document.setFont("Helvetica", 11)
    document.drawString(
        38,
        page_height - 72,
        "Support-enriched conservative-high products; diversity-selected within each route stratum",
    )
    document.setFillColor(black)
    summary = result["summary"]
    document.setFont("Helvetica-Bold", 15)
    document.drawString(38, page_height - 132, "Full candidate-pool census")
    y = page_height - 168
    for stratum, count in summary["eligible_unique_products_by_route_stratum"].items():
        document.setFillColor(colors[stratum])
        document.roundRect(38, y - 8, 18, 18, 4, fill=1, stroke=0)
        document.setFillColor(black)
        document.setFont("Helvetica-Bold", 11)
        document.drawString(68, y, stratum.replace("_", " "))
        document.setFont("Helvetica", 11)
        document.drawRightString(500, y, str(count))
        y -= 34
    document.setFont("Helvetica", 11)
    document.drawString(
        38,
        y - 6,
        f"Selected for visual audit: {summary['selected_representatives']} molecules",
    )
    document.setFont("Helvetica-Bold", 12)
    document.drawString(38, y - 54, "Interpretation")
    document.setFont("Helvetica", 10)
    statements = [
        "Exact and family-projected classes encode current evidence strength, not intrinsic synthetic possibility.",
        "A learned proposal remains a search hypothesis until independently forward-verified and leaf-closed.",
        "The unresolved stratum is inspected specifically for false-negative route calls.",
        "Potency labels are computational prioritization only; they are not prospective biological evidence.",
    ]
    for statement in statements:
        document.drawString(52, y - 78, f"- {statement}")
        y -= 22
    document.showPage()

    for page_index, row in enumerate(selected, start=1):
        stratum = row["audit_stratum"]
        document.setFillColor(colors[stratum])
        document.rect(0, page_height - 56, page_width, 56, fill=1, stroke=0)
        document.setFillColor(black)
        document.setFont("Helvetica-Bold", 15)
        document.drawString(
            30,
            page_height - 35,
            f"{stratum.replace('_', ' ')} - representative {row['audit_rank_within_stratum']}",
        )
        document.setFont("Helvetica", 9)
        document.drawRightString(
            page_width - 30,
            page_height - 35,
            f"utility {row['potency_utility']:.3f} | LCB90 {row['lcb90']:.3f}",
        )

        product_png = _molecule_png(row["canonical_product"], width=1400, height=420)
        document.drawImage(
            ImageReader(io.BytesIO(product_png)),
            36,
            326,
            width=720,
            height=218,
            preserveAspectRatio=True,
            anchor="c",
        )
        document.setFont("Helvetica-Bold", 9)
        document.drawString(36, 314, "Complete generated Ugi product")
        document.setFont("Courier", 6.8)
        product_lines = _wrapped_lines(row["canonical_product"], width=126)
        for line_number, line in enumerate(product_lines[:2]):
            document.drawString(36, 300 - 10 * line_number, line)

        components = {component["role"]: component for component in row["components"]}
        panel_width = 236
        for role_index, role in enumerate(role_order):
            component = components[role]
            x = 30 + role_index * 254
            component_png = _molecule_png(component["canonical_smiles"], width=650, height=340)
            document.setFillColor(HexColor("#F7F8FA"))
            document.roundRect(x, 72, panel_width, 196, 8, fill=1, stroke=0)
            document.drawImage(
                ImageReader(io.BytesIO(component_png)),
                x + 5,
                125,
                width=panel_width - 10,
                height=132,
                preserveAspectRatio=True,
                anchor="c",
            )
            document.setFillColor(black)
            document.setFont("Helvetica-Bold", 9)
            document.drawString(x + 9, 112, role_names[role])
            document.setFont("Helvetica", 7.5)
            document.drawString(
                x + 9,
                99,
                component["route_readiness_class"].replace("_", " ")[:52],
            )
            family = component.get("program_family") or "no qualified family recorded"
            family_lines = _wrapped_lines(family.replace("_", " "), width=48)
            for line_number, line in enumerate(family_lines[:2]):
                document.drawString(x + 9, 87 - 10 * line_number, line)

        unresolved_roles = [
            role_names[component["role"]]
            for component in row["components"]
            if component["route_readiness_class"] != "exact_complete_current"
        ]
        document.setFont("Helvetica", 8)
        document.drawString(
            30,
            52,
            "Non-exact roles: " + (", ".join(unresolved_roles) if unresolved_roles else "none"),
        )
        document.drawString(
            30,
            38,
            f"Novelty pattern: {row.get('pattern_id') or 'none'} | authority: {row.get('authority_tier') or 'none'}",
        )
        document.drawRightString(
            page_width - 30,
            24,
            f"audit page {page_index + 1}/{len(selected) + 1}",
        )
        document.showPage()

    document.save()
    _atomic_write(output_path, buffer.getvalue())


__all__ = [
    "MoleculeAuditError",
    "build_audit",
    "render_pdf",
    "write_audit_artifacts",
]
