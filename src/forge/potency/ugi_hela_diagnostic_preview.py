"""Render a nonselecting HeLa-oracle preview from a frozen generated pool.

This module is deliberately diagnostic.  It does not authorize biological
guidance, alter candidate selection, or assess synthesis.  It restricts the
preview to role-shift patterns supported by the frozen interpolative
conditional-calibration audit and writes the limitations on every card.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import json
import math
import tempfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina

from forge.core.hashing import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_hela_supported_diagnostic_preview_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_hela_supported_diagnostic_preview.v1"
PANEL_SCHEMA_VERSION = "phase1_ugi_hela_supported_diagnostic_preview_panel.v1"

CARD_WARNING = "diagnostic preview; not potency-guided selection; synthesis not assessed"
SELECTED_ORACLE_ID = "supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble"


class UgiHelaDiagnosticPreviewError(ValueError):
    """Raised when the diagnostic preview contract is violated."""


@dataclass(frozen=True)
class Candidate:
    sample_index: int
    product_id: str
    product_smiles: str
    canonical_smiles: str
    exact_unseen_roles: tuple[str, ...]
    pattern_id: str
    scheme: str
    ensemble_mean: float
    ensemble_sd: float
    q90: float
    lcb90: float
    source_row: Mapping[str, str]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiHelaDiagnosticPreviewError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise UgiHelaDiagnosticPreviewError(f"{label} is not a JSON object")
    return payload


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise UgiHelaDiagnosticPreviewError(f"cannot read generated ledger: {path}") from error


def _verified_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UgiHelaDiagnosticPreviewError("preview config inputs are missing")
    output: dict[str, Path] = {}
    for name, record in sorted(inputs.items()):
        if not isinstance(record, Mapping):
            raise UgiHelaDiagnosticPreviewError(f"invalid input record: {name}")
        path = repo / str(record.get("path"))
        expected = str(record.get("sha256"))
        if not path.is_file():
            raise UgiHelaDiagnosticPreviewError(f"missing input: {name}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise UgiHelaDiagnosticPreviewError(
                f"input hash mismatch: {name}: expected {expected}, observed {observed}"
            )
        output[str(name)] = path
    return output


def _pattern_contract(config: Mapping[str, Any]) -> dict[tuple[str, ...], tuple[str, str]]:
    raw = config.get("supported_patterns")
    if not isinstance(raw, list) or not raw:
        raise UgiHelaDiagnosticPreviewError("supported role patterns are missing")
    output: dict[tuple[str, ...], tuple[str, str]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise UgiHelaDiagnosticPreviewError("invalid supported role pattern")
        roles = tuple(str(value) for value in record.get("exact_unseen_roles", []))
        pattern_id = str(record.get("pattern_id"))
        scheme = str(record.get("conformal_scheme"))
        if not roles or not pattern_id or not scheme or roles in output:
            raise UgiHelaDiagnosticPreviewError("ambiguous supported role pattern")
        output[roles] = (pattern_id, scheme)
    return output


def _pattern_q90(
    conformal: Mapping[str, Any],
    schemes: Sequence[str],
) -> dict[str, float]:
    radii = conformal.get("calibration_radii")
    if not isinstance(radii, list):
        raise UgiHelaDiagnosticPreviewError("conformal calibration radii are missing")
    output: dict[str, float] = {}
    for scheme in schemes:
        eligible = [
            float(record["q90"])
            for record in radii
            if isinstance(record, Mapping)
            and record.get("scheme") == scheme
            and record.get("eligible") is True
            and record.get("q90") is not None
        ]
        if not eligible or not all(math.isfinite(value) and value > 0 for value in eligible):
            raise UgiHelaDiagnosticPreviewError(
                f"supported scheme lacks finite eligible q90 radii: {scheme}"
            )
        output[scheme] = max(eligible)
    return output


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiHelaDiagnosticPreviewError(f"invalid product SMILES in frozen ledger: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def eligible_candidates(
    rows: Sequence[Mapping[str, str]],
    *,
    patterns: Mapping[tuple[str, ...], tuple[str, str]],
    q90_by_scheme: Mapping[str, float],
) -> tuple[list[Candidate], dict[str, int]]:
    """Return canonically deduplicated supported interpolative candidates."""

    counts = Counter()
    by_structure: dict[str, Candidate] = {}
    for row in rows:
        counts["all_records"] += 1
        bin_name = row.get("overall_distribution_bin", "")
        counts[f"bin_{bin_name}"] += 1
        try:
            unseen_roles = tuple(json.loads(row.get("exact_unseen_roles_json", "[]")))
        except json.JSONDecodeError as error:
            raise UgiHelaDiagnosticPreviewError("malformed exact-unseen-role JSON") from error
        if bin_name != "interpolative" or unseen_roles not in patterns:
            continue
        pattern_id, scheme = patterns[unseen_roles]
        counts["supported_interpolative_records"] += 1
        counts[f"supported_pattern_{pattern_id}_records"] += 1
        mean = float(row["ensemble_mean_descriptive_only"])
        sd = float(row["ensemble_standard_deviation"])
        q90 = float(q90_by_scheme[scheme])
        if not all(math.isfinite(value) for value in (mean, sd, q90)) or sd < 0:
            raise UgiHelaDiagnosticPreviewError("nonfinite oracle diagnostic in frozen ledger")
        canonical = _canonical(row["product_smiles"])
        candidate = Candidate(
            sample_index=int(row["sample_index"]),
            product_id=row["product_id"],
            product_smiles=row["product_smiles"],
            canonical_smiles=canonical,
            exact_unseen_roles=unseen_roles,
            pattern_id=pattern_id,
            scheme=scheme,
            ensemble_mean=mean,
            ensemble_sd=sd,
            q90=q90,
            lcb90=mean - q90,
            source_row=dict(row),
        )
        incumbent = by_structure.get(canonical)
        if incumbent is None or (candidate.lcb90, -candidate.sample_index) > (
            incumbent.lcb90,
            -incumbent.sample_index,
        ):
            by_structure[canonical] = candidate
    output = sorted(by_structure.values(), key=lambda row: (-row.lcb90, row.sample_index))
    counts["supported_interpolative_unique_products"] = len(output)
    counts["supported_interpolative_duplicate_records"] = counts[
        "supported_interpolative_records"
    ] - len(output)
    return output, dict(sorted(counts.items()))


def _fingerprints(candidates: Sequence[Candidate]) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [
        generator.GetFingerprint(Chem.MolFromSmiles(candidate.canonical_smiles))
        for candidate in candidates
    ]


def _clusters(fingerprints: Sequence[Any], distance_threshold: float) -> list[tuple[int, ...]]:
    if not fingerprints:
        return []
    distances = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index])
        distances.extend(1.0 - value for value in similarities)
    raw = Butina.ClusterData(
        distances,
        len(fingerprints),
        distance_threshold,
        isDistData=True,
        reordering=True,
    )
    return [tuple(int(index) for index in cluster) for cluster in raw]


def select_panel(
    candidates: Sequence[Candidate],
    *,
    top_count: int,
    cluster_distance_threshold: float,
) -> list[dict[str, Any]]:
    """Select high-LCB cluster representatives and structurally matched lower controls."""

    if top_count <= 0:
        raise UgiHelaDiagnosticPreviewError("top_count must be positive")
    fingerprints = _fingerprints(candidates)
    clusters = _clusters(fingerprints, cluster_distance_threshold)
    representatives = []
    for cluster_id, members in enumerate(clusters):
        best = min(
            members, key=lambda index: (-candidates[index].lcb90, candidates[index].sample_index)
        )
        representatives.append((best, cluster_id, members))
    representatives.sort(
        key=lambda record: (-candidates[record[0]].lcb90, candidates[record[0]].sample_index)
    )
    top = representatives[: min(top_count, len(representatives))]
    top_indices = {record[0] for record in top}
    used_controls: set[int] = set()
    rows: list[dict[str, Any]] = []
    for pair_index, (top_index, cluster_id, members) in enumerate(top, start=1):
        lead = candidates[top_index]
        control_options = [
            index
            for index, candidate in enumerate(candidates)
            if index not in top_indices
            and index not in used_controls
            and candidate.pattern_id == lead.pattern_id
            and candidate.lcb90 < lead.lcb90
        ]
        if not control_options:
            raise UgiHelaDiagnosticPreviewError(
                f"no lower-LCB same-pattern control for {lead.product_id}"
            )
        control_index = min(
            control_options,
            key=lambda index: (
                -DataStructs.TanimotoSimilarity(fingerprints[top_index], fingerprints[index]),
                lead.lcb90 - candidates[index].lcb90,
                candidates[index].sample_index,
            ),
        )
        used_controls.add(control_index)
        similarity = float(
            DataStructs.TanimotoSimilarity(fingerprints[top_index], fingerprints[control_index])
        )
        rows.append(
            _panel_row(
                lead,
                selection_class="top_lcb_diverse",
                pair_index=pair_index,
                cluster_id=cluster_id,
                matched_product_id=candidates[control_index].product_id,
                match_similarity=similarity,
            )
        )
        rows.append(
            _panel_row(
                candidates[control_index],
                selection_class="matched_lower_lcb_control",
                pair_index=pair_index,
                cluster_id=cluster_id,
                matched_product_id=lead.product_id,
                match_similarity=similarity,
            )
        )
    return rows


def _panel_row(
    candidate: Candidate,
    *,
    selection_class: str,
    pair_index: int,
    cluster_id: int,
    matched_product_id: str,
    match_similarity: float,
) -> dict[str, Any]:
    return {
        "selection_class": selection_class,
        "pair_index": pair_index,
        "cluster_id": cluster_id,
        "sample_index": candidate.sample_index,
        "product_id": candidate.product_id,
        "product_smiles": candidate.canonical_smiles,
        "exact_unseen_roles_json": json.dumps(candidate.exact_unseen_roles, separators=(",", ":")),
        "supported_pattern": candidate.pattern_id,
        "conformal_scheme": candidate.scheme,
        "overall_distribution_bin": "interpolative",
        "ensemble_mean_descriptive_only": candidate.ensemble_mean,
        "ensemble_standard_deviation": candidate.ensemble_sd,
        "pattern_specific_q90": candidate.q90,
        "descriptive_lcb90": candidate.lcb90,
        "matched_product_id": matched_product_id,
        "matched_product_tanimoto": match_similarity,
        "card_warning": CARD_WARNING,
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
        "synthesis_assessed": False,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise UgiHelaDiagnosticPreviewError("cannot write an empty preview panel")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _molecule_svg(smiles: str, width: int, height: int) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiHelaDiagnosticPreviewError("panel contains an invalid molecule")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.bondLineWidth = 1.8
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    start = svg.find(">", svg.find("<svg")) + 1
    end = svg.rfind("</svg>")
    return svg[start:end]


def _svg_bytes(rows: Sequence[Mapping[str, Any]], columns: int = 3) -> bytes:
    card_width, card_height = 540, 430
    molecule_height = 285
    total_rows = math.ceil(len(rows) / columns)
    width, height = columns * card_width, total_rows * card_height
    chunks = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#f7f9fc'/>",
    ]
    for index, row in enumerate(rows):
        column, grid_row = index % columns, index // columns
        x, y = column * card_width, grid_row * card_height
        accent = "#2c6e9b" if row["selection_class"] == "top_lcb_diverse" else "#8a6a3f"
        chunks.extend(
            [
                f"<rect x='{x + 8}' y='{y + 8}' width='{card_width - 16}' "
                f"height='{card_height - 16}' rx='14' fill='white' stroke='{accent}' "
                "stroke-width='2'/>",
                f"<text x='{x + 24}' y='{y + 33}' font-family='Arial,sans-serif' "
                f"font-size='16' font-weight='700' fill='{accent}'>"
                f"Pair {row['pair_index']} | {html.escape(str(row['selection_class']))}</text>",
                f"<svg x='{x + 20}' y='{y + 42}' width='{card_width - 40}' "
                f"height='{molecule_height}' viewBox='0 0 {card_width - 40} {molecule_height}'>"
                f"{_molecule_svg(str(row['product_smiles']), card_width - 40, molecule_height)}"
                "</svg>",
                f"<text x='{x + 24}' y='{y + 346}' font-family='Arial,sans-serif' "
                "font-size='14' fill='#152238'>"
                f"mean {float(row['ensemble_mean_descriptive_only']):.3f} | "
                f"SD {float(row['ensemble_standard_deviation']):.3f} | "
                f"q90 {float(row['pattern_specific_q90']):.3f} | "
                f"LCB {float(row['descriptive_lcb90']):.3f}</text>",
                f"<text x='{x + 24}' y='{y + 370}' font-family='Arial,sans-serif' "
                "font-size='13' fill='#334a63'>"
                f"{html.escape(str(row['supported_pattern']))} | matched Tanimoto "
                f"{float(row['matched_product_tanimoto']):.3f}</text>",
                f"<text x='{x + 24}' y='{y + 393}' font-family='Arial,sans-serif' "
                "font-size='11' font-weight='700' fill='#9b2c2c'>diagnostic preview; not "
                "potency-guided selection</text>",
                f"<text x='{x + 24}' y='{y + 411}' font-family='Arial,sans-serif' "
                "font-size='11' font-weight='700' fill='#9b2c2c'>synthesis not assessed</text>",
            ]
        )
    chunks.append("</svg>")
    return "".join(chunks).encode()


def _molecule_png(smiles: str, width: int, height: int) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiHelaDiagnosticPreviewError("panel contains an invalid molecule")
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.bondLineWidth = 1.8
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _png_bytes(rows: Sequence[Mapping[str, Any]], columns: int = 3) -> bytes:
    card_width, card_height = 540, 430
    molecule_height = 285
    total_rows = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * card_width, total_rows * card_height), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        column, grid_row = index % columns, index // columns
        x, y = column * card_width, grid_row * card_height
        accent = "#2c6e9b" if row["selection_class"] == "top_lcb_diverse" else "#8a6a3f"
        draw.rounded_rectangle(
            (x + 8, y + 8, x + card_width - 8, y + card_height - 8),
            radius=14,
            fill="white",
            outline=accent,
            width=2,
        )
        draw.text(
            (x + 24, y + 19),
            f"Pair {row['pair_index']} | {row['selection_class']}",
            fill=accent,
            font=font,
        )
        molecule = _molecule_png(str(row["product_smiles"]), card_width - 40, molecule_height)
        canvas.paste(molecule, (x + 20, y + 42))
        draw.text(
            (x + 24, y + 340),
            f"mean {float(row['ensemble_mean_descriptive_only']):.3f} | "
            f"SD {float(row['ensemble_standard_deviation']):.3f} | "
            f"q90 {float(row['pattern_specific_q90']):.3f} | "
            f"LCB {float(row['descriptive_lcb90']):.3f}",
            fill="#152238",
            font=font,
        )
        draw.text(
            (x + 24, y + 362),
            f"{row['supported_pattern']} | matched Tanimoto "
            f"{float(row['matched_product_tanimoto']):.3f}",
            fill="#334a63",
            font=font,
        )
        draw.text(
            (x + 24, y + 386),
            "diagnostic preview; not potency-guided selection",
            fill="#9b2c2c",
            font=font,
        )
        draw.text((x + 24, y + 404), "synthesis not assessed", fill="#9b2c2c", font=font)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _pdf_bytes(png: bytes) -> bytes:
    """Embed the rendered RGB panel in a deterministic one-page PDF.

    Pillow's PDF writer records a wall-clock creation timestamp, which makes an
    otherwise identical scientific artifact hash differently on every replay.
    This tiny writer emits only the five PDF objects required to paint the RGB
    raster and therefore has no mutable metadata.
    """

    image = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = image.size
    compressed = zlib.compress(image.tobytes(), level=9)
    content = f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q\n".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode(),
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n"
        ).encode()
        + compressed
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def _result_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_preview(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path, "preview config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiHelaDiagnosticPreviewError("unsupported preview config schema")
    required_scope = {
        "biological_guidance_authorized": False,
        "candidate_selection_changed": False,
        "synthesis_assessed": False,
        "raw_mean_ranking": False,
        "preview_only": True,
    }
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise UgiHelaDiagnosticPreviewError("preview scope is missing")
    for key, expected in required_scope.items():
        if scope.get(key) != expected:
            raise UgiHelaDiagnosticPreviewError(f"preview scope changed: {key}")

    paths = _verified_inputs(config, repo)
    campaign = _read_json(paths["oracle_campaign_selection"], "campaign selection")
    selected = campaign.get("selected_model")
    if not isinstance(selected, Mapping) or selected.get("candidate_id") != SELECTED_ORACLE_ID:
        raise UgiHelaDiagnosticPreviewError("frozen selected HeLa oracle identity changed")
    if campaign.get("guidance_policy", {}).get("authorized") is not False:
        raise UgiHelaDiagnosticPreviewError("diagnostic requires biological guidance to remain off")

    applicability = _read_json(paths["applicability_result"], "applicability result")
    if applicability.get("adjudication", {}).get("biological_guidance_authorized") is not False:
        raise UgiHelaDiagnosticPreviewError(
            "applicability artifact unexpectedly authorizes guidance"
        )
    conformal = _read_json(paths["conformal_result"], "conformal result")
    if conformal.get("adjudication", {}).get("biological_guidance_authorized") is not False:
        raise UgiHelaDiagnosticPreviewError("conformal artifact unexpectedly authorizes guidance")

    patterns = _pattern_contract(config)
    q90_by_scheme = _pattern_q90(conformal, [scheme for _, scheme in patterns.values()])
    candidates, counts = eligible_candidates(
        _read_gzip_csv(paths["generated_applicability"]),
        patterns=patterns,
        q90_by_scheme=q90_by_scheme,
    )
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise UgiHelaDiagnosticPreviewError("preview selection contract is missing")
    panel = select_panel(
        candidates,
        top_count=int(selection.get("top_count")),
        cluster_distance_threshold=float(selection.get("cluster_distance_threshold")),
    )

    panel_json = json.dumps(
        {"schema_version": PANEL_SCHEMA_VERSION, "rows": panel},
        indent=2,
        sort_keys=True,
    ).encode()
    panel_csv = _csv_bytes(panel)
    panel_svg = _svg_bytes(panel)
    panel_png = _png_bytes(panel)
    panel_pdf = _pdf_bytes(panel_png)
    artifacts = {
        "panel.csv": panel_csv,
        "panel.json": panel_json,
        "panel.svg": panel_svg,
        "panel.png": panel_png,
        "panel.pdf": panel_pdf,
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_diagnostic_preview",
        "scope": dict(scope),
        "card_warning": CARD_WARNING,
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "oracle": {
            "candidate_id": SELECTED_ORACLE_ID,
            "endpoint": "expt_Hela",
            "endpoint_interpretation": "AGILE mTP measured in HeLa cells",
            "mean_role": "descriptive only",
        },
        "eligibility": {
            "overall_distribution_bin": "interpolative",
            "supported_patterns": [
                {
                    "exact_unseen_roles": list(roles),
                    "pattern_id": pattern_id,
                    "conformal_scheme": scheme,
                    "q90": q90_by_scheme[scheme],
                    "q90_reduction": "maximum_over_eligible_folds",
                }
                for roles, (pattern_id, scheme) in patterns.items()
            ],
            "counts": counts,
        },
        "selection": {
            "method": (
                "highest descriptive LCB representative per Morgan cluster, followed by the "
                "highest-LCB clusters; controls are unused same-pattern lower-LCB products with "
                "maximum product-fingerprint Tanimoto similarity"
            ),
            "cluster_distance_threshold": float(selection.get("cluster_distance_threshold")),
            "top_lcb_diverse_count": sum(
                row["selection_class"] == "top_lcb_diverse" for row in panel
            ),
            "matched_lower_lcb_control_count": sum(
                row["selection_class"] == "matched_lower_lcb_control" for row in panel
            ),
            "panel_rows": len(panel),
        },
        "artifacts": {
            name: {"sha256": sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(artifacts.items())
        },
        "interpretation": {
            "supported": (
                "A compact descriptive visualization of supported interpolative candidates "
                "already present in the frozen unguided pool."
            ),
            "not_supported": [
                "biological-guidance authorization",
                "potency-guided candidate selection",
                "experimental potency improvement",
                "synthesis feasibility or route closure",
            ],
        },
    }
    logical = json.loads(json.dumps(result, sort_keys=True))
    result["result_sha256"] = _result_hash(logical)
    artifacts["result.json"] = json.dumps(result, indent=2, sort_keys=True).encode()
    return result, artifacts


def write_preview(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise UgiHelaDiagnosticPreviewError(f"preview output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as raw:
        temporary = Path(raw)
        for name, content in artifacts.items():
            (temporary / name).write_bytes(content)
        temporary.rename(output_dir)


__all__ = [
    "CARD_WARNING",
    "CONFIG_SCHEMA_VERSION",
    "Candidate",
    "UgiHelaDiagnosticPreviewError",
    "build_preview",
    "eligible_candidates",
    "select_panel",
    "write_preview",
]
