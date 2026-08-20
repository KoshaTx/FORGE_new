#!/usr/bin/env python3
"""Build the blinded packet the reviewing chemist uses to choose twelve designs.

The reviewer sees structures, component SMILES and everything about how each design
would be made. She does not see predicted activity, the calibrated lower bound, the
evidence tier or the original candidate identifier, because that identifier is ordered by
rank and would leak the ranking on its own. Codes are assigned in shuffled order under a
recorded seed, and the code that maps back to the panel is written to a separate file that
does not go out with the packet.

The packet is laid out for someone deciding what to make: structures first, then a plain
SMILES table, then the preparation chemistry organised by component rather than by
candidate, because the thirty-four lipids are built from far fewer tails than that and a
single tail batch serves several designs.

Outputs
  results/phase1/ugi_chemist_selection_blind_v1/blind_key.csv   internal, do not send
  results/phase1/ugi_chemist_selection_blind_v1/manifest.json   pinned inputs and seed
  manuscript/chemist_packet/FORGE_candidate_dossier.pdf         send this
  manuscript/chemist_packet/FORGE_candidate_smiles.csv          send this
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = "configs/bio/phase1_ugi_chemist_selection_blind_v1.json"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

ROLE_LABEL = {
    "amine_head": "amine head",
    "oxoester_aldehyde_body_tail": "ester-linked aldehyde tail",
    "isocyanide_tail": "isocyanide tail",
}
ROLE_ORDER = ["amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"]

INK = "#17191D"
MUTED = "#555A62"
RULE = "#C9CDD2"
FAINT = "#EDEFF1"
PDF_DPI = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(smiles: str, px: int, py: int) -> bytes | None:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(px, py)
        options = drawer.drawOptions()
        options.clearBackground = False
        # Bond weight has to track the raster resolution: a fixed 3 px stroke on a
        # 2600 px render placed in a three-inch cell reaches the page at a fifth of
        # a point and the structure looks washed out.
        options.bondLineWidth = max(2, round(px / 380))
        options.multipleBondOffset = 0.14
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()


ESTERIFY = "[CX3:1](=[OX1:2])[OX2H1].[OX2H1][CX4:3]>>[C:1](=[O:2])[OX2][C:3]"
OXIDISE = "[OX2H1][CH2:1]>>[CH1:1]=O"
FORMYLATE = "[NX3;H2:1]>>[N:1]C=O"


def _run(smarts: str, *smiles: str) -> set[str]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    with rdBase.BlockLogs():
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            return set()
        out = set()
        for products in AllChem.ReactionFromSmarts(smarts).RunReactants(tuple(mols)):
            for product in products:
                try:
                    Chem.SanitizeMol(product)
                    out.add(Chem.MolToSmiles(product))
                except Exception:
                    pass
        return out


def route_stages(component: dict, target: str) -> list[tuple[str, str]]:
    """Starting materials, verified intermediate and product for one component.

    The intermediate is derived, then checked by carrying it forward: a hydroxy ester is
    kept only if oxidising it returns the target aldehyde exactly. Anything that fails the
    check is dropped rather than drawn, so no scheme asserts chemistry we have not closed.
    """
    materials = component.get("starting_materials", [])
    route = component.get("route", "")
    if route == "AGILE Tail A" and len(materials) == 2:
        acid = next((m["canonical_smiles"] for m in materials
                     if "acid" in m.get("role", "")), None)
        diol = next((m["canonical_smiles"] for m in materials
                     if "diol" in m.get("role", "")), None)
        if acid and diol:
            good = [e for e in _run(ESTERIFY, acid, diol) if target in _run(OXIDISE, e)]
            stages = [(acid, "carboxylic acid"), (diol, "alpha,omega-diol")]
            if good:
                stages.append((sorted(good)[0], "hydroxy ester"))
            return stages + [(target, "aldehyde tail")]
    if route == "isocyanide from primary amine" and len(materials) == 1:
        amine = materials[0]["canonical_smiles"]
        formamide = sorted(_run(FORMYLATE, amine))
        stages = [(amine, "primary amine")]
        if formamide:
            stages.append((formamide[0], "formamide"))
        return stages + [(target, "isocyanide tail")]
    return [(m["canonical_smiles"], m.get("role", "material")) for m in materials] + \
           [(target, ROLE_LABEL.get(component["role"], "component"))]


SAMPLE_PAGES = {
    "structures": "dossier_structures.png",
    "route": "dossier_synthesis_route.png",
    "materials": "dossier_starting_materials.png",
}


def emit(fig, pdf, dpi: int, sample: str | None = None, sample_dir=None) -> None:
    """Write one page to the packet, and to a PNG when it is one of the pages the
    manuscript reproduces as a representative sample of what the reviewer sees."""
    pdf.savefig(fig, facecolor="white", dpi=dpi)
    if sample and sample_dir is not None:
        fig.savefig(sample_dir / SAMPLE_PAGES[sample], dpi=200, facecolor="white",
                    bbox_inches="tight")


def prepared_count(record: dict) -> int:
    return sum(1 for c in record["route_dossier"] if c["action"] != "purchase")


def new_page(plt, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.972, title, ha="center", va="top", fontsize=15, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=9.6, color=MUTED)
        ax.plot([0.06, 0.94], [0.933, 0.933], color=RULE, lw=1.0)
        return fig, ax, 0.912
    ax.plot([0.06, 0.94], [0.947, 0.947], color=RULE, lw=1.0)
    return fig, ax, 0.925


def place(ax, plt, smiles: str, cx: float, cy: float, half_w: float, half_h: float, px: int):
    from PIL import Image

    png = render(smiles, px, max(1, int(px * half_h / half_w)))
    if png is None:
        return
    image = Image.open(io.BytesIO(png))
    ax.imshow(image, extent=(cx - half_w, cx + half_w, cy - half_h, cy + half_h),
              zorder=3, interpolation="antialiased")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()
    repo = args.repo.resolve()

    config = json.loads((repo / CONFIG).read_text())
    panel_path = repo / PANEL
    panel_hash = sha256(panel_path)

    out_internal = repo / "results/phase1/ugi_chemist_selection_blind_v1"
    out_packet = repo / "manuscript/chemist_packet"
    sample_dir = repo / "manuscript/figures/chemist_packet"
    out_internal.mkdir(parents=True, exist_ok=True)
    out_packet.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in gzip.open(panel_path, "rt")]
    pool = [r for r in records if r["arm"] == config["input"]["arm_included"]]
    if len(pool) != config["input"]["n_candidates"]:
        raise SystemExit(f"expected {config['input']['n_candidates']} candidates, found {len(pool)}")

    seed = config["blinding"]["seed"]
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    prefix = config["blinding"]["code_prefix"]
    blinded = []
    for position, index in enumerate(order, start=1):
        record = dict(pool[index])
        record["blind_code"] = f"{prefix}{position:02d}"
        blinded.append(record)
    blinded.sort(key=lambda r: r["blind_code"])

    # ---- component groupings: the thirty-four lipids share far fewer tails than that
    by_component: dict[str, dict[str, dict]] = {role: {} for role in ROLE_ORDER}
    for record in blinded:
        for component in record["route_dossier"]:
            role = component["role"]
            entry = by_component[role].setdefault(
                component["canonical_smiles"],
                {"component": component, "codes": []})
            entry["codes"].append(record["blind_code"])

    materials: dict[tuple[str, str], dict] = {}
    for record in blinded:
        for component in record["route_dossier"]:
            for material in component.get("starting_materials", []):
                key = (material["canonical_smiles"], material.get("role", "material"))
                entry = materials.setdefault(
                    key, {"vendor_count": material.get("vendor_count", "?"), "uses": 0})
                entry["uses"] += 1

    # ---- internal key, never sent
    with (out_internal / "blind_key.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["blind_code", "candidate_id", "authority_tier", "oracle_mean",
                         "lcb90", "selection_order_within_arm"])
        for record in blinded:
            writer.writerow([record["blind_code"], record["candidate_id"],
                             record["authority_tier"], f"{record['scores']['oracle_mean']:.4f}",
                             f"{record['scores']['lcb90']:.4f}",
                             record["selection_order_within_arm"]])

    # ---- spreadsheet
    smiles_csv = out_packet / "FORGE_candidate_smiles.csv"
    with smiles_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "code", "product_smiles", "molecular_weight",
            "amine_head_smiles", "amine_head_action", "amine_head_suppliers",
            "aldehyde_tail_smiles", "aldehyde_tail_action", "aldehyde_tail_starting_materials",
            "isocyanide_tail_smiles", "isocyanide_tail_action", "isocyanide_tail_starting_materials",
            "components_to_prepare", "steps_before_final_ugi",
            "candidates_sharing_aldehyde_tail", "candidates_sharing_isocyanide_tail",
            "selected", "preference_rank", "reason",
        ])
        for record in blinded:
            dossier = {c["role"]: c for c in record["route_dossier"]}
            row = [record["blind_code"], record["canonical_product"],
                   f"{record['descriptors']['molecular_weight']:.2f}"]
            for role in ROLE_ORDER:
                component = dossier[role]
                mats = "; ".join(
                    f"{m['canonical_smiles']} ({m.get('role', 'material')})"
                    for m in component.get("starting_materials", []))
                if role == "amine_head":
                    row += [component["canonical_smiles"], component["action"],
                            component.get("vendor_count", "")]
                else:
                    row += [component["canonical_smiles"], component["action"], mats]
            shared_ald = len(by_component["oxoester_aldehyde_body_tail"][
                dossier["oxoester_aldehyde_body_tail"]["canonical_smiles"]]["codes"])
            shared_iso = len(by_component["isocyanide_tail"][
                dossier["isocyanide_tail"]["canonical_smiles"]]["codes"])
            row += [prepared_count(record), record["synthetic_steps"],
                    shared_ald, shared_iso, "", "", ""]
            writer.writerow(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": INK})

    pdf_path = out_packet / "FORGE_candidate_dossier.pdf"
    brief = config["brief"]
    n_ald = len(by_component["oxoester_aldehyde_body_tail"])
    n_iso = len(by_component["isocyanide_tail"])

    with PdfPages(pdf_path) as pdf:
        # -------------------------------------------------------- structures
        per_page, columns = 6, 2
        pages = [blinded[i:i + per_page] for i in range(0, len(blinded), per_page)]
        for page_index, chunk in enumerate(pages, start=1):
            fig, ax, top = new_page(
                plt, f"Candidate structures  ({page_index} of {len(pages)})",
                f"{len(pool)} candidates, shuffled order")
            rows = -(-len(chunk) // columns)
            bottom = 0.055
            cell_h = (top - bottom) / max(rows, 1)
            for position, record in enumerate(chunk):
                col = position % columns
                row = position // columns
                cx = 0.06 + (col + 0.5) * (0.88 / columns)
                cy = top - (row + 0.5) * cell_h
                ax.text(cx, cy + 0.44 * cell_h, record["blind_code"], ha="center",
                        va="center", fontsize=14, fontweight="bold")
                place(ax, plt, record["canonical_product"], cx, cy,
                      0.205, 0.29 * cell_h, px=2600)
                prepared = prepared_count(record)
                ax.text(cx, cy - 0.38 * cell_h,
                        f"MW {record['descriptors']['molecular_weight']:.2f}   ·   "
                        f"{record['synthetic_steps']} steps   ·   {prepared} tail"
                        f"{'s' if prepared != 1 else ''} to prepare",
                        ha="center", va="center", fontsize=9.2, color=MUTED)
            emit(fig, pdf, PDF_DPI, "structures" if page_index == 1 else None, sample_dir)
            plt.close(fig)

        # ---------------------------------------------------- SMILES, one page
        fig, ax, y = new_page(plt, "Product SMILES")
        for i, record in enumerate(blinded):
            if i % 2 == 0:
                ax.add_patch(plt.Rectangle((0.045, y - 0.0175), 0.91, 0.0235,
                                           facecolor=FAINT, edgecolor="none", zorder=0))
            ax.text(0.055, y, record["blind_code"], fontsize=8.4, fontweight="bold", va="top")
            ax.text(0.115, y, f"{record['descriptors']['molecular_weight']:.2f}", fontsize=8,
                    color=MUTED, va="top")
            ax.text(0.165, y, record["canonical_product"], fontsize=8,
                    family="monospace", color=INK, va="top")
            y -= 0.0245
        pdf.savefig(fig, facecolor="white", dpi=PDF_DPI)
        plt.close(fig)

        # -------------------------------------- suggested synthesis, per lipid
        per_route_page = 2
        route_pages = [blinded[i:i + per_route_page]
                       for i in range(0, len(blinded), per_route_page)]
        for page_index, chunk in enumerate(route_pages, start=1):
            fig, ax, y = new_page(
                plt, f"Suggested synthesis route  ({page_index} of {len(route_pages)})")
            for record in chunk:
                dossier = {c["role"]: c for c in record["route_dossier"]}
                prepared = prepared_count(record)
                ax.text(0.06, y, record["blind_code"], fontsize=13, fontweight="bold", va="top")
                ax.text(0.145, y + 0.002,
                        f"MW {record['descriptors']['molecular_weight']:.2f}   ·   "
                        f"{record['synthetic_steps']} steps before assembly   ·   "
                        f"{prepared} tail{'s' if prepared != 1 else ''} to prepare",
                        fontsize=9, color=MUTED, va="top")
                y -= 0.028

                for role, arrows in (("oxoester_aldehyde_body_tail",
                                      ("esterify", "oxidise")),
                                     ("isocyanide_tail", ("formylate", "dehydrate"))):
                    component = dossier[role]
                    shared = len(by_component[role][component["canonical_smiles"]]["codes"])
                    tag = f"  (shared with {shared - 1} other candidate" \
                          f"{'s' if shared - 1 != 1 else ''})" if shared > 1 else ""
                    if component["action"] == "purchase":
                        ax.text(0.07, y, f"{ROLE_LABEL[role]} — purchase{tag}",
                                fontsize=9.2, va="top")
                        y -= 0.020
                        place(ax, plt, component["canonical_smiles"], 0.20, y - 0.030,
                              0.115, 0.028, px=1300)
                        y -= 0.072
                        continue
                    ax.text(0.07, y, f"{ROLE_LABEL[role]}{tag}", fontsize=9.2, va="top")
                    y -= 0.018
                    stages = route_stages(component, component["canonical_smiles"])
                    mid = y - 0.036
                    n = len(stages)
                    span = 0.86 / n
                    for i, (stage_smiles, label) in enumerate(stages):
                        cx = 0.07 + (i + 0.5) * span
                        place(ax, plt, stage_smiles, cx, mid, span * 0.42, 0.030, px=1300)
                        ax.text(cx, mid - 0.038, label, ha="center", va="top", fontsize=7.6,
                                color=MUTED)
                        if i:
                            bx = 0.07 + i * span
                            ax.annotate("", xy=(bx + 0.008, mid), xytext=(bx - 0.008, mid),
                                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
                            ax.text(bx, mid + 0.012, arrows[min(i - 1, len(arrows) - 1)],
                                    ha="center", va="bottom", fontsize=7.6, color=INK)
                    y = mid - 0.058

                ax.text(0.07, y, "final assembly — Ugi, amine + aldehyde + isocyanide",
                        fontsize=9.2, va="top")
                y -= 0.018
                place(ax, plt, record["canonical_product"], 0.50, y - 0.040, 0.36, 0.036,
                      px=2400)
                y -= 0.090
                ax.plot([0.06, 0.94], [y, y], color=RULE, lw=0.8)
                y -= 0.026
            emit(fig, pdf, PDF_DPI, "route" if page_index == 1 else None, sample_dir)
            plt.close(fig)

        # ------------------------------------------------- starting materials
        fig, ax, y = new_page(plt, "Starting materials")
        ax.text(0.06, y, f"{len(materials)} catalogue items cover all {len(pool)} candidates. "
                         "Uses counts how many candidates draw on each.",
                fontsize=9.8, color=INK, va="top")
        y -= 0.030
        for (smiles, mrole), entry in sorted(materials.items(), key=lambda kv: -kv[1]["uses"]):
            ax.text(0.065, y, f"{entry['uses']:>2}", fontsize=8.6, family="monospace",
                    color=MUTED, va="top")
            ax.text(0.10, y, smiles, fontsize=8.6, family="monospace", color=INK, va="top")
            ax.text(0.62, y, f"{mrole}, {entry['vendor_count']} suppliers", fontsize=8.6,
                    color=MUTED, va="top")
            y -= 0.0145
        emit(fig, pdf, PDF_DPI, "materials", sample_dir)
        plt.close(fig)

    manifest = {
        "schema_version": "phase1_ugi_chemist_selection_packet.v1",
        "config": CONFIG,
        "config_sha256": sha256(repo / CONFIG),
        "panel": PANEL,
        "panel_sha256": panel_hash,
        "seed": seed,
        "n_candidates": len(pool),
        "n_to_select": brief["n_to_select"],
        "distinct_components": {role: len(by_component[role]) for role in ROLE_ORDER},
        "distinct_starting_materials": len(materials),
        "outputs": {
            "dossier_pdf": str(pdf_path.relative_to(repo)),
            "smiles_csv": str(smiles_csv.relative_to(repo)),
            "blind_key_internal": str((out_internal / "blind_key.csv").relative_to(repo)),
        },
        "do_not_send": [str((out_internal / "blind_key.csv").relative_to(repo))],
        "manuscript_sample_pages": sorted(SAMPLE_PAGES.values()),
    }
    (out_internal / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {pdf_path.relative_to(repo)}")
    print(f"wrote {smiles_csv.relative_to(repo)}")
    print(f"wrote {(out_internal / 'blind_key.csv').relative_to(repo)}  (internal, do not send)")
    print(f"components: {len(by_component['oxoester_aldehyde_body_tail'])} aldehyde tails, "
          f"{len(by_component['isocyanide_tail'])} isocyanide tails, "
          f"{len(materials)} starting materials")


if __name__ == "__main__":
    main()
