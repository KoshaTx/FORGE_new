"""Verify scoped development evidence, pending final studies and portable PDF receipts."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def verify_pin_tree(value, verified):
    """Verify nested source receipts without loading large corpora into memory."""
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            path = ROOT / value["path"]
            assert path.is_file(), str(path)
            assert re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), value
            if value["path"] not in verified:
                verified[value["path"]] = digest(path)
            assert verified[value["path"]] == value["sha256"], value["path"]
        else:
            for child in value.values():
                verify_pin_tree(child, verified)
    elif isinstance(value, list):
        for child in value:
            verify_pin_tree(child, verified)


def verify_reaction_figures(ledger, sections, manuscript, export_root):
    """Link all-family drawings, source-event checks and portable figure bytes."""
    render_path = HERE / "figures/all_family_reactions/render.json"
    render = json.loads(render_path.read_text())
    if "reaction_figure_contract" in ledger:
        assert ledger["reaction_figure_contract"] == pin(render_path)
    archived_ledger = json.loads(
        (HERE / "archive/before_measured_results_20260925/RESULTS_LEDGER.json").read_text()
    )
    assert ledger["reaction_figure_contract"] == archived_ledger["reaction_figure_contract"]
    verified = {}
    verify_pin_tree(render["inputs"], verified)
    records_pin = next(
        p for p in render["inputs"] if Path(p["path"]).name == "reaction_records.json"
    )
    review_pin = next(
        p for p in render["inputs"] if Path(p["path"]).name == "chemistry_review.json"
    )
    records_path, review_path = ROOT / records_pin["path"], ROOT / review_pin["path"]
    records = json.loads(records_path.read_text())
    review = json.loads(review_path.read_text())
    verification_path = records_path.with_name("verification.json")
    verification = json.loads(verification_path.read_text())
    for document in (records, review, verification):
        verify_pin_tree(document, verified)
    assert verification["output"] == records_pin
    assert verification["families"] == 22 and verification["events"] == 34
    for key in (
        "all_22_qualified_families_present",
        "all_exact_source_roundtrips",
        "all_events_exact_unsaturated",
        "all_final_steps_equal_source_products",
    ):
        assert verification[key] is True, key
    assert verification["heldout_structures_used"] is False
    assert verification["model_calls"] == 0
    final_validation = verification["final_validation"]
    for key in ("black_check", "ruff_check", "independent_repeat_identical_record_bytes"):
        assert final_validation[key] is True, key
    for key in ("record_sha256_before_repeat", "record_sha256_after_repeat"):
        assert final_validation[key] == records_pin["sha256"]
    by_family = {r["family_id"]: r for r in records["records"]}
    assert len(by_family) == len(records["records"]) == 22
    assert set(by_family) == set(ledger["family_ids"])
    assert sum(len(r["steps"]) for r in by_family.values()) == 34
    for record in by_family.values():
        assert record["complete_source_roundtrip"] is True
        assert record["steps"][-1]["product_smiles"] == record["product_smiles"]
        for step in record["steps"]:
            assert step["forward_exact"] is True and step["forward_saturated"] is False

    assert review["status"].startswith("passed_")
    checked = review["record_review"]
    assert checked["inputs"]["reaction_records"] == records_pin
    assert checked["families_checked"] == 22 and checked["events_checked"] == 34
    reviewed = {r["family_id"]: r for r in checked["families"]}
    balance_unassessed = []
    assert len(reviewed) == len(checked["families"]) == 22
    assert set(reviewed) == set(by_family) == set(review["families"])
    for family, source in by_family.items():
        assessed = reviewed[family]
        assert assessed["source_target_id"] == source["source_target_id"]
        assert assessed["ordered_trace_continuity_pass"] is True
        assert len(assessed["steps"]) == len(source["steps"])
        for step, original in zip(assessed["steps"], source["steps"], strict=True):
            for field in ("reaction_id", "stage_index", "step_index"):
                assert step[field] == original[field]
            assert step["exact_product_reproduced"] is True
            if step["registered_net_balance_matches"] is None:
                registry = json.loads((ROOT / original["registry_pin"]["path"]).read_text())
                registered = next(
                    r for r in registry["reactions"] if r["reaction_id"] == original["reaction_id"]
                )
                assert registered.get("net_byproducts") is None
                assert original["net_byproducts"] is None
                balance_unassessed.append(
                    {
                        "family": family,
                        "reaction_id": original["reaction_id"],
                        "reason": "No net-byproduct balance contract in the pinned historical registry",
                        "state": "unassessed, not passed",
                    }
                )
            else:
                assert step["registered_net_balance_matches"] is True
            assert step["search_cap_reached"] is False

    existing = render["existing_figure_families"]
    assert len(existing) == len(set(existing)) == 3
    assert set(existing) == {"aldehyde_ugi3", "aza_michael_acrylate", "reductive_amination"}
    families = [family for figure in render["figures"] for family in figure["families"]]
    companion_ids = [f["family"] for f in families]
    assert len(companion_ids) == len(set(companion_ids)) == 19
    assert not (set(companion_ids) & set(existing))
    assert set(companion_ids) | set(existing) == set(by_family)
    assert r"\input{sections/reaction_figures_22.tex}" in manuscript
    assert r"\input{figures/forge_reaction_schemes/schemes_body.tex}" in manuscript
    assert r"\label{fig:reaction-schemes}" in manuscript
    scheme = HERE / "figures/forge_reaction_schemes/schemes_body.tex"
    assert len(re.findall(r"^\\rxnrow\{", scheme.read_text(), re.M)) == 3
    reaction_text = sections["reaction_figures_22.tex"]
    includes = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", reaction_text)
    expected_includes = [
        str((ROOT / figure["pdf"]["path"]).relative_to(HERE)) for figure in render["figures"]
    ]
    assert Counter(includes) == Counter(expected_includes)
    assert len(expected_includes) == len(set(expected_includes))
    figure_files = []
    for figure in render["figures"]:
        assert figure["families"]
        for extension in ("svg", "pdf"):
            source_pin = figure[extension]
            verify_pin_tree(source_pin, verified)
            source = ROOT / source_pin["path"]
            assert source.suffix == "." + extension
            if export_root is not None:
                exported = export_root / source.relative_to(HERE)
                assert digest(exported) == source_pin["sha256"], str(exported)
            figure_files.append(source_pin)
    if export_root is not None:
        assert digest(export_root / scheme.relative_to(HERE)) == digest(scheme)

    audits = defaultdict(list)
    required_checks = {
        "exact_registry_replay",
        "source_inputs_unchanged",
        "ordinary_atoms_and_bonds_only",
        "retained_atoms_hydrogens_charges_exact",
        "retained_bonds_exact",
        "all_hidden_fragments_identical",
        "all_hidden_attachments_identical",
        "all_heteroatoms_and_rings_retained",
        "all_matched_subsequent_reaction_sites_retained",
        "all_molecule_formal_charges_exact",
    }
    for audit in render["schematic_audits"]:
        family = audit["family"]
        assert family in companion_ids
        assert required_checks <= audit["checks"].keys()
        assert all(value is True for value in audit["checks"].values())
        verify_pin_tree(audit["registry_pin"], verified)
        assert audit["actual_formal_charges"] == audit["shown_formal_charges"]
        assert all(
            group["identical_atom_hydrogen_charge_bond_state"] is True
            for group in audit["r_groups"]
        )
        assert any(
            step["reaction_id"] == audit["reaction_id"]
            and step["registry_pin"] == audit["registry_pin"]
            and step["product_smiles"] == audit["actual_product_smiles"]
            and [r["smiles"] for r in step["reactants"]] == audit["actual_reactant_smiles"]
            for step in by_family[family]["steps"]
        ), (family, audit["displayed_step"])
        audits[family].append(audit)
    assert set(audits) == set(companion_ids)
    for item in families:
        family, group = item["family"], audits[item["family"]]
        source = by_family[family]
        assert item["source_target_id"] == source["source_target_id"]
        assert item["registered_events"] == len(source["steps"])
        assert item["displayed_transformations"] == len(group)
        assert all(
            type(a["represented_events"]) is int and a["represented_events"] > 0 for a in group
        )
        # Displayed steps retain original event indices when identical events are grouped.
        next_event = 1
        for audit in sorted(group, key=lambda a: a["displayed_step"]):
            assert audit["displayed_step"] == next_event
            next_event += audit["represented_events"]
        assert sum(a["represented_events"] for a in group) == len(source["steps"])
    return {
        "render": pin(render_path),
        "source_records": records_pin,
        "source_verification": pin(verification_path),
        "chemistry_review": review_pin,
        "existing_source_scheme": review["existing_scheme"],
        "active_existing_scheme": pin(scheme),
        "family_ids": sorted(by_family),
        "source_records_count": len(by_family),
        "source_events": 34,
        "existing_families": existing,
        "companion_families": companion_ids,
        "companion_figure_groups": len(render["figures"]),
        "displayed_schematic_audits": len(render["schematic_audits"]),
        "represented_companion_events": sum(
            a["represented_events"] for a in render["schematic_audits"]
        ),
        "all_schematic_checks_pass": True,
        "schematic_check_scope": sorted(required_checks),
        "broader_unsaturation_endpoint_retention": "Not assessed by the frozen schematic receipt; no new qualification claimed",
        "registered_net_balance_unassessed": balance_unassessed,
        "registered_net_balance_unassessed_is_not_pass": True,
        "historical_chemistry_contract_unchanged": True,
        "source_and_exported_svg_pdf_pins_verified": figure_files,
        "verified_dependency_pins": [
            {"path": path, "sha256": sha} for path, sha in sorted(verified.items())
        ],
        "renderer_visual_status": render.get("visual_review"),
        "visual_review_scope": "Separate rendered-page review receipt; chemical-record review and hash checks do not establish visual legibility.",
    }


def json_value(document, pointer):
    for key in pointer:
        document = document[key]
    return document


def table_cells(table, label):
    rows = [line for line in table.splitlines() if line.startswith(label + " & ")]
    assert len(rows) == 1, label
    return [cell.strip() for cell in rows[0].split(" & ")]


def verify_populated_tables(tables, ledger, quality_evidence, family_labels, quality_labels):
    """Bind displayed family values to the admitted development census."""

    def table(label):
        return next(t for t in tables if r"\label{" + label + "}" in t)

    summary = json.loads((HERE / "current_summary.json").read_text())
    appendix = json.loads((HERE / "current_appendix_evidence.json").read_text())
    routes = json.loads((ROOT / appendix["inputs"]["route_treatment"]["path"]).read_text())
    main = {row["family"]: row for row in summary["family_rows"]}
    for family, label, short in zip(
        ledger["family_ids"], family_labels, quality_labels, strict=True
    ):
        l1 = table_cells(table("tab:22-l1"), label)
        assert len(l1) == 7, family
        assert l1[3] == f'{100 * main[family]["route"]["exact_L1"] / 64:.1f}', family
        terminal = summary["same_cohort_terminal_argmax"]["draw0"][family]
        assert terminal["trajectories"] == 64
        assert l1[2] == f'{100 * terminal["exact_L1"] / 64:.1f}', family
        assert r"\PendingCell" in l1[4], family
        for column, arm in ((1, "conditioned_true_endpoint"), (5, "cyclic_true_endpoint")):
            raw = summary["same_cohort_readouts"][arm]["draw0"][family]
            assert raw["trajectories"] == 64
            assert l1[column] == f'{100*raw["exact_L1"]/64:.1f}', (family, arm)
        decomp = main[family]["decomposition"]
        expected_decomp = f'{100 * decomp["candidate_search_coverage"]["fraction"]:.1f}/{100 * decomp["checked_candidate_acceptance"]["fraction"]:.1f}'
        assert l1[6].removesuffix(r"\\").replace(" ", "") == expected_decomp, family
        route_cells = table_cells(table("tab:22-routes"), label)
        fields = (
            "requests",
            "exact_L1",
            "L2_ready_products",
            "L3_direct_only_products",
            "combined_primary_products",
            "strict_secondary_products",
        )
        for cell, field in zip(route_cells[1:], fields, strict=True):
            assert int(cell.removesuffix(r"\\").strip()) == routes["by_family"][family][field]
        quality = quality_evidence["by_family"][family]
        structure = table_cells(table("tab:22-structure"), short)
        counts = (
            quality["novelty"]["connected"],
            quality["quality"]["exact"],
            quality["quality"]["design_pass"],
            quality["quality"]["axes"]["chemical"].get("fail", 0),
            quality["quality"]["context"],
            quality["quality"]["design_and_no_context"],
        )
        for cell, count in zip(structure[1:], counts, strict=True):
            assert cell.removesuffix(r"\\").strip() == f"{count}/64", family
        conditions = table_cells(table("tab:22-conditions"), short)
        q = quality["structural_statistics"]
        allocation = q["conditions"]["role_cycle_allocation"]
        size = q["conditions"]["ring_size"]
        support = q["ring_TRAIN_support"]
        assert conditions[2] == f'{allocation.get("pass",0)}/64'
        assert conditions[3] == f'{size.get("pass",0)}/{size.get("pass",0)+size.get("fail",0)}'
        assert conditions[4] == f'{100*q["atom_environment_coverage"]["mean"]:.3f}'
        assert (
            conditions[5]
            == f'{support.get("supported",0)}/{support.get("supported",0)+support.get("unknown",0)}'
        )

        geometry = next(
            r
            for r in quality_evidence["synthetic_CAL"]["descriptor_statistics_and_neighborhoods"]
            if r["family"] == family
        )
        descriptor_panel = next(
            t for t in tables if "Descriptor-space synthetic-CAL development comparison." in t
        )
        descriptor_cells = table_cells(descriptor_panel, short)
        expected = []
        for arm in ("context_preserving", "saved", "TRAIN_control"):
            estimate = geometry["arms"][arm]["descriptor_neighborhood"]
            expected.extend(f"{100*estimate[k]:.3f}" for k in ("precision", "coverage"))
        assert descriptor_cells[1:7] == expected
        assert descriptor_cells[7].removesuffix(r"\\").strip() == str(
            geometry["arms"]["reference"]["counts"]["requests"]
        )
    # Macro, pooled and worst-family values must use their different denominators.
    decomposition = [r["decomposition"] for r in main.values()]
    terminal_rows = list(summary["same_cohort_terminal_argmax"]["draw0"].values())
    for label in ("Equal-family macro", "Pooled all requests", "Worst-family result"):
        cells = table_cells(table("tab:22-l1"), label)
        if label == "Equal-family macro":
            coverage = sum(r["candidate_search_coverage"]["fraction"] for r in decomposition) / 22
            acceptance = (
                sum(r["checked_candidate_acceptance"]["fraction"] for r in decomposition) / 22
            )
        elif label == "Pooled all requests":
            coverage = sum(r["has_returned_candidate"] for r in decomposition) / 1408
            acceptance = sum(r["passing_candidates"] for r in decomposition) / sum(
                r["checked_candidates"] for r in decomposition
            )
        else:
            coverage = min(r["candidate_search_coverage"]["fraction"] for r in decomposition)
            acceptance = min(r["checked_candidate_acceptance"]["fraction"] for r in decomposition)
        assert (
            cells[6].removesuffix(r"\\").replace(" ", "")
            == f"{100*coverage:.1f}/{100*acceptance:.1f}"
        )
        term = (
            min(r["exact_L1"] / r["trajectories"] for r in terminal_rows)
            if label == "Worst-family result"
            else 379 / 1408
        )
        assert cells[2] == f"{100*term:.1f}"
        assert r"\PendingCell" in cells[4]
        for column, arm in ((1, "conditioned_true_endpoint"), (5, "cyclic_true_endpoint")):
            group = summary["same_cohort_readouts"][arm]["draw0"]
            value = (
                min(v["exact_L1"] / v["trajectories"] for k, v in group.items() if k != "ALL")
                if label == "Worst-family result"
                else group["ALL"]["exact_L1"] / 1408
            )
            assert cells[column] == f"{100*value:.1f}"
    assert "candidate acceptance" in table("tab:22-l1")
    # These explicitly scoped rows must not silently acquire an all-request denominator.
    quality_main = table("tab:22-quality-metrics")
    for label, count, percentage in (
        ("Role-local cycle allocation / requests", "1,404/1,408", "99.7"),
        ("Requested ring-size agreement / applicable requests", "720/789", "91.3"),
        ("Mean per-product TRAIN atom-environment coverage", "1,408 products", "98.5"),
        ("Ring-system TRAIN support / ring-containing products", "589/810", "72.7"),
    ):
        cells = table_cells(quality_main, label)
        assert cells[1] == count and cells[2].removesuffix(r"\\").strip() == percentage
    readout = quality_evidence["readout_quality"]
    assert summary["same_cohort_draw0_quality"] == readout["summaries"]
    readout_rows = readout["families"]
    old_distribution = quality_evidence["synthetic_CAL"]["rows"]
    geometry_rows = quality_evidence["synthetic_CAL"]["descriptor_statistics_and_neighborhoods"]
    display_arms = ("conditioned_true_endpoint", "conditioned_terminal_argmax", "selected")
    for label, key, multiplier, digits in (
        ("Fingerprint precision (\\%)", "fingerprint_precision_among_unique", 100, 1),
        ("Fingerprint coverage (\\%)", "fingerprint_coverage", 100, 1),
        ("Fingerprint recall (\\%)", "fingerprint_recall", 100, 1),
        (
            "Distinct fingerprint-supported products / requests (\\%)",
            "distinct_in_reference_manifold_per_request",
            100,
            1,
        ),
        ("Nearest-reference Tanimoto similarity", "nearest_reference_tanimoto_mean", 1, 3),
        ("Within-family ECFP4 diversity", "mean_pairwise_tanimoto_distance", 1, 3),
    ):
        actual = table_cells(table("tab:22-realism-metrics"), label)[1:]
        values = [
            sum(r["arms"][arm]["fingerprints"][key] for r in readout_rows) / 22
            for arm in display_arms
        ]
        values.append(
            sum(r["arms"]["TRAIN_control"]["fingerprints"][key] for r in old_distribution) / 22
        )
        assert [x.removesuffix(r"\\").strip() for x in actual] == [
            f"{multiplier*x:.{digits}f}" for x in values
        ], label
    connected = table_cells(table("tab:22-realism-metrics"), "Connected / all requests (\\%)")[1:]
    assert [x.removesuffix(r"\\").strip() for x in connected] == [
        f'{100*readout["summaries"][arm]["connected"]/1408:.1f}' for arm in display_arms
    ] + ["100.0"]
    for metric, label in (
        ("precision", "Synthetic-CAL descriptor precision (\\%)"),
        ("coverage", "Synthetic-CAL descriptor coverage (\\%)"),
        ("member_observations_per_request", "Descriptor-supported observations / requests (\\%)"),
    ):
        cells = table_cells(table("tab:22-realism-metrics"), label)
        values = [
            sum(r["arms"][arm]["descriptor_neighborhood"][metric] for r in readout_rows) / 22
            for arm in display_arms
        ]
        train_metric = "precision" if metric == "member_observations_per_request" else metric
        values.append(
            sum(
                r["arms"]["TRAIN_control"]["descriptor_neighborhood"][train_metric]
                for r in geometry_rows
            )
            / 22
        )
        assert [c.removesuffix(r"\\").strip() for c in cells[1:]] == [
            f"{100*x:.1f}" for x in values
        ]
    panel = next(t for t in tables if "Same-request first-draw readout distributions" in t)
    for family, label in zip(ledger["family_ids"], quality_labels, strict=True):
        source = next(r for r in readout_rows if r["family"] == family)
        marker = (
            r"\multicolumn{6}{l}{\textbf{"
            + label
            + "}; reference observations "
            + str(source["reference_counts"]["connected"])
            + "}"
        )
        assert panel.count(marker) == 1
        group = panel.split(marker, 1)[1].split(r"\multicolumn{6}", 1)[0]
        for arm, arm_label in (
            ("conditioned_true_endpoint", "Conditioned raw"),
            ("conditioned_terminal_argmax", "Conditioned terminal"),
            ("cyclic_true_endpoint", "Cyclic raw"),
            ("cyclic_terminal_argmax", "Cyclic terminal"),
        ):
            a = source["arms"][arm]
            c = a["counts"]
            fp = a["fingerprints"]
            d = a["descriptor_neighborhood"]
            actual = table_cells(group, arm_label)[1:]
            expected = [
                f'{c["valid"]}/{c["unique_connected"]}',
                f'{100*fp["fingerprint_precision_among_unique"]:.1f}/{100*fp["fingerprint_coverage"]:.1f}',
                f'{100*d["precision"]:.1f}/{100*d["coverage"]:.1f}',
                f'{fp["unique_member_count"]}/64',
                f'{d["member_observations"]}/64',
            ]
            assert [x.removesuffix(r"\\").strip() for x in actual] == expected
    for label in (
        "Independent descriptor precision / coverage",
        "Independent source-grouped classifier AUC",
    ):
        assert all(
            r"\PendingCell" in cell
            for cell in table_cells(table("tab:22-realism-metrics"), label)[1:]
        )


def verify_table_completion(documents, verified):
    """Recount saved molecule/check ledgers without rerunning a scientific evaluator."""
    assembly = documents["assembly"]
    terminal = documents["terminal"]
    quality = documents["quality_statistics"]
    for source in (assembly, terminal, quality):
        assert source["complete"] is True
        verify_pin_tree(source["protocol"], verified)
    assert terminal["mode"] == "terminal" and terminal["model_flow_calls"] == 0
    assert terminal["requests"] == 1408 and terminal["trajectories"] == 7040
    assert len(terminal["shards"]) == 880
    verify_pin_tree(terminal["smoke"], verified)
    smoke = json.loads((ROOT / terminal["smoke"]["path"]).read_text())
    assert all(
        smoke[k] is True
        for k in ("passed", "conditioned_wrapper_equality", "original_prediction_equality")
    )
    tally = {"draw0": defaultdict(Counter), "all5": defaultdict(Counter)}
    identities = set()
    routes = {r["index"]: r for r in documents["routes"]["rows"]}
    for expected in terminal["shards"]:
        verify_pin_tree(expected, verified)
        shard = json.loads((ROOT / expected["path"]).read_text())
        for row in shard["rows"]:
            key = (row["draw"], row["index"])
            assert key not in identities and row["draw"] in range(5)
            identities.add(key)
            assert row["family"] == routes[row["index"]]["family"]
            arm = row["arms"]["terminal_argmax_from_saved"]
            if arm["check"]["exact"]:
                assert arm["smiles"] is not None
            for panel in (("draw0", "all5") if row["draw"] == 0 else ("all5",)):
                for group in ("ALL", row["family"]):
                    tally[panel][group].update(
                        trajectories=1,
                        exact_L1=int(arm["check"]["exact"]),
                        valid_connected=int(arm["smiles"] is not None),
                    )
    assert identities == {(draw, index) for draw in range(5) for index in range(1408)}
    assert {
        panel: {family: dict(counts) for family, counts in groups.items()}
        for panel, groups in tally.items()
    } == terminal["summaries"]["terminal_argmax_from_saved"]
    assert tally["draw0"]["ALL"] == dict(trajectories=1408, exact_L1=379, valid_connected=996)
    assert tally["all5"]["ALL"] == dict(trajectories=7040, exact_L1=1910, valid_connected=4870)
    assert len(assembly["requests"]) == 1408
    assert {r["index"] for r in assembly["requests"]} == set(routes)
    sums = Counter()
    for row in assembly["requests"]:
        selected = routes[row["index"]]
        assert (
            row["family"] == selected["family"] and row["selected_ordinal"] == selected["ordinal"]
        )
        assert (
            row["selected_smiles_sha256"] == hashlib.sha256(selected["smiles"].encode()).hexdigest()
        )
        assert row["exact"] == selected["exact_L1"]
        for key in (
            "checked_candidates",
            "passing_candidates",
            "returned_inverse_candidates",
            "has_returned_candidate",
            "exact",
            "ambiguous_accepted_component_tuples",
        ):
            sums[key] += row[key]
    assert all(sums[k] == assembly["totals"][k] for k in sums)
    assert sums["has_returned_candidate"] == 1392 and sums["checked_candidates"] == 1456
    assert sums["passing_candidates"] == 1387 and sums["ambiguous_accepted_component_tuples"] == 0
    q = quality["quality"]["totals"]
    assert q["requests"] == 1408
    assert q["atom_observations"] == 78532 and q["unknown_atom_observations"] == 1259
    assert q["ring_TRAIN_support"] == dict(supported=589, unknown=221, not_applicable=598)
    assert q["conditions"]["role_cycle_allocation"] == {"pass": 1404, "fail": 4}
    assert q["conditions"]["ring_size"] == {"pass": 720, "fail": 69, "not_applicable": 619}
    assert len(quality["descriptors"]) == 22
    for family in quality["descriptors"]:
        for arm in ("saved", "context_preserving", "TRAIN_control"):
            group = family["arms"][arm]
            assert group["counts"]["requests"] == 64
            assert len(group["descriptors"]) == 24
            estimate = group["descriptor_neighborhood"]
            assert estimate["generated_n"] == len(estimate["members"]) == 64
            assert estimate["member_observations"] == sum(estimate["members"])
            assert estimate["precision"] == sum(estimate["members"]) / 64
            assert estimate["covered_reference"] == sum(estimate["reference_covered"])
            assert estimate["reference_n"] == len(estimate["reference_covered"])
    readouts = documents["readout_quality"]
    assert readouts["complete"] and readouts["selected_frozen_estimator_reproduction_exact"]
    assert not readouts["independent_realism_admitted"] and not readouts["final_heldout_admitted"]
    verify_pin_tree(readouts["protocol"], verified)
    verify_pin_tree(readouts["observations"], verified)
    protocol = json.loads((ROOT / readouts["protocol"]["path"]).read_text())
    verify_pin_tree(protocol["inputs"], verified)
    verify_pin_tree(protocol["shards"], verified)
    observations = json.loads((ROOT / readouts["observations"]["path"]).read_text())
    assert len(observations) == len({r["index"] for r in observations}) == 1408
    assert len(readouts["families"]) == 22
    expected_counts = {
        "conditioned_true_endpoint": (866, 323, 399, 515),
        "conditioned_terminal_argmax": (996, 379, 488, 594),
        "cyclic_true_endpoint": (927, 313, 404, 465),
        "cyclic_terminal_argmax": (1067, 368, 490, 546),
        "selected": (1408, 1387, 801, 837),
    }
    for arm, (valid, exact, fp_members, descriptor_members) in expected_counts.items():
        group = readouts["summaries"][arm]
        assert (
            group["requests"] == 1408 and group["connected"] == valid and group["exact_L1"] == exact
        )
        assert (
            group["fingerprint_unique_members"] == fp_members
            and group["descriptor_member_observations"] == descriptor_members
        )
        assert sum(bool(r["arms"][arm]["smiles"]) for r in observations) == valid
        assert sum(r["arms"][arm]["exact_L1"] for r in observations) == exact
        for row in readouts["families"]:
            a = row["arms"][arm]
            subset = [r for r in observations if r["family"] == row["family"]]
            assert len(subset) == a["counts"]["requests"] == 64
            assert a["counts"]["connected"] == sum(bool(r["arms"][arm]["smiles"]) for r in subset)
            assert a["exact_L1"] == sum(r["arms"][arm]["exact_L1"] for r in subset)
            d = a["descriptor_neighborhood"]
            assert len(d["members"]) == a["counts"]["connected"]
            assert d["member_observations"] == sum(d["members"])
            assert d["member_observations_per_request"] == sum(d["members"]) / 64
    paired = documents["paired_readouts"]
    independent = documents["paired_readouts_review"]
    assert paired["complete"] and paired["full_conditioned_terminal_recovery_equal"]
    assert independent["passed"] and independent["comparison"] == pin(
        ROOT / documents["main"]["inputs"]["paired_readouts"]["path"]
    )
    assert independent["conditioned_prediction_shards_all14heads_tensor_exact"] == 880
    assert len(independent["prediction_fields"]) == 14
    assert independent["full_terminal_arm_dictionaries_exact"] == 7040
    assert independent["draw0_requests"] == 1408 and independent["all5_trajectories"] == 7040
    assert independent["all5_independent_requests"] is False
    verify_pin_tree(paired["protocol"], verified)
    verify_pin_tree(paired["inputs"], verified)
    verify_pin_tree(independent["reviewer"], verified)
    for path, sha in independent["verified_pins"].items():
        verify_pin_tree({"path": path, "sha256": sha}, verified)
    assert len(paired["counts"]) == len(independent["counts"]) == 184
    for count in independent["counts"]:
        key = (count["population"], count["family"], count["arm"])
        original = next(
            r for r in paired["counts"] if (r["population"], r["family"], r["arm"]) == key
        )
        assert all(count[k] == v for k, v in original.items())
        n = (1408 if count["population"] == "draw0" else 7040) // (
            1 if count["family"] == "ALL" else 22
        )
        assert count["trajectories"] == n
        assert count["invalid_or_disconnected"] + count["valid_nonexact"] + count["exact_L1"] == n
        assert count["nonexact"] + count["exact_L1"] == n
    assert len(paired["paired"]) == len(independent["paired_counts"]) == 368
    for a, b in zip(paired["paired"], independent["paired_counts"], strict=True):
        # Serialized ordering differs, so match complete contrast identity explicitly.
        b = next(
            x
            for x in independent["paired_counts"]
            if all(x[k] == a[k] for k in ("population", "family", "before", "after", "metric"))
        )
        assert len(a["gains"]) == b["gains"] and len(a["losses"]) == b["losses"]
        assert a["net"] == b["net"] and a["trajectories"] == b["trajectories"]
        assert b["gains"] + b["losses"] + b["both_pass"] + b["both_fail"] == b["trajectories"]
    return {
        "complete_paired_readout_cells": 184,
        "complete_paired_contrasts": 368,
        "original_14head_prediction_shards_exact": 880,
        "readout_quality_complete_1408_panels": 5,
        "readout_quality_first_draw_shards_authenticated": 352,
        "selected_requests": 1408,
        "terminal_draw0": dict(tally["draw0"]["ALL"]),
        "terminal_all5": dict(tally["all5"]["ALL"]),
        "terminal_shards_authenticated": 880,
        "decomposition_counts": dict(sums),
        "quality_requests": 1408,
        "descriptor_families": 22,
        "independent_evidence_admitted": False,
    }


def verify_current_evidence(ledger):
    """Check documentary values against immutable results, without rerunning science."""
    manifest_path = HERE / "current_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "forge.iclr22.current_evidence_manifest.v1"
    assert manifest["evidence_class"] == "Computed"
    assert manifest["final_evidence_admitted"] is False
    assert ledger["current_evidence_manifest"] == pin(manifest_path)
    assert set(manifest["documents"]) == {"main", "quality", "appendix"}
    verified = {}
    verify_pin_tree(manifest["sources"], verified)
    verify_pin_tree(manifest["documents"], verified)
    documents = {
        key: json.loads((ROOT / value["path"]).read_text())
        for key, value in {**manifest["sources"], **manifest["documents"]}.items()
    }
    for key in manifest["documents"]:
        verify_pin_tree(documents[key]["inputs"], verified)
    assert documents["main"]["final_heldout_evidence_admitted"] is False
    assert documents["quality"]["final_evidence_admitted"] is False
    for assertion in manifest["assertions"]:
        actual = json_value(documents[assertion["source"]], assertion["pointer"])
        assert actual == assertion["equals"], assertion
    for binding in manifest["document_bindings"]:
        actual = json_value(documents[binding["document"]], binding["document_pointer"])
        expected = json_value(documents[binding["source"]], binding["source_pointer"])
        assert actual == expected, binding
    routes = documents["routes"]
    assert set(routes["by_family"]) == set(ledger["family_ids"])
    assert all(row["requests"] == 64 for row in routes["by_family"].values())
    for field in ("requests", "exact_L1", "limited_design_pass", "computational_makeability"):
        assert sum(row[field] for row in routes["by_family"].values()) == routes["totals"][field]
    assert len(routes["rows"]) == 1408
    assert len({row["index"] for row in routes["rows"]}) == 1408
    assert sum(row["exact_L1"] for row in routes["rows"]) == 1387
    assert sum(row["exact_L1"] and row["limited_design_pass"] for row in routes["rows"]) == 1324
    assert sum(row["computational_makeability"] for row in routes["rows"]) == 616
    for row in documents["main"]["family_rows"]:
        assert row["route"] == routes["by_family"][row["family"]]
    completion_checks = verify_table_completion(documents, verified)
    return {
        "table_completion_checks": completion_checks,
        "manifest": pin(manifest_path),
        "verified_pins": verified,
        "source_assertions": len(manifest["assertions"]),
        "document_bindings": len(manifest["document_bindings"]),
        "full_route_identity_count": len(routes["rows"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-pdf", type=Path)
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Check sources only; never emits a successful PDF/build receipt",
    )
    args = parser.parse_args()
    if not args.sources_only and args.export_pdf is None:
        parser.error("--export-pdf is required unless --sources-only is supplied")
    ledger = json.loads((HERE / "RESULTS_LEDGER.json").read_text())
    for value in ledger["inputs"].values():
        assert digest(ROOT / value["path"]) == value["sha256"], value["path"]
    methods_verified = {}
    assert ledger["methods_implementation_inputs"]
    verify_pin_tree(ledger["methods_implementation_inputs"], methods_verified)
    cohort_path = ROOT / "results/phase1/compose_lipid_training_cohort_v1/cohort.json"
    cohort = json.loads(cohort_path.read_text())
    assert ledger["family_ids"] == cohort["families"]
    sections = {p.name: p.read_text() for p in (HERE / "sections").glob("*.tex")}
    new_results = (
        sections["results_22.tex"] + sections["appendix_22.tex"] + sections["quality_current.tex"]
    )
    current_evidence = verify_current_evidence(ledger)
    metric_pin = ledger["quality_metric_contract"]
    assert digest(ROOT / metric_pin["path"]) == metric_pin["sha256"]
    metrics = json.loads((ROOT / metric_pin["path"]).read_text())
    for value in metrics["inputs"]:
        assert digest(ROOT / value["path"]) == value["sha256"], value["path"]
    assert metrics["table_labels"] == ledger["explicit_quality_tables"]
    for label in metrics["table_labels"]:
        assert "\\label{" + label + "}" in new_results
    for group, expected_count in (("quality_metrics", 11), ("realism_metrics", 9)):
        assert len(metrics[group]) == expected_count
        assert len({metric["key"] for metric in metrics[group]}) == expected_count
        for metric in metrics[group]:
            assert metric["final_value"] is None
            assert metric["label"] in new_results
    descriptor_source = ROOT / "forge/model/common_lipid_realism.py"
    assignments = ast.parse(descriptor_source.read_text()).body
    descriptor_names = next(
        ast.literal_eval(node.value)
        for node in assignments
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DESCRIPTOR_NAMES"
            for target in node.targets
        )
    )
    assert [row["key"] for row in metrics["descriptors"]] == list(descriptor_names)
    for row in metrics["descriptors"]:
        assert row["label"] in new_results
        assert all(
            row[key] is None for key in ("reference", "raw", "final", "normalized_wasserstein")
        )
    for block in ledger["blocks"]:
        assert block["status"] in {"DEVELOPMENT_AVAILABLE_FINAL_PENDING", "CONDITIONAL_NOT_RUN"}
        assert block["final_result_artifact"] is None
        assert block["final_evidence_admitted"] is False
        assert block["remaining_final_work"]
        assert "\\label{" + block["latex_label"] + "}" in new_results
    # Preserve the declared family order independently of whether a row is populated.
    old_appendix = (
        HERE / "archive/before_measured_results_20260925/sections/appendix_22.tex"
    ).read_text()
    old_census = old_appendix.split(r"\label{tab:22-census}", 1)[1].split(r"\end{table}", 1)[0]
    family_labels = [
        line.split(" & ")[0] for line in old_census.splitlines() if " & \\PendingCell" in line
    ]
    assert len(family_labels) == len(set(family_labels)) == 22
    quality_evidence = json.loads((HERE / "current_quality_evidence.json").read_text())
    quality_labels = [
        quality_evidence["by_family"][family]["label_latex"] for family in ledger["family_ids"]
    ]
    assert len(quality_labels) == len(set(quality_labels)) == 22
    tables = [
        match.group(0)
        for match in re.finditer(
            r"\\begin\{(table\*?|longtable)\}.*?\\end\{\1\}", new_results, flags=re.S
        )
    ]
    matrix_rows = {}
    for label in ledger["family_matrices"]:
        table = next(t for t in tables if "\\label{" + label + "}" in t)
        expected_labels = (
            quality_labels
            if "\\label{" + label + "}" in sections["quality_current.tex"]
            else family_labels
        )
        rows = []
        for line in table.splitlines():
            cell = line.split(" & ")[0].strip()
            if cell in expected_labels:
                rows.append(cell)
            elif label in {"tab:22-realism", "tab:22-diversity"}:
                matches = [name for name in expected_labels if (r"\textbf{" + name + "}") in cell]
                assert len(matches) <= 1, line
                rows.extend(matches)
        panels = 1
        assert len(rows) == panels * len(ledger["family_ids"]), label
        assert rows == expected_labels * panels, label
        matrix_rows[label] = rows
    verify_populated_tables(tables, ledger, quality_evidence, family_labels, quality_labels)
    for label in ledger["figure_slots"]:
        assert "\\label{" + label + "}" in new_results
    for text in sections.values():
        assert not any(ord(c) < 32 and c not in "\n\t" for c in text)
    numerical = []
    for p in sorted((HERE / "generated").glob("*.tex")):
        historical = HERE / "archive/inherited_exports/FORGE_ICLR2027_Overleaf/generated" / p.name
        assert historical.exists() and digest(p) == digest(historical), p.name
        numerical.append(pin(p))
    manuscript = (HERE / "FORGE_ICLR2027_paper.tex").read_text()
    original = (HERE / "archive/three_family_source.tex").read_text()
    assert r"\section{Results}" in sections["results_22.tex"]
    active_text = (
        manuscript
        + new_results
        + sections["implementation_22.tex"]
        + sections["reaction_figures_22.tex"]
    )
    reaction_figures = verify_reaction_figures(
        ledger,
        sections,
        manuscript,
        None if args.sources_only else args.export_pdf.resolve().parent,
    )
    for forbidden in (
        "Working draft",
        "working draft",
        "This extension",
        "result slots",
        "Results to be completed",
        r"\ResultSlot",
        r"\TwentyTwoScope",
        r"\input{sections/historical_three_family_results.tex}",
    ):
        assert forbidden not in active_text, forbidden
    assert r"\title{FORGE: Reaction-Guided Generative Design of Ionizable Lipids}" in manuscript
    original_intro = original.split(r"\section{Introduction}", 1)[1].split(
        "Our contributions are:", 1
    )[0]
    final_intro = manuscript.split(r"\section{Introduction}", 1)[1].split(
        "Our contributions are:", 1
    )[0]
    assert original_intro == final_intro, "Original Introduction prose changed"
    archived_manuscript = (
        HERE / "archive/before_measured_results_20260925/FORGE_ICLR2027_paper.tex"
    ).read_text()
    intro_start = r"\section{Introduction}"
    intro_end = r"\section{Preliminaries and problem formulation}"
    before_intro = archived_manuscript.split(intro_start, 1)[1].split(intro_end, 1)[0]
    after_intro = manuscript.split(intro_start, 1)[1].split(intro_end, 1)[0]
    correction = ledger["introduction_scientific_correction"]
    assert before_intro.count(correction["before"]) == 1
    assert after_intro.count(correction["after"]) == 1
    assert before_intro.replace(correction["before"], correction["after"], 1) == after_intro
    theory_start = r"\section{Theoretical analysis and proofs}"
    original_theory = original.split(theory_start, 1)[1].split(
        r"\section{Reaction schemes and generated examples}", 1
    )[0]
    final_theory = manuscript.split(theory_start, 1)[1].split(
        r"\section{Assembly families and reaction schemes}", 1
    )[0]
    assert original_theory == final_theory, "Original theory changed"
    assert not any(ord(c) < 32 and c not in "\n\t" for c in active_text)
    assert r"\input{sections/quality_current.tex}" in manuscript + sections["appendix_22.tex"]
    if args.sources_only:
        print(
            json.dumps(
                {
                    "status": "passed_source_checks_build_pending",
                    "current_evidence": current_evidence,
                    "all_family_matrices": len(matrix_rows),
                    "PDF_and_export_checks_run": False,
                }
            )
        )
        return
    log = (HERE / "FORGE_ICLR2027_paper.log").read_text()
    assert not re.search(r"Overfull|undefined|multiply defined|^!", log, re.M)
    export_log = args.export_pdf.with_suffix(".log").read_text()
    assert not re.search(r"Overfull|undefined|multiply defined|^!", export_log, re.M)
    pdf = HERE / "FORGE_ICLR2027_paper.pdf"
    text = pdf_text(pdf)
    assert text == pdf_text(args.export_pdf), "Portable export differs from canonical PDF text"
    pages = text.split("\f")
    count = sum(bool(p.strip()) for p in pages)
    header_pages = [i + 1 for i, p in enumerate(pages) if "AI USE STATEMENT" in p]
    report = dict(
        schema_version="forge.iclr22.current_results_validation.v2",
        status="passed",
        claim="Verified computed development evidence and local builds; no independent final-study admission",
        current_evidence=current_evidence,
        source_inputs=ledger["inputs"],
        cohort=pin(cohort_path),
        family_count=len(ledger["family_ids"]),
        evidence_blocks=len(ledger["blocks"]),
        all_family_matrices=len(matrix_rows),
        matrix_rows=matrix_rows,
        figure_slots=ledger["figure_slots"],
        explicit_quality_tables=metrics["table_labels"],
        quality_metric_count=len(metrics["quality_metrics"]),
        realism_metric_count=len(metrics["realism_metrics"]),
        descriptor_count=len(descriptor_names),
        quality_metric_contract=metric_pin,
        quality_metric_implementation_pins_verified=metrics["inputs"],
        methods_implementation_pins_verified=ledger["methods_implementation_inputs"],
        reaction_figures=reaction_figures,
        pending_cell_count=new_results.count("\\PendingCell"),
        inline_value_placeholders=active_text.count("\\PendingValue"),
        original_introduction_prose_preserved=True,
        introduction_contribution_correction=correction,
        all_other_introduction_text_unchanged=True,
        original_theory_preserved=True,
        final_paper_narrative_without_draft_meta_commentary=True,
        all_final_result_artifacts_missing_and_explicit=True,
        historical_numerical_inputs_unchanged=numerical,
        compiled_pages=count,
        disclosure_starts_page=header_pages[0] if header_pages else None,
        canonical_and_fresh_zip_build_text_identical=True,
        unresolved_references=False,
        overfull_boxes=False,
        manuscript_inputs=[
            pin(p)
            for p in [
                HERE / "FORGE_ICLR2027_paper.tex",
                *sorted((HERE / "sections").glob("*.tex")),
                HERE / "RESULTS_LEDGER.json",
                HERE / "RESULTS_CHECKLIST.md",
                HERE / "QUALITY_METRICS.json",
            ]
        ],
        implementation=pin(Path(__file__)),
        outputs=[pin(pdf), pin(HERE / "FORGE_ICLR2027_22Families_Overleaf.zip")],
        no_new_training_sampling_or_test_access_for_document_task=True,
        sibling_v1_iclr_not_edited_by_this_task=True,
    )
    (HERE / "iclr22_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "family_count",
                    "evidence_blocks",
                    "all_family_matrices",
                    "pending_cell_count",
                    "compiled_pages",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
