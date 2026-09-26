"""Populate appendix tables from saved results only; no model or chemistry calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/v1_iclr22"
RESEARCH = Path("results/phase1/compose_lipid_iclr22_research_v1")
PAR = RESEARCH / "parallel_completion_v1"
TRAIN = PAR / "training_evaluation"
EVAL = TRAIN / "group_loss_evaluation_v1"
INPUTS: dict[str, dict[str, str]] = {}


def read(path: str | Path, key: str) -> dict:
    path = Path(path)
    data = (ROOT / path).read_bytes()
    INPUTS[key] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    return json.loads(data)


def row(*cells: object) -> str:
    return " & ".join(str(x) for x in cells) + r" \\" + "\n"


def table(label: str, caption: str, columns: str, header: str, body: str, note: str = "") -> str:
    return (
        "\\begin{table}[H]\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}"
        + "\n\\begin{tabular}{@{}"
        + columns
        + "@{}}\n\\toprule\n"
        + header
        + "\\midrule\n"
        + body
        + "\\bottomrule\n\\end{tabular}\n"
        + ("\\par\\smallskip\\raggedright\\scriptsize\n" + note + "\n" if note else "")
        + "\\end{table}"
    )


def replace_table(text: str, label: str, replacement: str) -> str:
    position = text.index("\\label{" + label + "}")
    start = text.rfind(r"\begin{table}", 0, position)
    stop = text.index(r"\end{table}", position) + len(r"\end{table}")
    assert start >= 0
    return text[:start] + replacement + text[stop:]


def main() -> None:
    cohort = read("results/phase1/compose_lipid_training_cohort_v1/cohort.json", "cohort")
    config = read(
        "results/phase1/compose_lipid_training_eight_fp32_v2/training-config.json",
        "training_config",
    )
    complete = read(
        "results/phase1/compose_lipid_training_eight_fp32_v2/completion-summary.json",
        "training_completion",
    )
    remote = read(
        "results/phase1/compose_lipid_training_eight_fp32_v2/remote-result.json", "training_runtime"
    )
    read("results/phase1/compose_lipid_training_eight_fp32_v2/request.json", "training_request")
    read(
        "results/phase1/compose_lipid_adaptive_eight_integration_v1/verification.json",
        "execution_qualification",
    )
    cal = read(TRAIN / "cal_bridge_full_v1/namespace_rollup_v1/result.json", "cal_representation")
    routes = read(PAR / "routes/recount_closeout_v3/result.json", "route_quality_recount")
    treatment = read(routes["inputs"]["treatment"]["path"], "route_treatment")
    read(routes["inputs"]["treatment_execution"]["path"], "route_execution")
    generations = {
        arm: read(EVAL / f"execution_{arm}/generation/result.json", f"generation_{arm}")
        for arm in ("original", "categorical", "coarse_mixture")
    }
    comparison = read(EVAL / "comparison_v1.json", "group66_comparison")
    read(EVAL / "independent_result_review_v1.json", "group66_review")
    group_run = read(
        TRAIN
        / "group_loss_pilot_v1/training_only_readiness_v1/prepared_v1/observations/20260926T025900.623469Z/result.json",
        "group66_training",
    )
    read(
        TRAIN
        / "group_loss_pilot_v1/training_only_readiness_v1/prepared_v1/training_admission.json",
        "group66_admission",
    )
    cyclic = read(
        TRAIN / "cyclic_inference_qualification_v1/full_pair_v1/paired_result.json",
        "cyclic_comparison",
    )
    read(
        TRAIN / "cyclic_inference_qualification_v1/full_pair_v1/root_admission.json",
        "cyclic_admission",
    )
    fallback = read(EVAL / "verified_fallback_diagnostic_v1/result.json", "fallback")
    read(EVAL / "verified_fallback_diagnostic_v1/independent_review_v1.json", "fallback_review")
    update = read(
        RESEARCH / "training_decoder/next_experiment/v4/TRAINING_CLOSEOUT.json", "update_training"
    )
    read(
        RESEARCH / "quality/training_comparison_review/supplement_review_v1/outcomes.json",
        "update_outcomes",
    )
    repairs = read(
        PAR / "quality/repair_experiment_readiness_v1/full_run_v1/result.json", "repair192"
    )
    read(
        PAR / "quality/repair_experiment_readiness_v1/full_verification_v1.json",
        "repair192_verification",
    )
    read(TRAIN / "shared_null_integration_candidate_v1/result.json", "null_integration")
    read(TRAIN / "shared_null_prior_admission_v1/result.json", "null_prior")
    read(TRAIN / "shared_null_prior_v1/run_v1/result.json", "full_physical_TRAIN_prior")
    theory = read(RESEARCH / "theory_correspondence.json", "theory_correspondence")
    activity = read(
        "results/phase1/continuous_activity_guidance_v1/analysis/result.json",
        "activity_conditioning",
    )
    read(
        "results/phase1/continuous_activity_guidance_v1/verification.json", "activity_verification"
    )

    families = cohort["families"]
    names = [
        "A3 amine--aldehyde--alkyne",
        "Acid--epoxide diester",
        "AEMA",
        "Aldehyde Ugi 3-CR",
        "Aldehyde Ugi 4-CR",
        r"$\alpha$-Isocyanoester dihydroimidazole",
        "Amine alkylation",
        "Amine--epoxide",
        "Aryl reductive amination",
        "Aza-Michael acrylamide",
        "Aza-Michael acrylate",
        "Disulfide Michael",
        "Epoxide O-acylation",
        "iPhos",
        "Ketone--isocyanide amide",
        "Ketone Ugi 4-CR",
        "Maleate",
        "O-esterification",
        "Passerini",
        "Preassembled thiol-yne",
        "Reductive amination",
        "Thiolactone",
    ]
    assert len(families) == len(names) == 22
    name = dict(zip(families, names))
    pending = r"\PendingCell"
    wide = r">{\raggedright\arraybackslash}p{.31\textwidth}"
    summary = {
        arm: {(x["family"], x["arm"]): x for x in result["summary"]["by_family_arm"]}
        for arm, result in generations.items()
    }
    cy = {(x["family"], x["readout"]): x for x in cyclic["paired"]}
    fb = {
        arm: {(x["family"], x["metric"]): x for x in rows}
        for arm, rows in fallback["paired_within_arm"].items()
    }
    assert treatment["totals"]["requests"] == 1408
    assert treatment["totals"]["combined_primary_products"] == 616
    assert sum(cohort["by_family"].values()) == 1192065
    assert generations["original"]["complete"] and comparison["complete"]
    assert config["runtime"]["optimizer_steps"] == complete["completed_steps"] == 2794
    assert complete["presentations_per_family"] == 402336
    assert complete["examples_seen"] == 8851392
    assert round(remote["total_seconds"], 3) == 1377.725
    assert round(remote["preparation_seconds"], 3) == 31.801
    assert group_run["complete"] and round(group_run["function_GPU_minutes"], 3) == 55.706
    assert update["arms"]["large"]["optimizer_updates"] == 220
    assert update["arms"]["small"]["optimizer_updates"] == 2640
    assert activity["complete_training_fits"] == 27
    assert activity["biological_improvement_established"] is False
    text = (PAPER / "sections/appendix_22.tex").read_text()
    tables = {}

    # Table 4: unit-specific corpus and CAL counts; no invented independent census.
    body = ""
    census_rows = []
    for f in families:
        counts = cal["by_family"][f]
        admitted = counts.get("admitted_unique_physical", 0) + counts.get(
            "admitted_exact_role_alias", 0
        )
        total = admitted + counts.get("not_source_qualified", 0)
        variants = sum(x["model_condition_variants"] for x in cal["rows"] if x["family"] == f)
        census_rows.append(
            {
                "family": f,
                "TRAIN_records": cohort["by_family"][f],
                "CAL_admitted": admitted,
                "CAL_total": total,
                "CAL_variants": variants,
            }
        )
        body += row(
            name[f], f"{cohort['by_family'][f]:,}", "1/22", f"{admitted}/{total}", variants, pending
        )
    body += r"\midrule" + "\n" + row("Total", "1,192,065", "1", "1,859/2,059", "1,885", pending)
    tables["tab:22-census"] = table(
        "tab:22-census",
        r"\textbf{Admitted training corpus and guarded CAL representation.} TRAIN records and CAL molecules are different units. A CAL molecule may have multiple qualified condition variants. These counts measure support and admission, not model success.",
        wide + "ccccc",
        row(
            "Family",
            r"\shortstack{TRAIN\\records}",
            r"\shortstack{Sampling\\mass}",
            r"\shortstack{Admitted CAL /\\guarded molecules}",
            r"\shortstack{CAL\\variants}",
            r"\shortstack{Independent\\study census}",
        ),
        body,
        r"The training measure assigns equal family mass and uniform constitutional-product mass within each family. Pending support excludes 80,401 records; vitamin-B5 multistep has no admitted training support. Declared graph support is 254 atoms and 12 closures. The corpus contains 44,312 records above 96 atoms. Independent study counts, source-disjoint evaluation and a separate source-row versus constitution census remain unmeasured. TEST outcomes are unopened.",
    )

    # Table 5: existing full fit, fixed continuation policies and explicitly scoped costs.
    settings = [
        ("Independent full fit / seed", "1 / 2026092401"),
        ("Families / presentations per family / total", "22 / 402,336 / 8,851,392"),
        ("Optimizer updates / updates containing each family", "2,794 / 381"),
        ("Global batch / participating families / examples per family", "3,168 / 3 / 1,056"),
        (
            "Optimizer / learning rate / weight decay / clip",
            r"AdamW / $10^{-4}$ / $5\!\times\!10^{-5}$ / 1.0",
        ),
        (r"AdamW $(\beta_1,\beta_2)$ / $\epsilon$", r"$(0.9,0.999)$ / $10^{-8}$"),
        ("Transformer layers / width / heads / dropout", "6 / 192 / 8 / 0.1"),
        ("Routed experts / adapter width / graph support", "3 / 64 / 254 atoms, 12 closures"),
        ("Gradient and chemistry balancing", "Family PCGrad; equal present-role mass"),
        ("Auxiliary role / core / repeat weights", "0.25 / 0.25 / 0.25"),
        ("Offspring / junction / topology-chemistry weights", "1.0 / 0.5 / 1.0"),
        ("Hardware / precision / determinism", "8 H100 / FP32 / deterministic"),
        ("Prefetch workers / depth / checkpoint interval", "2 / 2 / 220 updates"),
        ("Execution / included preparation time", "1,377.725 s / 31.801 s"),
        ("Serial versus eight-GPU; kill/restart", "Exact 44-step model, optimizer, RNG replay"),
        ("Development sampling", "352 requests; 16/family; 64 flow steps"),
        ("Adaptive selected cohort", "1,408 separate requests; 64/family"),
        ("Final independent held-out protocol and fits", pending),
    ]
    tables["tab:22-training"] = table(
        "tab:22-training",
        r"\textbf{Measured training configuration and execution.} One complete fit initializes the continuation diagnostics. Hash-pinned inputs, checkpoints and commands accompany the evidence ledger.",
        r">{\raggedright\arraybackslash}p{.50\textwidth}>{\raggedright\arraybackslash}p{.46\textwidth}",
        row("Field", "Setting / result"),
        "".join(row(*x) for x in settings),
        r"The 22.96-minute execution excludes allocation/image initialization and later evaluation; it does not meet a 15-minute execution target. Fixed-presentation continuation uses 31,680 additional presentations/family: batch 3,168 for 220 updates (8 GPUs; 40.859 function-GPU-minutes) versus batch 264 for 2,640 updates (1 GPU; 16.082 function-GPU-minutes). A separate paired categorical/coarse-parent pilot fixes 66 updates and 9,504 presentations/family/arm (55.706 function-GPU-minutes for the successful paired execution). Function-GPU-minutes are not billed dollar costs. Failed staging and timeout attempts remain in the run ledger; neither continuation is an independent full fit.",
    )

    # Table 6: complete unchanged 352-attempt readout cohort, not the selected 1408.
    body = ""
    outcomes = []
    for f in families + ["ALL"]:
        s = summary["original"]
        d, raw, terminal = (s[(f, a)] for a in ("d1", "flow_endpoint", "t1_argmax"))
        exact_smiles = {
            a["arms"]["d1"]["smiles"]
            for a in generations["original"]["attempts"]
            if (f == "ALL" or a["family"] == f) and a["arms"]["d1"]["check"].get("exact") is True
        }
        record = {
            "family": f,
            "requests": d["requests"],
            "raw_exact": raw["source_exact"],
            "terminal_exact": terminal["source_exact"],
            "D1_exact": d["source_exact"],
            "D1_valid_connected": d["valid_connected"],
            "D1_distinct_exact": len(exact_smiles),
            "D1_nonexact": d["requests"] - d["source_exact"],
        }
        outcomes.append(record)
        if f == "ALL":
            body += r"\midrule" + "\n"
        body += row(
            name.get(f, "All requests"),
            d["requests"],
            raw["source_exact"],
            terminal["source_exact"],
            d["source_exact"],
            d["valid_connected"],
            len(exact_smiles),
            record["D1_nonexact"],
        )
    tables["tab:22-outcomes"] = table(
        "tab:22-outcomes",
        r"\textbf{Attempt-level readout outcomes on the fixed 352-request development cohort.} One original checkpoint; 16 TRAIN-derived requests per family. Raw is the flow endpoint, terminal is the $t=1$ argmax, and D1 is constrained readout. Every failed request remains in $N$.",
        r">{\raggedright\arraybackslash}p{.29\textwidth}" + "ccccccc",
        row(
            "Family",
            "$N$",
            r"\shortstack{Raw\\exact}",
            r"\shortstack{Terminal\\exact}",
            r"\shortstack{D1\\exact}",
            r"\shortstack{D1 valid +\\connected}",
            r"\shortstack{D1 distinct\\exact}",
            r"\shortstack{D1\\nonexact}",
        ),
        body,
        r"These outputs precede the larger proposal/selection pipeline used for the separate 1,408-product cohort. A failed exact check is not experimental evidence of unmakeability. Ambiguity, checked-trace precision and independent-seed failure distributions are retained as unmeasured endpoints (\PendingCell); replay among accepted exact traces is a verifier invariant, not a precision estimate.",
    )

    body = ""
    for f in families:
        body += row(
            name[f], f"{treatment['by_family'][f]['exact_L1']}/64", pending, pending, pending
        )
    tables["tab:22-seeds"] = table(
        "tab:22-seeds",
        r"\textbf{Current single-fit result and missing independent replications.} The first column is the adaptively selected 1,408-request cohort from seed 2026092401, not a fresh held-out evaluation. Two further independent fits and their matched inference remain unrun. Sampling seeds and continuations do not supply training-seed replication.",
        wide + "cccc",
        row(
            "Family",
            r"\shortstack{Existing fit\\exact / $N$}",
            r"\shortstack{Independent\\fit B}",
            r"\shortstack{Independent\\fit C}",
            r"\shortstack{Mean $\pm$\\sample SD}",
        ),
        body,
        r"The existing selected cohort has 1,387/1,408 exact products. No multi-fit mean, sample standard deviation or confidence interval is estimated from this single fit. The prospective matched held-out values remain \PendingCell.",
    )

    body = ""
    for f in families + ["ALL"]:
        if f == "ALL":
            body += r"\midrule" + "\n"
        x = cy[(f, "d1")]["metrics"]
        body += row(
            name.get(f, "All requests"),
            "16" if f != "ALL" else "352",
            x["exact_L1"]["conditioned"],
            x["exact_L1"]["cyclic"],
            x["limited_design"]["conditioned"],
            x["limited_design"]["cyclic"],
        )
    tables["tab:22-controls"] = table(
        "tab:22-controls",
        r"\textbf{Paired inference-only cyclic program-ID control.} The same checkpoint, requests and random draws receive the true or cyclically reassigned program ID. True structural context and source verification remain fixed. Counts use D1; design denotes the limited structural/design checks.",
        wide + "ccccc",
        row(
            "Family",
            "$N$",
            r"\shortstack{True ID\\exact}",
            r"\shortstack{Cyclic ID\\exact}",
            r"\shortstack{True ID\\design}",
            r"\shortstack{Cyclic ID\\design}",
        ),
        body,
        r"On all 352 requests, raw exact counts are 80 versus 71 and terminal-argmax counts are 91 versus 85. The intervention tests the incremental program-ID signal with correct core/role/layout/depth information. A consistent training/test label bijection would instead be a recoding. Independently trained true shared-null, finite-catalogue selector, matched/generous FACT, qualified native external baselines and the matched three-family bridge remain \PendingCell. Shared-null implementation tests are not trained-null performance.",
    )

    body = ""
    for key, display in [
        ("flow_endpoint", "Original raw endpoint"),
        ("t1_argmax", "Original terminal argmax"),
        ("d1", "Original D1"),
    ]:
        x = summary["original"][("ALL", key)]
        body += row(
            display, 352, x["source_exact"], x["qualified_design_pass"], "Same saved trajectories"
        )
    for arm, display in [
        ("original", "Original verified fallback"),
        ("categorical", "Categorical66 verified fallback"),
        ("coarse_mixture", "Coarse66 verified fallback"),
    ]:
        body += row(
            display,
            352,
            fb[arm][("ALL", "exact_L1")]["after"],
            fb[arm][("ALL", "limited_design")]["after"],
            "No new model/proposal calls",
        )
    body += r"\midrule" + "\n"
    repair_totals = {}
    for arm, display in [
        ("original", "Three-family repair identity control"),
        ("ring_scheduled", "Ring-scheduled proposals"),
        ("anchor_preserving", "Anchor-preserving proposals"),
    ]:
        fs = repairs["arms"][arm]
        exact = sum(x["exact"] for x in fs.values())
        design = sum(x["limited_design_pass"] for x in fs.values())
        repair_totals[arm] = {"requests": 192, "exact": exact, "limited_design": design}
        body += row(display, 192, exact, design, "Separate construction diagnostic")
    body += row("Equal-cost / no TRAIN-motif-prior control", pending, pending, pending, pending)
    tables["tab:22-decoder"] = table(
        "tab:22-decoder",
        r"\textbf{Readout, saved-candidate selection and repair attribution.} The 352-request readouts and 192-request repair study have distinct populations. They do not estimate a model improvement or independent realism.",
        r">{\raggedright\arraybackslash}p{.38\textwidth}ccc>{\raggedright\arraybackslash}p{.29\textwidth}",
        row("Arm", "$N$", "Exact", "Design", "Attribution"),
        body,
        r"Verified fallback retains an exact D1 candidate, otherwise takes an already exact raw endpoint, otherwise an already exact terminal argmax, otherwise retains the D1 failure. It rescues 5/7/7 original/categorical/coarse requests, with no new candidates or exact/design losses. The categorical phosphate-tail Shannon count falls from 9.1180 to 9.0757; cross-checkpoint diversity floors still fail. The repair study includes all 64 A3, aldehyde-Ugi4 and ketone-Ugi4 requests: design counts change 51/50/39 to 55/51/40. Both repair arms change the same six requests and preserve the declared integer diversity/novelty floors; three resulting structures differ between the ring and anchor arms without an additional design-pass gain. Equal proposal caps do not mean equal measured cost; the full study uses 7.209 CPU seconds, 84 source checks and 84 design checks. The changed molecules have no transferred routing verdict, and the official 1,408-product cohort is unchanged. Individual decoder-rule and matched-cost effects remain \PendingCell.",
    )

    body = ""
    for f in families + ["ALL"]:
        x = treatment["totals"] if f == "ALL" else treatment["by_family"][f]
        if f == "ALL":
            body += r"\midrule" + "\n"
        body += row(
            name.get(f, "All requests"),
            x["requests"],
            x["exact_L1"],
            x["L2_ready_products"],
            x["L3_direct_only_products"],
            x["combined_primary_products"],
            x["strict_secondary_products"],
        )
    tables["tab:22-routes"] = table(
        "tab:22-routes",
        r"\textbf{Product-level computational makeability and strict secondary dossiers.} The fixed, adaptively developed 1,408-request selected cohort contains 64 requests per family. All columns retain all requested products, including the 21 nonexact-L1 products. Evidence is assessed as of 2026-09-26 01:56:45 UTC.",
        r">{\raggedright\arraybackslash}p{.31\textwidth}" + "cccccc",
        row(
            "Family",
            "$N$",
            r"\shortstack{Exact\\L1}",
            r"\shortstack{L2\\ready}",
            r"\shortstack{L3\\direct-only}",
            r"\shortstack{Computationally\\makeable}",
            r"\shortstack{Strict\\secondary}",
        ),
        body,
        r"Computational makeability requires every required component to be directly vendor-listed or connected through a complete admitted computational route to vendor-listed terminal materials. L2-ready counts products whose required components have a direct listing or admitted computational path; it does not require every route leaf to be listed. Direct-only requires all required components to be directly listed. Strict dossiers retain the original exact-source/current-item checks. Vendor-directory listings are not stock, prices or experimental synthesis. Missing listings remain unknown. The 3,723 required branches are distinct from 4,510 incorporated component occurrences: 3,488 branches are L2-ready and 2,787 computationally makeable. There are 792 products without demonstrated primary closure. Of the 616 makeable products, 606 pass the limited design checks and 604 additionally have no context flags. Independent-seed held-out route coverage remains \PendingCell.",
    )

    architecture_rows = [
        ("Whole graph versus factorized; matched/generous capacity", pending),
        ("Input-only versus layerwise conditioning", pending),
        ("Role/core/repeat, routed adapters and PCGrad retraining ablations", pending),
        ("Original / categorical66 / coarse66 raw exact", "80 / 83 / 86 of 352"),
        ("Original / categorical66 / coarse66 D1 exact", "140 / 136 / 135 of 352"),
        ("Original / categorical66 / coarse66 D1 design", "111 / 110 / 114 of 352"),
        ("D1 design without TRAIN-context flags", "110 / 105 / 109 of 352"),
        ("Clean-CAL nonconsecutive-parent errors", "1,164 / 1,034 / 1,026 of 7,936"),
        ("Large / small update pilot D1 exact; design", "135 / 122 exact; 113 / 107 design (352)"),
        ("Continuation advancement", "Neither objective nor update regime advances"),
    ]
    tables["tab:22-architecture"] = table(
        "tab:22-architecture",
        r"\textbf{Learning diagnostics and unmeasured architecture contrasts.} Matched retraining ablations remain separate from the completed continuation studies. The clean-CAL diagnostic uses all 1,885 admitted condition variants of 1,859 molecules, with coordinate counts as its denominator.",
        r">{\raggedright\arraybackslash}p{.57\textwidth}>{\raggedright\arraybackslash}p{.39\textwidth}",
        row("Contrast / readout", "Result"),
        "".join(row(*x) for x in architecture_rows),
        r"The coarse-parent objective averages categorical parent cross-entropy with a binary immediate-predecessor versus other-legal-parent loss at fixed weight one. Its eight fewer nonconsecutive-parent errors versus ordinary continuation do not establish a general architecture or final-quality improvement. All pilots retain the original model initializer; generation improvements coexist with family and diversity regressions. No checkpoint was selected using the best intermediate result.",
    )

    theory_rows = [
        (
            "Bounded representation",
            "Declared 254 atoms / 12 closures; qualified vocabulary and masks. No claim of arbitrary graph support.",
        ),
        (
            "Full-support source and endpoint population",
            "TRAIN-fitted priors and admitted records; conditional layout support remains narrower.",
        ),
        (
            "Population-optimal cross-entropy predictor",
            "Finite Transformer and updates; auxiliary losses, balanced reductions and PCGrad. Ideal optimum unestablished.",
        ),
        (
            "Exact continuous-time limit",
            "Finite capped sampling and 64-step diagnostics; no measured convergence or total-variation bound.",
        ),
        (
            "Program-fixed coordinates",
            "Restoration and equality assertions; an implementation invariant, not molecular quality.",
        ),
        (
            "Terminal prediction and readout",
            "Raw, terminal-argmax and D1 yields differ (Table~\\ref{tab:22-outcomes}); output maps are distinct.",
        ),
        (
            "Reference versus altered time law",
            "Clean-state continuation is an objective intervention; its diagnostic gains did not establish generated-quality gains.",
        ),
        (
            "Independent attempts",
            "Batch diversity/novelty selection couples requests; binomial independence is not assumed.",
        ),
        (
            "Forward replay and recursive evidence",
            "Exact L1, computational makeability and strict dossiers use separate checks and denominators.",
        ),
        ("Final released-code correspondence and remaining proof checks", pending),
    ]
    assert len(theory["rows"]) == 9
    tables["tab:22-theory"] = table(
        "tab:22-theory",
        r"\textbf{Scoped correspondence between theory and implementation.} Nine assumptions are mapped to the implementation and existing counterchecks. This audit is not a new theorem proof or an empirical guarantee of synthesis success.",
        r">{\raggedright\arraybackslash}p{.32\textwidth}>{\raggedright\arraybackslash}p{.64\textwidth}",
        row("Obligation", "Current evidence / limitation"),
        "".join(row(*x) for x in theory_rows),
    )

    guidance_rows = [
        ("22-family synthesis-guidance zero-weight identity", pending),
        ("Matched unguided/post-hoc/in-trajectory synthesis comparison", pending),
        ("Synthesis-guidance eligibility, abstention and total cost", pending),
        (
            "Separate activity-conditioning data",
            "550 HeLa/Ugi and 581 A549/aza-Michael constitutions",
        ),
        (
            "Activity fits and generation attempts",
            "27 final fits, 6 capacity comparisons; 24,192 attempts",
        ),
        ("Genuine versus hidden denoising loss reduction", r"HeLa $-1.109\%$; A549 $+0.0213\%$"),
        ("Genuine versus shuffled denoising loss reduction", r"HeLa $-0.935\%$; A549 $+0.0086\%$"),
        (
            "High-target HeLa joint versus prior-only (exact / eligible)",
            "438 / 53 versus 438 / 55, each of 576",
        ),
        (
            "High-target A549 joint versus prior-only (exact / eligible)",
            "103 / 1 versus 104 / 1, each of 576",
        ),
        ("Primary potency gain / advancement", "Not estimable / neither cell advances"),
    ]
    tables["tab:22-guidance"] = table(
        "tab:22-guidance",
        r"\textbf{Synthesis-guidance gaps and separate negative activity-conditioning evidence.} The synthesis comparison remains unrun. The completed activity experiment adapts the frozen backbone in two assay/family contexts and cannot establish all-family guidance.",
        r">{\raggedright\arraybackslash}p{.53\textwidth}>{\raggedright\arraybackslash}p{.43\textwidth}",
        row("Validation / contrast", "Result"),
        "".join(row(*x) for x in guidance_rows),
        r"Activity fits use three reused structural development folds and three seeds with genuine, hidden and shuffled labels. Shared precursors are allowed; study, cell and chemistry are confounded. None of the 64-attempt cohorts fills 16 distinct eligible selection slots, so the primary potency-gain comparison is not estimable. Exact-yield guardrails fail in one of nine HeLa and two of nine A549 blocks; diversity guards pass. Predictions are not new activity measurements, and these negative results do not qualify synthesis guidance or independent confirmation.",
    )

    for label, rendered in tables.items():
        text = replace_table(text, label, rendered)
    quality_start = r"\subsection{Structural plausibility and design-condition agreement}"
    quality_stop = r"\subsection{Upstream synthesis and terminal-evidence completeness}"
    if quality_start in text:
        a, b = text.index(quality_start), text.index(quality_stop)
        text = text[:a] + "\\input{sections/quality_current.tex}\n\n" + text[b:]
    text = text.replace(
        "Checkpoint selection uses\n\\PendingValue{checkpoint-selection rule}.",
        "The original full fit ends at its fixed 2,794-update budget. Both continuation diagnostics use their fixed final endpoints; saved intermediate checkpoints are descriptive, and no best-checkpoint selection is performed.",
    )
    text = text.replace(
        "Source rows and unique constitutional products are counted\nseparately. The family census includes sampling mass, component inventories, source-study overlap\nand exclusions;",
        "Admitted training records and qualified CAL molecules are counted separately; a complete source-row versus constitutional-product and study census remains unmeasured. The family census reports sampling mass, representation admission and exclusions;",
    )
    text = text.replace(
        "Training-seed\nvariability accompanies the family-wise estimates.",
        "Independent training-seed variability remains unmeasured; the reported development outcomes derive from one full fit.",
    )
    text = text.replace(
        "Control comparisons use paired request distributions and report differences within each training\nseed, observed ranges and uncertainty.",
        "The completed control comparisons use paired request distributions from one original fit. Independent training-seed ranges and uncertainty remain unmeasured.",
    )
    text = text.replace(
        "Training diagnostics resolve atom, bond, parent, closure and repeated-component errors by family\nand corruption time.",
        "Training artifacts retain coordinate losses; the completed guarded-CAL diagnostic resolves clean-parent errors by family and topology. Full generation learning curves across corruption time remain unmeasured.",
    )
    start = text.index(r"\begin{figure}[H]", text.index(r"\label{tab:22-training}"))
    end = text.index(r"\end{figure}", start) + len(r"\end{figure}")
    text = text[:start] + r"""\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/current_results/training_diagnostics.pdf}
\caption{\textbf{Fixed-endpoint continuation diagnostics on 352 development requests.}
The matched-presentation large/small-update pilot has D1 exact counts 135/122 and design counts
113/107, versus original 140/111. The separate categorical/coarse66 pilot gives 136/135 exact
and 110/114 design, versus the same original 140/111. These are paired continuations of one fit,
not independent training seeds; no sampling uncertainty bars or held-out claim is attached.}
\label{fig:22-training}\end{figure}""" + text[end:]
    start = text.index(
        r"\begin{figure}[H]",
        text.index(r"\subsection{Assembly, structural and synthesis-evidence outcomes}"),
    )
    end = text.index(r"\end{figure}", start) + len(r"\end{figure}")
    text = text[:start] + r"""\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/current_results/family_overview.pdf}
\caption{\textbf{Current development results across 22 reaction families.}
Exact L1, limited-design and computational-makeability counts use all 64 selected requests per
family. ECFP reference precision and CAL coverage use separate generated/reference denominators.
The selected cohort is adaptively developed from one checkpoint; independent-seed and independent
held-out confirmation remain unmeasured. Assembly consistency, limited checks, reference similarity
and route evidence are separate outcomes.}
\label{fig:22-overview}\end{figure}""" + text[end:]
    old_route = """Route assessment follows the actual generated precursor identity. Method-masked adjudication of
the component union records source identifiers and locators, source/file hashes, reactive-site and
forward checks, terminal identity and availability dates, route depths, planner calls and elapsed time.
We report E0--E3 dispositions and distinguish raw from selection-conditioned coverage. An E2
assignment requires closure of the routes supporting its generated precursors. Evidence depth is
family-specific; detailed Ugi evidence is not transferred to another reaction family. L3 denotes
documentary terminal-material qualification rather than prospective procurement or synthesis."""
    text = text.replace(
        old_route,
        """Route assessment follows the actual generated component identity. The primary computational
criterion admits checked planner-generated and published-transform paths to vendor-listed leaves;
it does not require an exact-substrate experimental precedent at every step. Strict dossiers remain
a separate secondary outcome with their original checks. Neither outcome is an experimental yield,
stock confirmation or synthesis-success probability. Route-search and listing unknowns are retained;
the available searches are not exhaustive. Detailed source evidence is not transferred between
reaction families. Changed molecules from the separate repair diagnostic require fresh route
assessment and are not substituted into Table~\\ref{tab:22-routes}.""",
    )
    text = text.replace(
        "The atlas uses a fixed selection seed across all families. Atom origins, component novelty and L1/L2/L3 evidence are annotated separately. Random samples and targeted failure examples are distinguished.",
        "A prespecified all-family atlas remains unmeasured. Existing targeted failure galleries are developmental diagnostics, not a random or method-masked quality sample. Atom origins, component novelty and L1/L2/L3 evidence require separate annotations.",
    )
    text = text.replace(
        "Family-wise estimates retain their own denominators, comparator budgets and\ntraining-seed uncertainty.",
        "Family-wise estimates retain their own denominators and comparator budgets; independent training-seed uncertainty remains unmeasured.",
    )
    (PAPER / "sections/appendix_22.tex").write_text(text)
    evidence = {
        "schema": "forge.current_appendix_evidence.v1",
        "scope": "Documentary rendering of frozen development results; no new model, chemistry or TEST calls.",
        "producer": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "inputs": INPUTS,
        "computed_from_saved_records": {
            "census": census_rows,
            "attempt_outcomes_352": outcomes,
            "repair192": repair_totals,
        },
        "table_labels": list(tables),
        "delegated_quality_table_count": 7,
        "preserved_total_appendix_tables": 17,
        "populations": {
            "independent_full_fits": 1,
            "development_readout_requests": 352,
            "selected_route_requests": 1408,
            "repair_subset_requests": 192,
            "CAL_molecules": 1859,
            "CAL_condition_variants": 1885,
        },
        "routing_totals": treatment["totals"],
        "unmeasured": [
            "independent final fits and held-out outcomes",
            "trained true-null performance",
            "matched architecture and external baseline suite",
            "22-family synthesis-guidance contrast",
            "independent realism and study/component-disjoint confirmation",
        ],
        "output": {
            "path": "paper/v1_iclr22/sections/appendix_22.tex",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }
    (PAPER / "current_appendix_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "tables": len(tables),
                "inputs": len(INPUTS),
                "route_requests": 1408,
                "diagnostic_requests": 352,
            }
        )
    )


if __name__ == "__main__":
    main()
