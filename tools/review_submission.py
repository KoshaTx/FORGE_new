"""Offline submission integrity/arithmetic checks; never load weights or launch experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("paper/submission/manifest.json")
GENERATED = Path("paper/v1_iclr/generated")
PROGRAMS = ("Ugi", "Aza-Michael", "Reductive amination")


def inspect_pin(repo: Path, pin: dict[str, str]) -> dict[str, str]:
    """Check a repo-relative pin without deserializing its payload."""
    relative = Path(pin["path"])
    expected = pin["sha256"]
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    path = repo / relative
    if not path.resolve().is_relative_to(repo.resolve()):
        raise ValueError(f"artifact escapes repository: {relative}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"invalid SHA-256: {relative}")
    result = {"path": relative.as_posix(), "expected_sha256": expected}
    if not path.is_file():
        return {**result, "status": "missing"}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    return {
        **result,
        "actual_sha256": actual,
        "status": "verified" if actual == expected else "mismatch",
    }


def check_conditioned_counts(repo: Path) -> list[dict[str, Any]]:
    """Cross-check Table 4 denominators/seed counts against conditioned Table 1."""
    rows: dict[str, dict[int, tuple[int, int]]] = {name: {} for name in PROGRAMS}
    text = (repo / GENERATED / "production_seed_exact_counts_rows.tex").read_text()
    for line in text.splitlines():
        if not line.strip() or line.strip() == r"\hline":
            continue
        cells = [cell.strip() for cell in line.removesuffix(r"\\").split("&")]
        if len(cells) != 5 or cells[0] not in rows:
            raise ValueError("unexpected Table 4 row")
        program, seed_text, count_text, attempts_text, percent_text = cells
        seed = int(seed_text)
        count = int(count_text.replace("{,}", ""))
        attempts = int(attempts_text.replace("{,}", ""))
        if seed not in (0, 1, 2) or seed in rows[program]:
            raise ValueError(f"duplicate or invalid seed for {program}")
        if attempts != 3072 or not 0 <= count <= attempts:
            raise ValueError(f"invalid attempt denominator/count for {program}")
        if f"{100 * count / attempts:.2f}" != percent_text:
            raise ValueError(f"Table 4 percentage disagrees with count for {program}")
        rows[program][seed] = (count, attempts)

    summary: dict[str, str] = {}
    for line in (repo / GENERATED / "shared_program_figure_rows.tex").read_text().splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split("&")]
        if len(cells) != 5 or cells[0] not in rows or cells[0] in summary:
            raise ValueError("unexpected or duplicate Table 1 row")
        summary[cells[0]] = cells[1]
    if set(summary) != set(PROGRAMS):
        raise ValueError("Table 1 must contain all three programs")

    reports = []
    for program, seeds in rows.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError(f"Table 4 must contain three independent seeds for {program}")
        yields = [100 * count / attempts for count, attempts in seeds.values()]
        mean, sd = statistics.mean(yields), statistics.stdev(yields)
        expected = f"${mean:.1f}\\pm{sd:.1f}$"
        if summary[program] != expected:
            raise ValueError(f"Table 1 conditioned mean/sample SD disagrees for {program}")
        reports.append({"program": program, "mean_percent": mean, "sample_sd_percent": sd})
    return reports


def audit(repo: Path, *, require_run_artifacts: bool = False) -> dict[str, Any]:
    """Report asset integrity and the explicitly scoped historical-input inventory."""
    manifest = json.loads((repo / MANIFEST).read_text())
    if manifest.get("schema_version") != "forge.submission_review.v1":
        raise ValueError("unsupported submission manifest schema")
    groups = {}
    for name in ("review_assets", "historical_inputs"):
        pins = manifest.get(name)
        if not isinstance(pins, list) or not pins:
            raise ValueError(f"manifest must list {name}")
        paths = [pin["path"] for pin in pins]
        if len(set(paths)) != len(paths):
            raise ValueError(f"duplicate path in {name}")
        groups[name] = [inspect_pin(repo, pin) for pin in pins]
    assets_ok = all(row["status"] == "verified" for row in groups["review_assets"])
    inputs_ok = all(row["status"] == "verified" for row in groups["historical_inputs"])
    input_mismatch = any(row["status"] == "mismatch" for row in groups["historical_inputs"])
    errors = []
    arithmetic = []
    try:
        arithmetic = check_conditioned_counts(repo)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    passed = assets_ok and not input_mismatch and not errors
    if require_run_artifacts and not inputs_ok:
        passed = False
    return {
        "schema_version": "forge.submission_review_report.v1",
        "status": "pass" if passed else "fail",
        "scope": "asset identity and Table 4-to-Table 1 conditioned arithmetic; no model rerun",
        "full_reproduction_verified": False,
        "historical_inventory_complete": inputs_ok,
        "historical_inventory_scope": manifest["historical_inventory_scope"],
        "require_run_artifacts": require_run_artifacts,
        **groups,
        "conditioned_arithmetic": arithmetic,
        "errors": errors,
        "limitations": manifest["limitations"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print all paths, hashes and outcomes")
    parser.add_argument(
        "--require-run-artifacts",
        action="store_true",
        help="also fail if any indexed historical input is absent; not a full rerun",
    )
    args = parser.parse_args(argv)
    try:
        report = audit(ROOT, require_run_artifacts=args.require_run_artifacts)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "fail", "errors": [str(exc)]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Submission review check: {report['status'].upper()}")
        for group in ("review_assets", "historical_inputs"):
            rows = report.get(group, [])
            verified = sum(row["status"] == "verified" for row in rows)
            print(f"  {group}: {verified}/{len(rows)} hash-verified")
            for row in rows:
                if row["status"] == "mismatch" or (
                    group == "review_assets" and row["status"] == "missing"
                ):
                    print(f"    {row['status']}: {row['path']}")
        for row in report.get("conditioned_arithmetic", []):
            print(
                f"  {row['program']}: {row['mean_percent']:.1f} ± "
                f"{row['sample_sd_percent']:.1f}% (three seeds)"
            )
        for error in report["errors"]:
            print(f"  ERROR: {error}")
        print("No training/inference rerun. Full reproduction remains unverified.")
        print("Use --json for missing historical paths/hashes; see paper/submission/ARTIFACTS.md.")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
