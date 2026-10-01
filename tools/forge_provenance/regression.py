"""Compare evidence availability with a base checkout without accepting old gaps as verified.

Both trees use the same verifier. No count allowance or editable exception ledger can mask a
new loss: declarations, available identities and archived identities are compared individually.
The existing absolute audit (``make verify-pins``) remains unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge.core.io import write_json
from forge_provenance import pins
from forge_provenance.resolver import HistoricalPinArchive

Identity = tuple[str, str]
ROOTS = ("results", "docs/provenance", "configs", "experiments")


@dataclass
class Snapshot:
    declarations: set[pins.Pin]
    available: set[Identity]
    archived: set[Identity]
    unresolved: set[pins.Pin]
    drift: set[pins.Pin]
    foreign: set[pins.Pin]


def snapshot(repo: Path) -> Snapshot:
    repo = repo.resolve()
    if not (repo / "results").is_dir() or not (repo / "configs").is_dir():
        raise ValueError(f"expected a FORGE checkout with results/ and configs/: {repo}")
    roots = tuple(repo / name for name in ROOTS)
    # The legacy collector skips unreadable JSON. This gate must not let a corrupt declaration
    # disappear silently, including newly added files that were not present in the base tree.
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz")):
                if path.name.endswith(".json.gz"):
                    with gzip.open(path, "rt") as handle:
                        json.load(handle)
                else:
                    json.loads(path.read_text())
    archive = HistoricalPinArchive.load(repo / "provenance/frozen-code/manifest.json", repo)
    # Validate even archive entries no longer reached from a current declaration.
    for original_path, digest in archive.entries:
        archive.resolve(original_path, digest)
    original_repo = pins.REPO
    try:
        pins.REPO = repo
        declarations = pins.collect_pins(roots)
        report = pins.verify(
            declarations,
            moves=pins.load_moves(repo / "docs/artifact_path_moves.json"),
            archive=archive,
        )
    finally:
        pins.REPO = original_repo
    drift = {pin for pin, _ in report.drift}
    return Snapshot(
        declarations=set(declarations),
        available={(pin.path, pin.sha256) for pin in report.verified + report.archived},
        archived=set(archive.entries),
        unresolved=set(report.absent) | drift,
        drift=drift,
        foreign=set(report.foreign),
    )


def pin_rows(values: set[pins.Pin]) -> list[dict[str, str]]:
    return [asdict(pin) for pin in sorted(values, key=lambda p: (p.path, p.sha256, p.declared_by))]


def identity_rows(values: set[Identity]) -> list[dict[str, str]]:
    return [{"path": path, "sha256": digest} for path, digest in sorted(values)]


def compare(base: Snapshot, current: Snapshot) -> dict[str, Any]:
    regressions = {
        "removed_declarations": pin_rows(base.declarations - current.declarations),
        "lost_available_identities": identity_rows(base.available - current.available),
        "removed_archive_identities": identity_rows(base.archived - current.archived),
        "new_unresolved_declarations": pin_rows(current.unresolved - base.unresolved),
    }
    return {
        "schema_version": "forge.provenance_regression.v1",
        "regression_status": "fail" if any(regressions.values()) else "pass",
        "historical_availability": "incomplete" if current.unresolved else "available",
        "scope": list(ROOTS),
        "limitations": [
            "Availability and byte identity do not establish numerical reproduction.",
            "Container-side and external paths are reported but cannot be verified here.",
            "The inventory includes maintenance records as well as experiment declarations.",
            "Existing missing or changed identities remain unresolved, not accepted as verified.",
        ],
        "counts": {
            "base_available_identities": len(base.available),
            "current_available_identities": len(current.available),
            "base_unresolved_declarations": len(base.unresolved),
            "current_unresolved_declarations": len(current.unresolved),
            "current_drift_declarations": len(current.drift),
            "current_foreign_declarations": len(current.foreign),
        },
        "regressions": regressions,
        "recovered_identities": identity_rows(current.available - base.available),
        "unresolved_declarations": pin_rows(current.unresolved),
        "foreign_declarations": pin_rows(current.foreign),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=pins.REPO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.base_root.resolve() == args.current_root.resolve():
        parser.error("base and current must be different checkouts")
    report = compare(snapshot(args.base_root), snapshot(args.current_root))
    write_json(args.output, report)
    counts = report["counts"]
    message = (
        f"Evidence regression check: {report['regression_status'].upper()}\n\n"
        f"Historical availability: {report['historical_availability'].upper()}\n\n"
        f"{counts['current_unresolved_declarations']} unresolved local declarations; "
        f"{counts['current_foreign_declarations']} external/container declarations unverified.\n\n"
        "A regression pass means no new evidence loss relative to the base checkout. "
        "It is not a historical reproduction pass. See the JSON artifact for every gap.\n"
    )
    print(message)
    for category, rows in report["regressions"].items():
        if rows:
            print(f"{category}: {len(rows)}")
            for row in rows[:10]:
                print(json.dumps(row, sort_keys=True))
    if args.summary:
        with args.summary.open("a") as handle:
            handle.write(message)
    return 1 if report["regression_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
