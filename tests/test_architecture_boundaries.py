"""Executable dependency rules for packages already moved onto the target architecture."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "forge"

ALLOWED_PACKAGE_IMPORTS = {
    "core": {"core"},
    "chem": {"chem", "core"},
    "assembly": {"assembly", "chem", "core"},
    "bio": {"bio", "core"},
    "corpus": {"corpus", "chem", "core"},
    "generate": {"generate", "assembly", "corpus", "chem", "core"},
}

# Exact migration shims.  They preserve frozen implementation paths while all new callers use the
# target-domain API.  This list may shrink; adding an entry requires an architectural review.
TRANSITIONAL_IMPORTS = {
    ("assembly/ugi3.py", "forge.data.r1_prime_audit"),
    ("assembly/ugi3.py", "forge.design.corpus.ugi_held_component_gate"),
    ("corpus/phase1.py", "forge.design.corpus.phase1_data"),
}


def _forge_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("forge."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("forge."):
            imports.add(node.module)
    return imports


def test_migrated_packages_follow_the_dependency_direction() -> None:
    violations: list[str] = []
    for package, allowed in ALLOWED_PACKAGE_IMPORTS.items():
        for path in sorted((SOURCE / package).rglob("*.py")):
            relative = path.relative_to(SOURCE).as_posix()
            for imported in sorted(_forge_imports(path)):
                target_package = imported.split(".", 2)[1]
                if (
                    target_package not in allowed
                    and (relative, imported) not in TRANSITIONAL_IMPORTS
                ):
                    violations.append(f"{relative} -> {imported}")
    assert not violations, "dependency boundary violations:\n" + "\n".join(violations)


def test_domain_code_never_imports_the_experiment_runner() -> None:
    violations = []
    for path in sorted(SOURCE.rglob("*.py")):
        relative = path.relative_to(SOURCE)
        if relative.parts[0] == "experiment" or relative == Path("cli.py"):
            continue
        for imported in _forge_imports(path):
            if imported == "forge.experiment" or imported.startswith("forge.experiment."):
                violations.append(f"{relative.as_posix()} -> {imported}")
    assert not violations, "orchestration leaked into domain code:\n" + "\n".join(violations)


def test_bio_to_potency_move_is_complete_and_has_no_stale_runtime_imports() -> None:
    """Historical pins keep old paths; importable code must use the new namespace."""

    move_document = json.loads((REPO / "docs/artifact_path_moves.json").read_text())
    moves = {
        old: new for old, new in move_document["moves"].items() if old.startswith("src/forge/bio/")
    }
    assert moves, "the bio-to-potency migration has no declared provenance moves"

    moved_modules = {
        Path(old).with_suffix("").as_posix().removeprefix("src/").replace("/", ".") for old in moves
    }
    violations: list[str] = []
    for old, new in sorted(moves.items()):
        if (REPO / old).exists():
            violations.append(f"old implementation still exists: {old}")
        if not (REPO / new).is_file():
            violations.append(f"moved implementation is missing: {new}")
        if not new.startswith("src/forge/potency/"):
            violations.append(f"bio implementation moved outside potency: {old} -> {new}")

    for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            for imported in _forge_imports(path):
                if any(
                    imported == module or imported.startswith(f"{module}.")
                    for module in moved_modules
                ):
                    violations.append(
                        f"{path.relative_to(REPO).as_posix()} imports removed {imported}"
                    )

    assert not violations, "incomplete bio-to-potency migration:\n" + "\n".join(violations)


def test_bio_contains_only_endpoint_domain_modules() -> None:
    modules = {path.name for path in (SOURCE / "bio").glob("*.py")}
    assert modules == {
        "__init__.py",
        "endpoint.py",
        "endpoint_decision.py",
        "liver.py",
        "muscle.py",
        "vaccine.py",
    }


def test_no_package_directory_survives_only_as_a_cache() -> None:
    """A deleted package must not come back as an empty namespace package.

    Removing a package leaves `__pycache__` behind, and `rmdir` then fails silently. Python treats
    the surviving directory as a namespace package, so `import forge.<name>` keeps succeeding and
    returns nothing -- the deletion looks done and is not. This caught `forge.maintenance` and
    `forge.dossier` after they were moved out, both still importable with no modules in them.
    """
    empty = [
        directory.relative_to(REPO).as_posix()
        for directory in SOURCE.iterdir()
        if directory.is_dir()
        and directory.name != "__pycache__"
        and not any(directory.rglob("*.py"))
    ]
    assert not empty, "package directories with no Python modules:\n" + "\n".join(empty)


def test_every_package_declares_what_it_is() -> None:
    """No package may carry the placeholder docstring the original layout generated.

    `\"\"\"FORGE <x> module — see docs/M0_TASKS.md.\"\"\"` says nothing, and the packages still
    carrying it are exactly the ones the restructure has not reached. Declaring a real surface is
    what lets ALLOWED_PACKAGE_IMPORTS above cover a package at all, so this list shrinking is the
    ratchet -- entries may be removed, never added.
    """
    undeclared = {"data", "product", "route"}
    stubs = set()
    for init in SOURCE.glob("*/__init__.py"):
        text = init.read_text()
        if "see docs/M0_TASKS.md" in text:
            stubs.add(init.parent.name)
    assert stubs <= undeclared, f"new placeholder package docstring: {sorted(stubs - undeclared)}"
