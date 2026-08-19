#!/usr/bin/env python3
"""Vendor hash-pinned source assets into data/vendor/.

Local sources live outside this repo (see docs/DATA_PROVENANCE.md). Canonical public sources may
also be fetched from an immutable upstream revision. This script copies or fetches exact bytes and
records a manifest. `--verify` re-hashes what is already vendored and changes nothing.

Never fabricate, synthesize, or download substitute data. Every source must match its declared hash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "data" / "vendor"
MANIFEST = VENDOR / "MANIFEST.json"

# The originating workstation's absolute paths are gone. Everything that used to live under
# $COMPOSE is recoverable from the COMPOSE repository's git history instead: the files were
# committed there before being dropped from its working tree, so the exact pinned bytes are
# still in its object store. Override any root with the matching environment variable.
COMPOSE_REPO = Path(os.environ.get("FORGE_COMPOSE_REPO", Path.home() / "Kosha/compose_rgm"))
COMBINATORIAL = Path(
    os.environ.get(
        "FORGE_COMBINATORIAL_DIR",
        "/Users/rmaganti/Desktop/thesis_projects_ML/combinatorial_papers",
    )
)
REACTION_DATASETS = Path(
    os.environ.get(
        "FORGE_REACTION_DATASETS_DIR",
        "/Users/rmaganti/Desktop/thesis_projects_ML/reaction_datasets",
    )
)

# vendored_name -> (git revision, path within COMPOSE_REPO, expected_sha256)
# The revision is a hint, not a requirement: if it does not resolve, or resolves to different
# bytes, we scan the repository's history for a blob matching the pinned hash. Recovery is by
# content, never by name, so a renamed or relocated file is still found.
# NOTE the rename: r1_reaction_grounded_corpus_v1 -> r1_reaction_enumerated_support_v1.
# The original name asserts route-certification the data does not have (PLAN section 5).
GIT_ASSETS: dict[str, tuple[str, str, str]] = {
    "r0_observed_real_structures.csv": (
        "5088053",
        "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv",
        "3e2a35416c98f44e2261c06717c1fdee1b5ea393fb8013a38235631aa5ee6c0d",
    ),
    "r1_reaction_enumerated_support_v1.csv": (
        "6985bbe7facf",
        "artifacts/datasets/compose_lipid_pretraining_v1/r1_reaction_grounded_corpus_v1.csv",
        "8a691e47e24a921233a05627094627e54f1a5a404e998c11ee69a51716d62f3a",
    ),
    "training_corpus_manifest_v1.json": (
        "6985bbe7facf",
        "artifacts/datasets/compose_lipid_pretraining_v1/training_corpus_manifest_v1.json",
        "03eafbbeac5d63b95df4cc488f884fef1c8e9eae0a7d13a1ba9a3df1005c4a39",
    ),
    "corpus_fold_assignments.csv": (
        "56e8706bf2ea",
        "artifacts/datasets/compose_lipid_pretraining_v1/splits_v1/corpus_fold_assignments.csv",
        "92a39758bb35daf3590a83522dee31541b2586e1d18eef64a5cb403a7e50d369",
    ),
    "splits_manifest.json": (
        "56e8706bf2ea",
        "artifacts/datasets/compose_lipid_pretraining_v1/splits_v1/manifest.json",
        "51d2f714886cbed0982a47e7c8c78ad49ebe91383c80cf97ec7bfee0b582322a",
    ),
    "qualified_reaction_families_v1.json": (
        "aed2bb6f009e",
        "configs/lipid_reactions/qualified_reaction_families_v1.json",
        "961dea191bbf6c8d169aab09f6aee18082e51696e4056fc4dc9f6fc97ff82587",
    ),
    "qualified_reactions_v1.json": (
        "5dd44c449c6f",
        "configs/lipid_reactions/qualified_reactions_v1.json",
        "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf",
    ),
    "ugi_3cr_building_blocks_v1.json": (
        "5dd44c449c6f",
        "configs/lipid_reactions/ugi_3cr_building_blocks_v1.json",
        "4ba3fec67b904fe507eec25215ce0c31a62fdc5b9536a3b33a5994acafc48f81",
    ),
    "building_block_pool_v1.json": (
        "cf01527c0233",
        "configs/lipid_reactions/building_block_pool_v1.json",
        "7ad9699364f3bffe09019751e9437df5122becf0f4f6547f500fc849bbf08d18",
    ),
}

# vendored_name -> (source_path, expected_sha256)
# What remains here is genuinely local: Nitya's bench records and the two USPTO archives.
ASSETS: dict[str, tuple[Path, str]] = {
    "uspto_50k.csv": (
        REACTION_DATASETS / "USPTO_50K.csv",
        "1d69b90a299bd255b00342ccd15c3bc11d6c3047a3fecefaec539c3d04168dc9",
    ),
    "uspto_mit_data.zip": (
        REACTION_DATASETS / "uspto_mit_data.zip",
        "6d94a136e11f76fe464430cb95d1ae6db37b6ca352161ca4edddd9e6fe76a88a",
    ),
    "rm_006_tail_1a.docx": (
        COMBINATORIAL / "RM_006_Tail_1a.docx",
        "f39b98e609328ff5fa3a6fe6506fca39f18b53cedcb0afa527921afce261c134",
    ),
    "rm_007_tail_1b.docx": (
        COMBINATORIAL / "RM_007_Tail_1b.docx",
        "11d7f5753c7b1c1a8d02148d86e3f084c71158d4152685f9012c5a1ad9b21637",
    ),
    "rm_008_tail_1c.docx": (
        COMBINATORIAL / "RM_008_Tail_1c.docx",
        "6b222e7bb2df23840cdede744b4754b61a99dabc6fe1e99e1b51b4a6c1f70959",
    ),
    "rm_009_tail_1d.docx": (
        COMBINATORIAL / "RM_009_Tail_1d.docx",
        "ad8a3f153470e0b3b945c06213e0aab0bbaa0589d56c7aa05ac5fc1f822d15a5",
    ),
    "rm_016_tail2b_corrected.docx": (
        COMBINATORIAL / "RM_016_Tail2b_corrected.docx",
        "0a4f14cbae53c02921fc237e54fd7b399ce0e92204b22af1973c1c8c2065f45f",
    ),
    "rm_066_tail2a_bf3oet2.docx": (
        COMBINATORIAL / "RM_066_Tail2a_BF₃OEt₂.docx",
        "f5fa383b1484e2c01512cf5664783d49ba64dbf3d89084fa45c518877058781c",
    ),
    "rm_067_tail2c_bf3oet2.docx": (
        COMBINATORIAL / "RM_067_Tail2c_BF₃OEt₂.docx",
        "94d52884b8cf74b59d02b6678e53d9c73bafabc30865f1ca5e5889650078e9b0",
    ),
}

REMOTE_ASSETS: dict[str, tuple[str, str]] = {
    # Both AGILE assets were originally recorded as local-only, but the published article and
    # the authors' repository serve the identical bytes. Verified against the pinned hashes.
    "AGILE_smiles_with_value_group.csv": (
        "https://raw.githubusercontent.com/bowang-lab/AGILE/main/AGILE_smiles_with_value_group.csv",
        "1b3dd460125ba7d70d8cce266bd5febaabe09eeddc6a4622405c2706420b9686",
    ),
    "agile_supplementary_information.pdf": (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41467-024-50619-z/MediaObjects/"
        "41467_2024_50619_MOESM1_ESM.pdf",
        "ab21e9fa8ad4f2d991c97298c64232eec9942bce6b95f82419bfef67f5c830e9",
    ),
    "lnpdb_fc7c389.csv": (
        "https://raw.githubusercontent.com/evancollins1/LNPDB/"
        "fc7c38933b445eb54985014f2b8917606462eec1/"
        "data/LNPDB_for_LiON/LNPDB.csv",
        "3493f27306419facd0958589030ed37f272f05c81ad47dd7bc1b8ce0284a57b3",
    ),
    "agile_article_source_data.xlsx": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fs41467-024-50619-z/MediaObjects/"
        "41467_2024_50619_MOESM4_ESM.xlsx",
        "e795d4e7ef1ec5c82b0358c626ab1eae059b76485a3ebdb7cbe5565aa6aac723",
    ),
    "lantern_agile_curated.csv": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/data/AGILE.csv",
        "d0ecfd94ec40a7f013af546113520175e70f78045af1798673c2680fe6f23c54",
    ),
    "lantern_agile_cis_trans_audit.csv": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/data/CisTrans.csv",
        "eceea1dcfc86063d0129d2b25918a17f5a0493bd3601641719f8d6ee29dff44c",
    ),
    "lantern_agile_random_split.npy": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "data/splits/AGILE/random.npy",
        "72d7cb6d5769c8b45f6634c530547c51147532539cfa2e728a7d630f24d699e8",
    ),
    "lantern_agile_murcko_split.npy": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "data/splits/AGILE/Murcko_scaffold.npy",
        "cd52f769972cde1e513a889ebfce0764017c56cd17032e3f9750b76f6e59632e",
    ),
    "lantern_agile_scaffold_balanced_split.npy": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "data/splits/AGILE/scaffold_balanced.npy",
        "1f3720b6fc8ce2dfc3cf71f3e62cd394c74058d25641bf009eb7ddb146493c01",
    ),
    "lantern_agile_circular.pkl": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "data/fingerprints/AGILE/circular.pkl",
        "3a1331eeba6d16ccc9b12e53f68f957a1726a4b3c6ef4944cf97b6e353a0d619",
    ),
    "lantern_agile_expert.pkl": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "data/fingerprints/AGILE/expert.pkl",
        "23ba1fceaec5f10480ff7d14734d1ec70901f2e7680c129d88ee64405f9285a7",
    ),
    "lantern_agile_mlp.pth": (
        "https://raw.githubusercontent.com/AsalMehradfar/LANTERN/"
        "11240f29ef92323649ae60d177b21df77e2d428b/"
        "checkpoints/circular-expert-model-MLP.pth",
        "f419bfce435f1a08750b5d06902de9941b23343aa90d55aaefd6b3b2fa703382",
    ),
}

# Assets not required by tasks other than the M0-04 baseline control. Allows a ~22 MB
# partial vendor in environments where the 96 MB R1 file cannot be transferred.
#
# The USPTO archives and the bench records join it: both are still pinned above and must match
# their hashes if present, but neither backs a claim in the manuscript. The USPTO corpora are
# needed only to retrain the single-step route engines, whose benchmark scores are already
# frozen artifacts, and the bench records are source documents for M0-09 route reviews whose
# conclusions are likewise recorded under results/m0_09/. Only the originating workstation has
# them, so requiring them makes `make vendor` unrunnable anywhere else.
OPTIONAL = {
    "r1_reaction_enumerated_support_v1.csv",
    "uspto_50k.csv",
    "uspto_mit_data.zip",
    "rm_006_tail_1a.docx",
    "rm_007_tail_1b.docx",
    "rm_008_tail_1c.docx",
    "rm_009_tail_1d.docx",
    "rm_016_tail2b_corrected.docx",
    "rm_066_tail2a_bf3oet2.docx",
    "rm_067_tail2c_bf3oet2.docx",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)


def git_blob(repo: Path, rev: str, path: str, expected: str) -> bytes | None:
    """Return the pinned bytes from a git repository, or None if they are not there.

    The recorded revision is tried first. If it is missing or its content has drifted, every
    revision that touched the path is searched for a blob matching the expected hash. Bytes are
    returned only on an exact hash match, so this can recover a file but never substitute one.
    """
    if not (repo / ".git").exists():
        return None
    if rev:
        found = _git(repo, "cat-file", "-p", f"{rev}:{path}")
        if found.returncode == 0 and hashlib.sha256(found.stdout).hexdigest() == expected:
            return found.stdout
    listed = _git(repo, "log", "--all", "--format=%H", "--", path)
    if listed.returncode != 0:
        return None
    for candidate in listed.stdout.decode().split():
        found = _git(repo, "cat-file", "-p", f"{candidate}:{path}")
        if found.returncode == 0 and hashlib.sha256(found.stdout).hexdigest() == expected:
            return found.stdout
    return None


def do_vendor(allow_partial: bool, refresh_provenance: bool = False) -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    missing, entries = [], {}

    # MANIFEST.json is itself a pinned input of results/m0_02/result.json, so rewriting it on a
    # re-vendor breaks that pin even when every asset is byte-identical. Entries are therefore
    # reused verbatim whenever the asset still hashes to what the manifest already records; only
    # a genuine content change, or an explicit --refresh-provenance, rewrites one.
    prior: dict[str, dict[str, Any]] = {}
    if MANIFEST.exists():
        try:
            prior = json.loads(MANIFEST.read_text())
        except ValueError:
            prior = {}

    def entry(name: str, digest: str, size: int, source: dict[str, str]) -> dict[str, Any]:
        recorded = prior.get(name)
        if not refresh_provenance and recorded and recorded.get("sha256") == digest:
            return recorded
        return {
            **source,
            "sha256": digest,
            "bytes": size,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }

    for name, (rev, path, expected) in GIT_ASSETS.items():
        blob = git_blob(COMPOSE_REPO, rev, path, expected)
        if blob is None:
            missing.append((name, f"{COMPOSE_REPO}@{rev or 'any'}:{path}", expected))
            continue
        (VENDOR / name).write_bytes(blob)
        entries[name] = entry(
            name, expected, len(blob), {"source_git": f"{COMPOSE_REPO}@{rev}:{path}"}
        )
        print(f"  vendored {name}  ({len(blob):,} bytes, from git)")

    for name, (src, expected) in ASSETS.items():
        if not src.exists():
            missing.append((name, src, expected))
            continue
        actual = sha256(src)
        if actual != expected:
            print(f"FAIL {name}: source hash mismatch\n  expected {expected}\n  actual   {actual}")
            return 1
        shutil.copy2(src, VENDOR / name)
        entries[name] = entry(name, actual, src.stat().st_size, {"source_path": str(src)})
        print(f"  vendored {name}  ({src.stat().st_size:,} bytes)")

    for name, (url, expected) in REMOTE_ASSETS.items():
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(
                    "wb",
                    dir=VENDOR,
                    prefix=f".{name}.",
                    delete=False,
                ) as handle:
                    shutil.copyfileobj(response, handle)
                    temporary_path = Path(handle.name)
        except OSError as exc:
            print(f"FAIL {name}: canonical source fetch failed\n  url {url}\n  error {exc}")
            return 1
        actual = sha256(temporary_path)
        if actual != expected:
            temporary_path.unlink(missing_ok=True)
            print(f"FAIL {name}: source hash mismatch\n  expected {expected}\n  actual   {actual}")
            return 1
        target = VENDOR / name
        temporary_path.replace(target)
        entries[name] = entry(name, actual, target.stat().st_size, {"source_url": url})
        print(f"  vendored {name}  ({target.stat().st_size:,} bytes)")

    if missing:
        blocking = [m for m in missing if m[0] not in OPTIONAL or not allow_partial]
        print("\nMISSING SOURCE ASSETS:")
        for name, src, expected in missing:
            tag = "optional" if name in OPTIONAL else "REQUIRED"
            print(f"  [{tag}] {name}\n      path {src}\n      sha256 {expected}")
        if blocking:
            print(
                "\nDo not fabricate, synthesize, or download substitute data.\n"
                "See docs/DATA_PROVENANCE.md for options. Stopping."
            )
            return 1
        print("\nProceeding with partial vendor (--allow-partial).")

    # A partial run must not erase the record of what it could not fetch. Entries for assets
    # that were skipped this time are carried forward from the existing manifest, so the pinned
    # hash of an asset stays documented even on a machine that has never held it.
    carried = 0
    if MANIFEST.exists():
        for name, meta in json.loads(MANIFEST.read_text()).items():
            if name not in entries:
                entries[name] = meta
                carried += 1

    # MANIFEST.json is a pinned input of results/m0_02/result.json, and the pinned bytes were
    # serialized with a key order this writer no longer reproduces. Rewriting a semantically
    # identical file would therefore break that pin for no gain, so an unchanged manifest is
    # left untouched. Only a real content change is written.
    if prior == entries:
        print(f"\n{MANIFEST.relative_to(REPO)} unchanged ({len(entries)} assets); left as-is")
        return 0

    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    suffix = f" ({carried} carried forward from the previous manifest)" if carried else ""
    print(f"\nwrote {MANIFEST.relative_to(REPO)} with {len(entries)} assets{suffix}")
    return 0


def do_verify(allow_partial: bool = False) -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST} — run `make vendor` first")
        return 1
    entries = json.loads(MANIFEST.read_text())
    bad, skipped = 0, 0
    for name, meta in sorted(entries.items()):
        path = VENDOR / name
        if not path.exists():
            # A hash mismatch is always fatal. A merely absent optional asset is not, so long
            # as the caller asked for a partial check: its pin stays in the manifest either way.
            if allow_partial and name in OPTIONAL:
                print(f"  -- {name}: absent (optional)")
                skipped += 1
                continue
            print(f"FAIL {name}: vendored file missing")
            bad += 1
            continue
        actual = sha256(path)
        if actual != meta["sha256"]:
            print(f"FAIL {name}: hash mismatch\n  expected {meta['sha256']}\n  actual   {actual}")
            bad += 1
        else:
            print(f"  ok {name}")
    if bad:
        print(f"\n{bad} asset(s) failed verification")
        return 1
    tail = f", {skipped} optional asset(s) absent" if skipped else ""
    print(f"\nall {len(entries) - skipped} present vendored assets verified{tail}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="verify only; do not copy")
    ap.add_argument(
        "--refresh-provenance",
        action="store_true",
        help="rewrite source and timestamp fields even when an asset is unchanged",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit missing optional assets (the 96 MB R1 file)",
    )
    args = ap.parse_args()
    return (
        do_verify(args.allow_partial)
        if args.verify
        else do_vendor(args.allow_partial, args.refresh_provenance)
    )


if __name__ == "__main__":
    sys.exit(main())
