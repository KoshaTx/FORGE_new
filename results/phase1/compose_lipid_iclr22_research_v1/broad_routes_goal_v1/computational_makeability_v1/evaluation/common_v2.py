"""Local, hash-authenticated inputs for the separate makeability evaluation."""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[6]
GOAL = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1"
OUT = Path(__file__).resolve().parent
V5 = GOAL / "planner/combined_v5"


def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}


def physical_path(location):
    path = Path(location.split("#", 1)[0])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"nonlocal/unadmitted evidence path: {location}")
    return path


@cache
def _authenticate_physical(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"evidence digest mismatch: {path}")
    return data


def authenticate(location, digest):
    return _authenticate_physical(physical_path(location), digest)


@cache
def _read_json(path, digest):
    data = _authenticate_physical(path, digest)
    return json.loads(gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data)


def read_pin(value):
    path = physical_path(value.get("path", value.get("location")))
    return _read_json(path, value["sha256"])


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def canonical(value):
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError(f"invalid supplied identity: {value}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
