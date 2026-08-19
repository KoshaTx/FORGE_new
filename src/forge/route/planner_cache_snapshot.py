"""Immutable snapshots and isolated overlays for :mod:`planner_cache`.

This additive layer preserves the hash-frozen ``FilePlannerCache`` implementation
while providing byte-owned cache states for matched diagnostic experiments.  It
does not define a synthesis value, run guidance, or select candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.route.planner import RouteTarget, SynthesisAssessment
from forge.route.planner_cache import (
    PLANNER_CACHE_ENTRY_SCHEMA_VERSION,
    PLANNER_CACHE_KEY_SCHEMA_VERSION,
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
    PlannerCacheKey,
)

PLANNER_CACHE_SNAPSHOT_SCHEMA_VERSION = "forge.planner_cache_snapshot.v1"
PLANNER_CACHE_OVERLAY_AUDIT_SCHEMA_VERSION = "forge.planner_cache_overlay_audit.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CACHE_RELATIVE_PATH_PATTERN = re.compile(r"^[0-9a-f]{2}/[0-9a-f]{64}\.json$")


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannerCacheError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise PlannerCacheError(f"{label} must contain 64 lowercase hexadecimal characters")
    return value


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlannerCacheError(f"{label} must be a nonnegative integer")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    _require_nonempty(value, label=label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlannerCacheError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PlannerCacheError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def require_current_l3_context(
    context: PlannerCacheContext,
    *,
    assessment_at_utc: str,
) -> None:
    """Fail unless the frozen L3 evidence is current at an explicit UTC time."""

    assessment_at = _parse_utc(assessment_at_utc, label="assessment_at_utc")
    accessed_at = _parse_utc(context.l3_accessed_at_utc, label="l3_accessed_at_utc")
    expires_at = _parse_utc(context.l3_expires_at_utc, label="l3_expires_at_utc")
    if accessed_at >= expires_at:
        raise PlannerCacheError("l3_accessed_at_utc must precede l3_expires_at_utc")
    if assessment_at < accessed_at:
        raise PlannerCacheError("assessment_at_utc predates the frozen L3 snapshot")
    if assessment_at >= expires_at:
        raise PlannerCacheError("frozen L3 snapshot is expired at assessment_at_utc")


def planner_cache_context_sha256(context: PlannerCacheContext) -> str:
    """Return a deterministic digest of every cache-relevant context field."""

    return _sha256_payload(context.to_dict())


def _canonical_constitution(smiles: str, *, label: str) -> str:
    _require_nonempty(smiles, label=label)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise PlannerCacheError(f"{label} is invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _canonical_target_payload(target: RouteTarget) -> dict[str, Any]:
    return {
        "role": target.role,
        "canonical_smiles": _canonical_constitution(
            target.canonical_smiles,
            label="route target",
        ),
        "product_context_smiles": [
            _canonical_constitution(value, label="product context")
            for value in target.product_context_smiles
        ],
    }


def _require_key_matches_assessment(
    key: PlannerCacheKey,
    assessment: SynthesisAssessment,
) -> None:
    payload = key.payload
    if payload.get("schema_version") != PLANNER_CACHE_KEY_SCHEMA_VERSION:
        raise PlannerCacheError("cache key has an unsupported schema")
    if payload.get("target") != _canonical_target_payload(assessment.target):
        raise PlannerCacheError("cache key target does not match assessment target")


def _require_key_context(key: PlannerCacheKey, context: PlannerCacheContext) -> None:
    if key.payload.get("context") != context.to_dict():
        raise PlannerCacheError("cache key context differs from the frozen snapshot context")


@dataclass(frozen=True)
class PlannerCacheSnapshotEntry:
    """One byte-owned cache entry in a root-independent snapshot manifest."""

    key_sha256: str
    relative_path: str
    file_sha256: str
    assessment_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("key_sha256", "file_sha256", "assessment_sha256"):
            _require_sha256(getattr(self, field_name), label=field_name)
        if not isinstance(self.relative_path, str) or not _CACHE_RELATIVE_PATH_PATTERN.fullmatch(
            self.relative_path
        ):
            raise PlannerCacheError("snapshot relative_path is not a canonical cache-entry path")
        expected = f"{self.key_sha256[:2]}/{self.key_sha256}.json"
        if self.relative_path != expected:
            raise PlannerCacheError("snapshot relative_path does not match key_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "key_sha256": self.key_sha256,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "assessment_sha256": self.assessment_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlannerCacheSnapshotEntry:
        if not isinstance(value, dict) or set(value) != {
            "key_sha256",
            "relative_path",
            "file_sha256",
            "assessment_sha256",
        }:
            raise PlannerCacheError("cache snapshot entry has an unsupported schema")
        return cls(
            key_sha256=value.get("key_sha256"),
            relative_path=value.get("relative_path"),
            file_sha256=value.get("file_sha256"),
            assessment_sha256=value.get("assessment_sha256"),
        )


@dataclass(frozen=True)
class PlannerCacheSnapshotManifest:
    """Deterministic immutable ownership manifest for one cache directory."""

    context_sha256: str
    l3_snapshot_sha256: str
    l3_accessed_at_utc: str
    l3_expires_at_utc: str
    entries: tuple[PlannerCacheSnapshotEntry, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.context_sha256, label="context_sha256")
        _require_sha256(self.l3_snapshot_sha256, label="l3_snapshot_sha256")
        accessed_at = _parse_utc(self.l3_accessed_at_utc, label="l3_accessed_at_utc")
        expires_at = _parse_utc(self.l3_expires_at_utc, label="l3_expires_at_utc")
        if accessed_at >= expires_at:
            raise PlannerCacheError("snapshot L3 access time must precede expiry")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, PlannerCacheSnapshotEntry) for entry in self.entries
        ):
            raise PlannerCacheError("snapshot entries must be a tuple of typed records")
        keys = [entry.key_sha256 for entry in self.entries]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise PlannerCacheError("snapshot entries must be unique and sorted by cache key")

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_CACHE_SNAPSHOT_SCHEMA_VERSION,
            "context_sha256": self.context_sha256,
            "l3_snapshot_sha256": self.l3_snapshot_sha256,
            "l3_accessed_at_utc": self.l3_accessed_at_utc,
            "l3_expires_at_utc": self.l3_expires_at_utc,
            "entry_count": self.entry_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def snapshot_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "snapshot_sha256": self.snapshot_sha256}

    @classmethod
    def from_dict(cls, value: Any) -> PlannerCacheSnapshotManifest:
        expected_fields = {
            "schema_version",
            "context_sha256",
            "l3_snapshot_sha256",
            "l3_accessed_at_utc",
            "l3_expires_at_utc",
            "entry_count",
            "entries",
            "snapshot_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected_fields
            or value.get("schema_version") != PLANNER_CACHE_SNAPSHOT_SCHEMA_VERSION
        ):
            raise PlannerCacheError("cache snapshot manifest has an unsupported schema")
        entries = value.get("entries")
        if not isinstance(entries, list):
            raise PlannerCacheError("cache snapshot entries must be a list")
        record = cls(
            context_sha256=value.get("context_sha256"),
            l3_snapshot_sha256=value.get("l3_snapshot_sha256"),
            l3_accessed_at_utc=value.get("l3_accessed_at_utc"),
            l3_expires_at_utc=value.get("l3_expires_at_utc"),
            entries=tuple(PlannerCacheSnapshotEntry.from_dict(entry) for entry in entries),
        )
        _require_nonnegative_integer(value.get("entry_count"), label="snapshot entry_count")
        if value.get("entry_count") != record.entry_count:
            raise PlannerCacheError("cache snapshot entry_count is inconsistent")
        _require_sha256(value.get("snapshot_sha256"), label="snapshot_sha256")
        if value.get("snapshot_sha256") != record.snapshot_sha256:
            raise PlannerCacheError("cache snapshot checksum mismatch")
        return record


@dataclass(frozen=True)
class _DecodedCacheEntry:
    key: PlannerCacheKey
    assessment: SynthesisAssessment
    raw: bytes
    assessment_sha256: str


def _decode_cache_entry(
    cache: FilePlannerCache,
    path: Path,
    *,
    expected_key: PlannerCacheKey | None = None,
) -> _DecodedCacheEntry:
    if path.is_symlink():
        raise PlannerCacheError(f"cache entries cannot be symbolic links: {path}")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PlannerCacheError(f"cache entry is not UTF-8: {path}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerCacheError(f"cache entry is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != (
        PLANNER_CACHE_ENTRY_SCHEMA_VERSION
    ):
        raise PlannerCacheError(f"cache entry has an unsupported schema: {path}")
    key_value = value.get("key")
    if not isinstance(key_value, dict) or set(key_value) != {"digest", "payload"}:
        raise PlannerCacheError(f"cache entry key is malformed: {path}")
    payload = key_value.get("payload")
    if not isinstance(payload, dict):
        raise PlannerCacheError(f"cache entry key payload is malformed: {path}")
    key = PlannerCacheKey(
        digest=key_value.get("digest"),
        payload_json=_stable_json(payload),
    )
    if expected_key is not None and key != expected_key:
        raise PlannerCacheError(f"cache entry key mismatch: {path}")
    relative_path = path.relative_to(cache.root).as_posix()
    expected_relative_path = f"{key.digest[:2]}/{key.digest}.json"
    if relative_path != expected_relative_path:
        raise PlannerCacheError(f"cache entry is stored at a noncanonical path: {path}")
    assessment_value = value.get("assessment")
    assessment_sha256 = value.get("assessment_sha256")
    _require_sha256(assessment_sha256, label="cache assessment_sha256")
    if assessment_sha256 != _sha256_payload(assessment_value):
        raise PlannerCacheError(f"cache assessment checksum mismatch: {path}")
    assessment = SynthesisAssessment.from_dict(assessment_value)
    _require_key_matches_assessment(key, assessment)
    canonical_bytes = (_stable_json(value) + "\n").encode()
    if raw != canonical_bytes:
        raise PlannerCacheError(f"cache entry is not canonical JSON: {path}")
    return _DecodedCacheEntry(
        key=key,
        assessment=assessment,
        raw=raw,
        assessment_sha256=assessment_sha256,
    )


def build_file_planner_cache_snapshot_manifest(
    cache: FilePlannerCache,
    context: PlannerCacheContext,
    *,
    assessment_at_utc: str,
) -> PlannerCacheSnapshotManifest:
    """Validate every cache entry and return a root-independent manifest."""

    require_current_l3_context(context, assessment_at_utc=assessment_at_utc)
    if not cache.root.exists():
        paths: tuple[Path, ...] = ()
    else:
        if not cache.root.is_dir() or cache.root.is_symlink():
            raise PlannerCacheError("cache root must be a real directory")
        discovered = []
        for path in cache.root.rglob("*"):
            if path.is_symlink():
                raise PlannerCacheError(f"cache snapshot contains a symbolic link: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise PlannerCacheError(f"cache snapshot contains an unsupported entry: {path}")
            discovered.append(path)
        paths = tuple(sorted(discovered, key=lambda item: item.relative_to(cache.root).as_posix()))
    entries = []
    for path in paths:
        relative_path = path.relative_to(cache.root).as_posix()
        if not _CACHE_RELATIVE_PATH_PATTERN.fullmatch(relative_path):
            raise PlannerCacheError(f"cache snapshot contains a noncanonical file: {path}")
        decoded = _decode_cache_entry(cache, path)
        entries.append(
            PlannerCacheSnapshotEntry(
                key_sha256=decoded.key.digest,
                relative_path=relative_path,
                file_sha256=hashlib.sha256(decoded.raw).hexdigest(),
                assessment_sha256=decoded.assessment_sha256,
            )
        )
    return PlannerCacheSnapshotManifest(
        context_sha256=planner_cache_context_sha256(context),
        l3_snapshot_sha256=context.l3_snapshot_sha256,
        l3_accessed_at_utc=context.l3_accessed_at_utc,
        l3_expires_at_utc=context.l3_expires_at_utc,
        entries=tuple(sorted(entries, key=lambda entry: entry.key_sha256)),
    )


def _require_manifest_context(
    manifest: PlannerCacheSnapshotManifest,
    context: PlannerCacheContext,
) -> None:
    if manifest.context_sha256 != planner_cache_context_sha256(context):
        raise PlannerCacheError("cache snapshot context checksum mismatch")
    if manifest.l3_snapshot_sha256 != context.l3_snapshot_sha256:
        raise PlannerCacheError("cache snapshot L3 checksum mismatch")
    if manifest.l3_accessed_at_utc != context.l3_accessed_at_utc:
        raise PlannerCacheError("cache snapshot L3 access time mismatch")
    if manifest.l3_expires_at_utc != context.l3_expires_at_utc:
        raise PlannerCacheError("cache snapshot L3 expiry mismatch")


class ReadOnlyFilePlannerCache:
    """Immutable logical view of exactly the entries owned by one manifest."""

    def __init__(
        self,
        cache: FilePlannerCache,
        manifest: PlannerCacheSnapshotManifest,
        context: PlannerCacheContext,
        *,
        assessment_at_utc: str,
    ):
        require_current_l3_context(context, assessment_at_utc=assessment_at_utc)
        _require_manifest_context(manifest, context)
        observed = build_file_planner_cache_snapshot_manifest(
            cache,
            context,
            assessment_at_utc=assessment_at_utc,
        )
        if observed != manifest:
            raise PlannerCacheError("cache root does not match the immutable snapshot manifest")
        self._cache = cache
        self._manifest = manifest
        self._context = context
        self._assessment_at_utc = assessment_at_utc
        self._entries_by_key = {entry.key_sha256: entry for entry in manifest.entries}

    @property
    def manifest(self) -> PlannerCacheSnapshotManifest:
        return self._manifest

    @property
    def context(self) -> PlannerCacheContext:
        return self._context

    @property
    def assessment_at_utc(self) -> str:
        return self._assessment_at_utc

    def get(self, key: PlannerCacheKey) -> SynthesisAssessment | None:
        require_current_l3_context(
            self._context,
            assessment_at_utc=self._assessment_at_utc,
        )
        _require_key_context(key, self._context)
        entry = self._entries_by_key.get(key.digest)
        if entry is None:
            return None
        path = self._cache.root / entry.relative_path
        try:
            decoded = _decode_cache_entry(self._cache, path, expected_key=key)
        except FileNotFoundError as exc:
            raise PlannerCacheError(f"immutable cache snapshot entry disappeared: {path}") from exc
        if hashlib.sha256(decoded.raw).hexdigest() != entry.file_sha256:
            raise PlannerCacheError(f"immutable cache snapshot entry changed: {path}")
        if decoded.assessment_sha256 != entry.assessment_sha256:
            raise PlannerCacheError(f"immutable cache snapshot assessment changed: {path}")
        return decoded.assessment

    def put(self, key: PlannerCacheKey, assessment: SynthesisAssessment) -> None:
        del key, assessment
        raise PlannerCacheError("immutable cache snapshot is read-only")

    def assert_unchanged(self) -> PlannerCacheSnapshotManifest:
        observed = build_file_planner_cache_snapshot_manifest(
            self._cache,
            self._context,
            assessment_at_utc=self._assessment_at_utc,
        )
        if observed != self._manifest:
            raise PlannerCacheError("immutable base cache changed after snapshot creation")
        return observed


class OverlayFilePlannerCache:
    """Writable copy-on-write cache layered over an immutable base snapshot."""

    def __init__(
        self,
        base: ReadOnlyFilePlannerCache,
        overlay: FilePlannerCache,
        *,
        clone_id: str,
    ):
        _require_nonempty(clone_id, label="cache clone_id")
        self._base = base
        self._overlay = overlay
        self._clone_id = clone_id
        self._before_manifest = self._validated_manifest()

    def _validated_manifest(self) -> PlannerCacheSnapshotManifest:
        manifest = build_file_planner_cache_snapshot_manifest(
            self._overlay,
            self._base.context,
            assessment_at_utc=self._base.assessment_at_utc,
        )
        for entry in manifest.entries:
            decoded = _decode_cache_entry(
                self._overlay,
                self._overlay.root / entry.relative_path,
            )
            _require_key_context(decoded.key, self._base.context)
            base_assessment = self._base.get(decoded.key)
            if base_assessment is not None and base_assessment != decoded.assessment:
                raise PlannerCacheError("overlay conflicts with its immutable base snapshot")
        return manifest

    @property
    def clone_id(self) -> str:
        return self._clone_id

    @property
    def before_manifest(self) -> PlannerCacheSnapshotManifest:
        return self._before_manifest

    def get(self, key: PlannerCacheKey) -> SynthesisAssessment | None:
        require_current_l3_context(
            self._base.context,
            assessment_at_utc=self._base.assessment_at_utc,
        )
        _require_key_context(key, self._base.context)
        assessment = self._overlay.get(key)
        if assessment is not None:
            _require_key_matches_assessment(key, assessment)
            return assessment
        return self._base.get(key)

    def put(self, key: PlannerCacheKey, assessment: SynthesisAssessment) -> None:
        require_current_l3_context(
            self._base.context,
            assessment_at_utc=self._base.assessment_at_utc,
        )
        _require_key_context(key, self._base.context)
        _require_key_matches_assessment(key, assessment)
        overlay_assessment = self._overlay.get(key)
        if overlay_assessment is not None:
            self._overlay.put(key, assessment)
            return
        base_assessment = self._base.get(key)
        if base_assessment is not None:
            if base_assessment != assessment:
                raise PlannerCacheError("overlay write conflicts with immutable base assessment")
            return
        self._overlay.put(key, assessment)

    def current_manifest(self) -> PlannerCacheSnapshotManifest:
        return self._validated_manifest()


@dataclass(frozen=True)
class MatchedPlannerCacheOverlayAudit:
    """Before/after ownership proof for one matched pair of cache overlays."""

    assessment_at_utc: str
    context_sha256: str
    base_before: PlannerCacheSnapshotManifest
    base_after: PlannerCacheSnapshotManifest
    guided_clone_id: str
    guided_before: PlannerCacheSnapshotManifest
    guided_after: PlannerCacheSnapshotManifest
    post_hoc_clone_id: str
    post_hoc_before: PlannerCacheSnapshotManifest
    post_hoc_after: PlannerCacheSnapshotManifest

    def __post_init__(self) -> None:
        _parse_utc(self.assessment_at_utc, label="assessment_at_utc")
        _require_sha256(self.context_sha256, label="context_sha256")
        _require_nonempty(self.guided_clone_id, label="guided_clone_id")
        _require_nonempty(self.post_hoc_clone_id, label="post_hoc_clone_id")
        if self.guided_clone_id == self.post_hoc_clone_id:
            raise PlannerCacheError("matched cache clone IDs must be distinct")
        if self.base_before != self.base_after:
            raise PlannerCacheError("matched cache audit cannot certify a changed base snapshot")
        if self.guided_before != self.post_hoc_before:
            raise PlannerCacheError("matched cache overlays did not start from identical states")
        for manifest in (
            self.base_before,
            self.base_after,
            self.guided_before,
            self.guided_after,
            self.post_hoc_before,
            self.post_hoc_after,
        ):
            if manifest.context_sha256 != self.context_sha256:
                raise PlannerCacheError("matched cache manifest context mismatch")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_CACHE_OVERLAY_AUDIT_SCHEMA_VERSION,
            "assessment_at_utc": self.assessment_at_utc,
            "context_sha256": self.context_sha256,
            "base_unchanged": True,
            "base_before": self.base_before.to_dict(),
            "base_after": self.base_after.to_dict(),
            "guided": {
                "clone_id": self.guided_clone_id,
                "before": self.guided_before.to_dict(),
                "after": self.guided_after.to_dict(),
            },
            "post_hoc": {
                "clone_id": self.post_hoc_clone_id,
                "before": self.post_hoc_before.to_dict(),
                "after": self.post_hoc_after.to_dict(),
            },
        }

    @property
    def audit_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "audit_sha256": self.audit_sha256}


@dataclass(frozen=True)
class MatchedFilePlannerCacheOverlays:
    """Distinct guided/post-hoc overlays sharing one immutable base snapshot."""

    base: ReadOnlyFilePlannerCache
    guided: OverlayFilePlannerCache
    post_hoc: OverlayFilePlannerCache

    def finalize(self) -> MatchedPlannerCacheOverlayAudit:
        base_after = self.base.assert_unchanged()
        return MatchedPlannerCacheOverlayAudit(
            assessment_at_utc=self.base.assessment_at_utc,
            context_sha256=planner_cache_context_sha256(self.base.context),
            base_before=self.base.manifest,
            base_after=base_after,
            guided_clone_id=self.guided.clone_id,
            guided_before=self.guided.before_manifest,
            guided_after=self.guided.current_manifest(),
            post_hoc_clone_id=self.post_hoc.clone_id,
            post_hoc_before=self.post_hoc.before_manifest,
            post_hoc_after=self.post_hoc.current_manifest(),
        )


def _roots_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def open_matched_file_planner_cache_overlays(
    base_cache: FilePlannerCache,
    guided_overlay: FilePlannerCache,
    post_hoc_overlay: FilePlannerCache,
    context: PlannerCacheContext,
    *,
    assessment_at_utc: str,
    guided_clone_id: str,
    post_hoc_clone_id: str,
) -> MatchedFilePlannerCacheOverlays:
    """Open two isolated overlays from one byte-verified immutable cache state."""

    _require_nonempty(guided_clone_id, label="guided_clone_id")
    _require_nonempty(post_hoc_clone_id, label="post_hoc_clone_id")
    if guided_clone_id == post_hoc_clone_id:
        raise PlannerCacheError("guided and post-hoc cache clone IDs must be distinct")
    roots = (base_cache.root, guided_overlay.root, post_hoc_overlay.root)
    if any(
        _roots_overlap(left, right)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise PlannerCacheError("base and matched overlay cache roots must not overlap")
    base_manifest = build_file_planner_cache_snapshot_manifest(
        base_cache,
        context,
        assessment_at_utc=assessment_at_utc,
    )
    base = ReadOnlyFilePlannerCache(
        base_cache,
        base_manifest,
        context,
        assessment_at_utc=assessment_at_utc,
    )
    guided = OverlayFilePlannerCache(base, guided_overlay, clone_id=guided_clone_id)
    post_hoc = OverlayFilePlannerCache(base, post_hoc_overlay, clone_id=post_hoc_clone_id)
    if guided.before_manifest != post_hoc.before_manifest:
        raise PlannerCacheError("guided and post-hoc overlays must begin byte-identically")
    return MatchedFilePlannerCacheOverlays(base=base, guided=guided, post_hoc=post_hoc)
