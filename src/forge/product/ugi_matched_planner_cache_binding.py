"""Lazy planner-cache binding for matched diagnostic route assessments.

The hash-pinned matched-budget runner derives arm clone identifiers internally
and exposes them only through ``MatchedAssessmentContext`` callbacks.  This
module therefore seals both empty overlay roots before generation, then binds
each arm lazily to the clone identifier supplied by its first callback.  It
does not reproduce the runner's schedule or clone-ID hash formulas.

This layer owns cache provenance only.  It does not run a generator or route
source, define a synthesis scalar, guide sampling, invoke biology, or select a
candidate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.product.ugi_matched_budget_orchestration import (
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.route.planner_cache import (
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
)
from forge.route.planner_cache_snapshot import (
    MatchedFilePlannerCacheOverlays,
    MatchedPlannerCacheOverlayAudit,
    OverlayFilePlannerCache,
    PlannerCacheSnapshotManifest,
    ReadOnlyFilePlannerCache,
    build_file_planner_cache_snapshot_manifest,
    planner_cache_context_sha256,
)

MATCHED_PLANNER_CACHE_PREFLIGHT_SCHEMA_VERSION = "forge.matched_planner_cache_binding_preflight.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannerCacheError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise PlannerCacheError(f"{label} must contain 64 lowercase hexadecimal characters")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    _require_nonempty(value, label=label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PlannerCacheError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PlannerCacheError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _roots_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


@dataclass(frozen=True)
class MatchedPlannerCacheBindingPreflight:
    """Root-independent ownership seal created before productive generation."""

    assessment_at_utc: str
    context_sha256: str
    base: PlannerCacheSnapshotManifest
    guided_before: PlannerCacheSnapshotManifest
    post_hoc_before: PlannerCacheSnapshotManifest

    def __post_init__(self) -> None:
        assessment_at = _parse_utc(self.assessment_at_utc, label="assessment_at_utc")
        _require_sha256(self.context_sha256, label="context_sha256")
        manifests = (self.base, self.guided_before, self.post_hoc_before)
        if any(not isinstance(value, PlannerCacheSnapshotManifest) for value in manifests):
            raise PlannerCacheError("preflight cache manifests are malformed")
        if any(value.context_sha256 != self.context_sha256 for value in manifests):
            raise PlannerCacheError("preflight cache manifests use different contexts")
        if self.guided_before != self.post_hoc_before:
            raise PlannerCacheError("matched overlay roots are not byte-identical at preflight")
        if self.guided_before.entry_count != 0 or self.post_hoc_before.entry_count != 0:
            raise PlannerCacheError("matched overlay roots must be empty at preflight")
        accessed_at = _parse_utc(self.base.l3_accessed_at_utc, label="l3_accessed_at_utc")
        expires_at = _parse_utc(self.base.l3_expires_at_utc, label="l3_expires_at_utc")
        if assessment_at < accessed_at or assessment_at >= expires_at:
            raise PlannerCacheError("preflight assessment time is outside the L3 validity window")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_PLANNER_CACHE_PREFLIGHT_SCHEMA_VERSION,
            "assessment_at_utc": self.assessment_at_utc,
            "context_sha256": self.context_sha256,
            "overlay_roots_empty": True,
            "base": self.base.to_dict(),
            "guided_before": self.guided_before.to_dict(),
            "post_hoc_before": self.post_hoc_before.to_dict(),
        }

    @property
    def preflight_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "preflight_sha256": self.preflight_sha256}

    @classmethod
    def from_dict(cls, value: Any) -> MatchedPlannerCacheBindingPreflight:
        expected = {
            "schema_version",
            "assessment_at_utc",
            "context_sha256",
            "overlay_roots_empty",
            "base",
            "guided_before",
            "post_hoc_before",
            "preflight_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema_version") != MATCHED_PLANNER_CACHE_PREFLIGHT_SCHEMA_VERSION
            or value.get("overlay_roots_empty") is not True
        ):
            raise PlannerCacheError("matched planner-cache preflight has an unsupported schema")
        record = cls(
            assessment_at_utc=value.get("assessment_at_utc"),
            context_sha256=value.get("context_sha256"),
            base=PlannerCacheSnapshotManifest.from_dict(value.get("base")),
            guided_before=PlannerCacheSnapshotManifest.from_dict(value.get("guided_before")),
            post_hoc_before=PlannerCacheSnapshotManifest.from_dict(value.get("post_hoc_before")),
        )
        _require_sha256(value.get("preflight_sha256"), label="preflight_sha256")
        if value.get("preflight_sha256") != record.preflight_sha256:
            raise PlannerCacheError("matched planner-cache preflight checksum mismatch")
        return record


class LazyMatchedPlannerCacheBinding:
    """Bind preflight-sealed arm overlays to runner-supplied clone IDs lazily."""

    def __init__(
        self,
        *,
        context: PlannerCacheContext,
        base: ReadOnlyFilePlannerCache,
        base_cache: FilePlannerCache,
        guided_overlay: FilePlannerCache,
        post_hoc_overlay: FilePlannerCache,
        preflight: MatchedPlannerCacheBindingPreflight,
    ):
        self._context = context
        self._base = base
        self._base_file = base_cache
        self._base_root = base_cache.root.resolve(strict=False)
        self._guided_file = guided_overlay
        self._post_hoc_file = post_hoc_overlay
        self._guided_root = guided_overlay.root.resolve(strict=False)
        self._post_hoc_root = post_hoc_overlay.root.resolve(strict=False)
        self._preflight = preflight
        self._guided: OverlayFilePlannerCache | None = None
        self._post_hoc: OverlayFilePlannerCache | None = None
        self._guided_frozen_at_post_hoc: PlannerCacheSnapshotManifest | None = None
        self._finalized = False

    @property
    def preflight(self) -> MatchedPlannerCacheBindingPreflight:
        return self._preflight

    @property
    def guided_bound(self) -> bool:
        return self._guided is not None

    @property
    def post_hoc_bound(self) -> bool:
        return self._post_hoc is not None

    def _observed_overlay_manifest(
        self,
        cache: FilePlannerCache,
    ) -> PlannerCacheSnapshotManifest:
        return build_file_planner_cache_snapshot_manifest(
            cache,
            self._context,
            assessment_at_utc=self._preflight.assessment_at_utc,
        )

    @staticmethod
    def _require_same_root(cache: FilePlannerCache, expected: Path, *, label: str) -> None:
        if cache.root.resolve(strict=False) != expected:
            raise PlannerCacheError(f"{label} cache root changed after preflight")

    def _require_root_identities(self) -> None:
        self._require_same_root(self._base_file, self._base_root, label="base")
        self._require_same_root(self._guided_file, self._guided_root, label="guided")
        self._require_same_root(self._post_hoc_file, self._post_hoc_root, label="post-hoc")

    def _require_root_unchanged(
        self,
        cache: FilePlannerCache,
        expected: PlannerCacheSnapshotManifest,
        *,
        arm: str,
    ) -> None:
        observed = self._observed_overlay_manifest(cache)
        if observed != expected:
            raise PlannerCacheError(f"{arm} overlay root changed after the matched-cache preflight")

    def _validate_assessment_context(self, value: MatchedAssessmentContext) -> None:
        if not isinstance(value, MatchedAssessmentContext):
            raise PlannerCacheError("cache binding requires a MatchedAssessmentContext")
        if not isinstance(value.arm, MatchedArm):
            raise PlannerCacheError("matched assessment arm is unsupported")
        _require_nonempty(value.cache_clone_id, label="cache_clone_id")
        _require_sha256(value.cache_snapshot_sha256, label="cache_snapshot_sha256")
        if value.cache_snapshot_sha256 != self._preflight.base.snapshot_sha256:
            raise PlannerCacheError("runner cache snapshot differs from the sealed base manifest")
        if (
            isinstance(value.route_seed, bool)
            or not isinstance(value.route_seed, int)
            or value.route_seed < 0
        ):
            raise PlannerCacheError("route_seed must be a nonnegative integer")
        if not isinstance(value.remaining_budget, RouteComputeUsage) or not isinstance(
            value.unit_reservation, RouteComputeUsage
        ):
            raise PlannerCacheError("matched route budgets are malformed")
        if value.arm is MatchedArm.GUIDED:
            if value.post_hoc_lock_manifest_sha256 is not None:
                raise PlannerCacheError(
                    "guided cache binding cannot carry a post-hoc lock manifest"
                )
        else:
            _require_sha256(
                value.post_hoc_lock_manifest_sha256,
                label="post_hoc_lock_manifest_sha256",
            )

    def bind(self, context: MatchedAssessmentContext) -> OverlayFilePlannerCache:
        """Return the exact arm cache after binding its runner-supplied clone ID."""

        if self._finalized:
            raise PlannerCacheError("matched planner-cache binding is already finalized")
        self._validate_assessment_context(context)
        self._require_root_identities()
        self._base.assert_unchanged()

        if context.arm is MatchedArm.GUIDED:
            if self._post_hoc is not None:
                raise PlannerCacheError("guided cache callback occurred after post-hoc binding")
            self._require_root_unchanged(
                self._post_hoc_file,
                self._preflight.post_hoc_before,
                arm=MatchedArm.POST_HOC.value,
            )
            if self._guided is None:
                self._require_root_unchanged(
                    self._guided_file,
                    self._preflight.guided_before,
                    arm=MatchedArm.GUIDED.value,
                )
                self._guided = OverlayFilePlannerCache(
                    self._base,
                    self._guided_file,
                    clone_id=context.cache_clone_id,
                )
                if self._guided.before_manifest != self._preflight.guided_before:
                    raise PlannerCacheError("guided overlay opened from the wrong preflight state")
            elif self._guided.clone_id != context.cache_clone_id:
                raise PlannerCacheError("guided runner callback changed its cache clone ID")
            self._guided.current_manifest()
            return self._guided

        if self._guided is None:
            raise PlannerCacheError("post-hoc cache cannot bind before the guided cache")
        if self._guided.clone_id == context.cache_clone_id:
            raise PlannerCacheError("guided and post-hoc callbacks reused one cache clone ID")
        if self._post_hoc is None:
            self._require_root_unchanged(
                self._post_hoc_file,
                self._preflight.post_hoc_before,
                arm=MatchedArm.POST_HOC.value,
            )
            self._guided_frozen_at_post_hoc = self._guided.current_manifest()
            self._post_hoc = OverlayFilePlannerCache(
                self._base,
                self._post_hoc_file,
                clone_id=context.cache_clone_id,
            )
            if self._post_hoc.before_manifest != self._preflight.post_hoc_before:
                raise PlannerCacheError("post-hoc overlay opened from the wrong preflight state")
        elif self._post_hoc.clone_id != context.cache_clone_id:
            raise PlannerCacheError("post-hoc runner callback changed its cache clone ID")
        if self._guided.current_manifest() != self._guided_frozen_at_post_hoc:
            raise PlannerCacheError("guided overlay changed after post-hoc binding")
        self._post_hoc.current_manifest()
        return self._post_hoc

    def finalize(self) -> MatchedPlannerCacheOverlayAudit:
        """Seal before/after ownership proof through the shared cache audit."""

        if self._finalized:
            raise PlannerCacheError("matched planner-cache binding is already finalized")
        self._require_root_identities()
        if self._guided is None or self._post_hoc is None:
            raise PlannerCacheError("both matched cache arms must bind before finalization")
        if self._guided.current_manifest() != self._guided_frozen_at_post_hoc:
            raise PlannerCacheError("guided overlay changed after post-hoc binding")
        audit = MatchedFilePlannerCacheOverlays(
            base=self._base,
            guided=self._guided,
            post_hoc=self._post_hoc,
        ).finalize()
        if audit.base_before != self._preflight.base:
            raise PlannerCacheError("final cache audit used a different base preflight manifest")
        if audit.guided_before != self._preflight.guided_before:
            raise PlannerCacheError("final cache audit used a different guided preflight manifest")
        if audit.post_hoc_before != self._preflight.post_hoc_before:
            raise PlannerCacheError(
                "final cache audit used a different post-hoc preflight manifest"
            )
        self._finalized = True
        return audit


def preflight_lazy_matched_planner_cache_binding(
    base_cache: FilePlannerCache,
    guided_overlay: FilePlannerCache,
    post_hoc_overlay: FilePlannerCache,
    context: PlannerCacheContext,
    *,
    assessment_at_utc: str,
) -> LazyMatchedPlannerCacheBinding:
    """Seal empty arm roots and return a lazy runner-callback cache binding."""

    if not all(
        isinstance(value, FilePlannerCache)
        for value in (base_cache, guided_overlay, post_hoc_overlay)
    ):
        raise PlannerCacheError("matched cache roots must be FilePlannerCache instances")
    if not isinstance(context, PlannerCacheContext):
        raise PlannerCacheError("matched cache context is malformed")
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
    guided_before = build_file_planner_cache_snapshot_manifest(
        guided_overlay,
        context,
        assessment_at_utc=assessment_at_utc,
    )
    post_hoc_before = build_file_planner_cache_snapshot_manifest(
        post_hoc_overlay,
        context,
        assessment_at_utc=assessment_at_utc,
    )
    preflight = MatchedPlannerCacheBindingPreflight(
        assessment_at_utc=assessment_at_utc,
        context_sha256=planner_cache_context_sha256(context),
        base=base_manifest,
        guided_before=guided_before,
        post_hoc_before=post_hoc_before,
    )
    return LazyMatchedPlannerCacheBinding(
        context=context,
        base=base,
        base_cache=base_cache,
        guided_overlay=guided_overlay,
        post_hoc_overlay=post_hoc_overlay,
        preflight=preflight,
    )
