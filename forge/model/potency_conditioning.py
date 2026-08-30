"""Typed, fail-closed potency conditions for bounded FORGE diagnostics.

The potency condition is deliberately narrower than a generic property-conditioning API.  It
identifies one frozen endpoint and policy, carries only a percentile target, and is accepted only
for the reaction program and flow-time intervals admitted by the attached adapter policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PotencyConditioningError(ValueError):
    """A potency request lies outside the explicitly admitted diagnostic support."""


@dataclass(frozen=True)
class PotencyCondition:
    """One scalar potency request broadcast across a generation batch."""

    endpoint_id: str
    target_quantile: float
    policy_id: str

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.policy_id:
            raise PotencyConditioningError("potency endpoint and policy identifiers are required")
        if not math.isfinite(self.target_quantile) or not 0.0 <= self.target_quantile <= 1.0:
            raise PotencyConditioningError("potency target quantile must lie in [0, 1]")


@dataclass(frozen=True)
class PotencyConditionBatch:
    """Per-example percentile targets used while fitting a potency adapter."""

    endpoint_id: str
    target_quantiles: Any
    policy_id: str

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.policy_id:
            raise PotencyConditioningError("potency endpoint and policy identifiers are required")


@dataclass(frozen=True)
class PotencyAdapterPolicy:
    """Support boundary attached to one lightweight potency-adapter overlay."""

    policy_id: str
    endpoint_id: str
    program_id: str
    minimum_quantile: float
    maximum_quantile: float
    active_time_intervals: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.policy_id or not self.endpoint_id or not self.program_id:
            raise PotencyConditioningError("potency policy identifiers cannot be empty")
        if (
            not math.isfinite(self.minimum_quantile)
            or not math.isfinite(self.maximum_quantile)
            or not 0.0 <= self.minimum_quantile <= self.maximum_quantile <= 1.0
        ):
            raise PotencyConditioningError("potency policy quantile support is invalid")
        previous = -1.0
        for lower, upper in self.active_time_intervals:
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or not 0.0 <= lower < upper <= 1.0
                or lower < previous
            ):
                raise PotencyConditioningError("potency time intervals are invalid or overlap")
            previous = upper

    def to_mapping(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "endpoint_id": self.endpoint_id,
            "program_id": self.program_id,
            "minimum_quantile": self.minimum_quantile,
            "maximum_quantile": self.maximum_quantile,
            "active_time_intervals": [list(interval) for interval in self.active_time_intervals],
        }

    @classmethod
    def from_mapping(cls, raw: Any) -> PotencyAdapterPolicy:
        if not isinstance(raw, dict):
            raise PotencyConditioningError("potency adapter policy must be an object")
        expected = {
            "policy_id",
            "endpoint_id",
            "program_id",
            "minimum_quantile",
            "maximum_quantile",
            "active_time_intervals",
        }
        if set(raw) != expected or not isinstance(raw["active_time_intervals"], list):
            raise PotencyConditioningError("potency adapter policy fields changed")
        try:
            intervals = tuple(
                (float(interval[0]), float(interval[1]))
                for interval in raw["active_time_intervals"]
                if isinstance(interval, list) and len(interval) == 2
            )
        except (TypeError, ValueError) as error:
            raise PotencyConditioningError("potency time intervals are malformed") from error
        if len(intervals) != len(raw["active_time_intervals"]):
            raise PotencyConditioningError("potency time intervals are malformed")
        return cls(
            policy_id=str(raw["policy_id"]),
            endpoint_id=str(raw["endpoint_id"]),
            program_id=str(raw["program_id"]),
            minimum_quantile=float(raw["minimum_quantile"]),
            maximum_quantile=float(raw["maximum_quantile"]),
            active_time_intervals=intervals,
        )


__all__ = [
    "PotencyAdapterPolicy",
    "PotencyCondition",
    "PotencyConditionBatch",
    "PotencyConditioningError",
]
