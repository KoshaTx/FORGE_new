"""Legacy computational makeability, separate from strict synthesis dossiers.

This pure classifier consumes evidence extracted by an authenticated adapter. It
does not fetch or authenticate files/URLs, canonicalize chemistry, or infer missing
evidence. Identities must already be canonical under the caller's declared form
policy; comparisons are exact. Receipt pins retain the source of every assertion.
``Receipt.verify_bytes`` is available for adapters with the source bytes in memory.

A planner-solved path needs no independent forward replay here. A constructed
path needs a known/source transform application at every step and exact target
membership in that step's outputs; uniqueness is a separate axis. Neither result
is an experimental-success claim, strict dossier, or synthesis-guidance value.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class MakeabilityError(ValueError):
    """Malformed evidence or policy; missing observations use explicit states."""


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MakeabilityError(f"{name} must be nonempty")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise MakeabilityError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True)
class Receipt:
    location: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.location, "receipt.location")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise MakeabilityError("receipt.sha256 must be a lowercase SHA-256")

    def verify_bytes(self, content: bytes) -> None:
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise MakeabilityError(f"receipt digest mismatch: {self.location}")


@dataclass(frozen=True)
class MakeabilityPolicy:
    policy_id: str
    vendor_snapshot_max_age_days: int

    def __post_init__(self) -> None:
        _text(self.policy_id, "policy_id")
        if (
            type(self.vendor_snapshot_max_age_days) is not int
            or self.vendor_snapshot_max_age_days < 1
        ):
            raise MakeabilityError("vendor_snapshot_max_age_days must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MakeabilityPolicy:
        if value.get("schema") != "forge.computational_makeability_policy.v1":
            raise MakeabilityError("unsupported computational makeability policy schema")
        return cls(value["policy_id"], value["vendor_snapshot_max_age_days"])


class LookupOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNASSESSED = "unassessed"


class ListingState(str, Enum):
    LISTED = "listed"
    ZERO = "zero_listings"
    STALE = "stale"
    FUTURE = "future_observation"
    FAILED = "lookup_failed"
    UNKNOWN = "unassessed"
    IDENTITY_MISMATCH = "identity_mismatch"


@dataclass(frozen=True)
class VendorListing:
    identity: str
    outcome: LookupOutcome
    vendor_count: int | None = None
    observed_at: datetime | None = None
    receipt: Receipt | None = None

    def __post_init__(self) -> None:
        _text(self.identity, "listing.identity")
        if not isinstance(self.outcome, LookupOutcome):
            raise MakeabilityError("listing.outcome must be a LookupOutcome")
        if self.observed_at is not None:
            _utc(self.observed_at, "listing.observed_at")
        if self.receipt is not None and not isinstance(self.receipt, Receipt):
            raise MakeabilityError("listing.receipt must be a Receipt")
        if self.outcome is LookupOutcome.SUCCESS:
            if type(self.vendor_count) is not int or self.vendor_count < 0:
                raise MakeabilityError("successful vendor_count must be a nonnegative integer")
            if self.observed_at is None or self.receipt is None:
                raise MakeabilityError("successful lookup requires timestamp and source receipt")
        elif self.vendor_count is not None:
            raise MakeabilityError("failed/unassessed lookup cannot carry vendor_count")


@dataclass(frozen=True)
class ListingAssessment:
    state: ListingState
    evidence: VendorListing


def classify_listing(
    identity: str, listing: VendorListing, *, policy: MakeabilityPolicy, as_of: datetime
) -> ListingAssessment:
    _text(identity, "identity")
    _utc(as_of, "as_of")
    if listing.identity != identity:
        state = ListingState.IDENTITY_MISMATCH
    elif listing.outcome is LookupOutcome.FAILED:
        state = ListingState.FAILED
    elif listing.outcome is LookupOutcome.UNASSESSED:
        state = ListingState.UNKNOWN
    elif listing.observed_at > as_of:
        state = ListingState.FUTURE
    elif as_of >= listing.observed_at + timedelta(days=policy.vendor_snapshot_max_age_days):
        state = ListingState.STALE
    else:
        state = ListingState.LISTED if listing.vendor_count > 0 else ListingState.ZERO
    return ListingAssessment(state, listing)


class RouteBasis(str, Enum):
    PLANNER_SOLVED = "planner_solved"
    FORWARD_APPLIED = "forward_applied_known_or_source_transform"


@dataclass(frozen=True)
class EvidenceAxes:
    """Recorded secondary axes. None means unassessed, not false or pass."""

    exact_source_execution: bool | None = None
    source_conditions: bool | None = None
    independent_forward_replay: bool | None = None
    unique_forward_product: bool | None = None
    substrate_applicability: bool | None = None

    def __post_init__(self) -> None:
        if any(value is not None and type(value) is not bool for value in asdict(self).values()):
            raise MakeabilityError("secondary evidence axes must be bool or None")


@dataclass(frozen=True)
class RouteStep:
    reactants: tuple[RouteNode, ...]
    receipt: Receipt
    forward_products: tuple[str, ...] | None = None
    transform_receipt: Receipt | None = None
    axes: EvidenceAxes = EvidenceAxes()

    def __post_init__(self) -> None:
        if not self.reactants or not all(isinstance(node, RouteNode) for node in self.reactants):
            raise MakeabilityError("route step requires explicit reactant nodes")
        if not isinstance(self.receipt, Receipt):
            raise MakeabilityError("route step requires a source/application receipt")
        if self.transform_receipt is not None and not isinstance(self.transform_receipt, Receipt):
            raise MakeabilityError("transform_receipt must be a Receipt")
        if self.forward_products is not None:
            for identity in self.forward_products:
                _text(identity, "step.forward_product")


@dataclass(frozen=True)
class RouteNode:
    identity: str
    step: RouteStep | None = None
    listings: tuple[VendorListing, ...] = ()

    def __post_init__(self) -> None:
        _text(self.identity, "route node identity")
        if self.step is not None and not isinstance(self.step, RouteStep):
            raise MakeabilityError("route node step must be a RouteStep")
        if not all(isinstance(item, VendorListing) for item in self.listings):
            raise MakeabilityError("route node listings must be VendorListing records")


@dataclass(frozen=True)
class ComputationalRoute:
    route_id: str
    basis: RouteBasis
    root: RouteNode
    receipt: Receipt

    def __post_init__(self) -> None:
        _text(self.route_id, "route_id")
        if not isinstance(self.basis, RouteBasis) or not isinstance(self.root, RouteNode):
            raise MakeabilityError("route requires a typed basis and explicit root")
        if not isinstance(self.receipt, Receipt):
            raise MakeabilityError("route requires its target-bound source receipt")


@dataclass(frozen=True)
class LeafAssessment:
    identity: str
    listings: tuple[ListingAssessment, ...]

    @property
    def listed(self) -> bool:
        return any(item.state is ListingState.LISTED for item in self.listings)


@dataclass(frozen=True)
class RouteAssessment:
    evidence: ComputationalRoute
    path_supported: bool
    leaves: tuple[LeafAssessment, ...]
    reasons: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.path_supported and bool(self.leaves) and all(leaf.listed for leaf in self.leaves)
        )


def classify_route(
    identity: str, route: ComputationalRoute, *, policy: MakeabilityPolicy, as_of: datetime
) -> RouteAssessment:
    """Evaluate one whole path; never borrow terminal observations from another."""
    _text(identity, "identity")
    _utc(as_of, "as_of")
    reasons = []
    leaves = []
    if route.root.identity != identity:
        reasons.append("target_identity_mismatch")
    if route.root.step is None:
        reasons.append("no_reaction_steps")
    pending = [(route.root, frozenset())]
    while pending:
        node, ancestors = pending.pop()
        if node.identity in ancestors:
            reasons.append("cyclic_route")
            continue
        if node.step is None:
            leaves.append(
                LeafAssessment(
                    node.identity,
                    tuple(
                        classify_listing(node.identity, item, policy=policy, as_of=as_of)
                        for item in node.listings
                    ),
                )
            )
            continue
        if route.basis is RouteBasis.FORWARD_APPLIED:
            if node.step.transform_receipt is None:
                reasons.append("missing_known_or_source_transform_receipt")
            if (
                node.step.forward_products is None
                or node.identity not in node.step.forward_products
            ):
                reasons.append("target_not_in_step_forward_products")
        pending.extend(
            (child, ancestors | {node.identity}) for child in reversed(node.step.reactants)
        )
    return RouteAssessment(route, not reasons, tuple(leaves), tuple(reasons))


class SearchOutcome(str, Enum):
    UNASSESSED = "unassessed"
    FAILED = "failed"
    COMPLETED = "completed"


class ComponentState(str, Enum):
    BUY = "buy"
    BUY_AND_MAKE = "buy_and_make"
    MAKE_FROM_VENDOR_LISTED = "make_from_vendor_listed"
    ROUTE_ONLY = "route_only_terminal_evidence_incomplete"
    BLOCKED = "blocked_in_supplied_search"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComponentEvidence:
    identity: str
    direct: tuple[VendorListing, ...] = ()
    routes: tuple[ComputationalRoute, ...] = ()
    search_outcome: SearchOutcome = SearchOutcome.UNASSESSED
    search_receipt: Receipt | None = None

    def __post_init__(self) -> None:
        _text(self.identity, "component.identity")
        if not all(isinstance(item, VendorListing) for item in self.direct):
            raise MakeabilityError("component.direct must contain VendorListing records")
        if not all(isinstance(item, ComputationalRoute) for item in self.routes):
            raise MakeabilityError("component.routes must contain ComputationalRoute records")
        if len({item.route_id for item in self.routes}) != len(self.routes):
            raise MakeabilityError("route IDs must be unique within a component")
        if not isinstance(self.search_outcome, SearchOutcome):
            raise MakeabilityError("component.search_outcome must be a SearchOutcome")
        if self.search_outcome is SearchOutcome.COMPLETED and not isinstance(
            self.search_receipt, Receipt
        ):
            raise MakeabilityError("completed search requires a receipt")


@dataclass(frozen=True)
class ComponentAssessment:
    evidence: ComponentEvidence
    state: ComponentState
    direct: tuple[ListingAssessment, ...]
    routes: tuple[RouteAssessment, ...]

    @property
    def makeable(self) -> bool:
        return self.state in {
            ComponentState.BUY,
            ComponentState.BUY_AND_MAKE,
            ComponentState.MAKE_FROM_VENDOR_LISTED,
        }


def classify_component(
    evidence: ComponentEvidence, *, policy: MakeabilityPolicy, as_of: datetime
) -> ComponentAssessment:
    _utc(as_of, "as_of")
    direct = tuple(
        classify_listing(evidence.identity, item, policy=policy, as_of=as_of)
        for item in evidence.direct
    )
    routes = tuple(
        classify_route(evidence.identity, item, policy=policy, as_of=as_of)
        for item in evidence.routes
    )
    buy = any(item.state is ListingState.LISTED for item in direct)
    make = any(item.complete for item in routes)
    if buy and make:
        state = ComponentState.BUY_AND_MAKE
    elif buy:
        state = ComponentState.BUY
    elif make:
        state = ComponentState.MAKE_FROM_VENDOR_LISTED
    elif any(item.path_supported for item in routes):
        state = ComponentState.ROUTE_ONLY
    elif (
        direct
        and all(item.state is ListingState.ZERO for item in direct)
        and evidence.search_outcome is SearchOutcome.COMPLETED
    ):
        state = ComponentState.BLOCKED
    else:
        state = ComponentState.UNKNOWN
    return ComponentAssessment(evidence, state, direct, routes)


@dataclass(frozen=True)
class BranchRequirement:
    branch_id: str
    role: str
    identity: str
    quantity: int = 1
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("branch_id", "role", "identity"):
            _text(getattr(self, name), name)
        if type(self.quantity) is not int or self.quantity < 1:
            raise MakeabilityError("branch quantity must be a positive integer")
        for stage in self.stages:
            _text(stage, "stage")


@dataclass(frozen=True)
class BranchEvidence:
    requirement: BranchRequirement
    component: ComponentEvidence


@dataclass(frozen=True)
class BranchAssessment:
    requirement: BranchRequirement
    assessment: ComponentAssessment | None
    reason: str | None = None


class ProductState(str, Enum):
    MAKEABLE = "computationally_makeable"
    NONEXACT_L1 = "nonexact_L1"
    BLOCKED = "blocked_in_supplied_search"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProductAssessment:
    request_id: str
    policy: MakeabilityPolicy
    as_of: datetime
    l1_exact: bool | None
    l1_receipt: Receipt | None
    state: ProductState
    branches: tuple[BranchAssessment, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready full-denominator result, including every input evidence axis."""

        def normalize(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            return value

        result = normalize(asdict(self))
        result.update(
            schema="forge.computational_makeability_assessment.v1",
            required_branches=len(self.branches),
            represented_branches=sum(item.assessment is not None for item in self.branches),
            makeable_branches=sum(
                item.assessment is not None and item.assessment.makeable for item in self.branches
            ),
            strict_dossier_admission=False,
            synthesis_guidance_admission=False,
        )
        return result


def classify_product(
    request_id: str,
    *,
    l1_exact: bool | None,
    l1_receipt: Receipt | None,
    required_branches: tuple[BranchRequirement, ...],
    evidence: tuple[BranchEvidence, ...],
    policy: MakeabilityPolicy,
    as_of: datetime,
) -> ProductAssessment:
    """AND every declared L1 branch, retaining omissions and nonexact requests.

    ``l1_receipt`` must bind the request and its complete required-branch tuple in
    the adapter's source. A bare strict-closed flag is intentionally not an input.
    """
    _text(request_id, "request_id")
    _utc(as_of, "as_of")
    if l1_exact is not None and type(l1_exact) is not bool:
        raise MakeabilityError("l1_exact must be bool or None")
    if l1_receipt is not None and not isinstance(l1_receipt, Receipt):
        raise MakeabilityError("l1_receipt must be a Receipt")
    ids = [item.branch_id for item in required_branches]
    provided = [item.requirement.branch_id for item in evidence]
    if len(set(ids)) != len(ids) or len(set(provided)) != len(provided):
        raise MakeabilityError("duplicate required or supplied branch IDs")
    if set(provided) - set(ids):
        raise MakeabilityError("supplied evidence has undeclared branches")
    by_id = {item.requirement.branch_id: item for item in evidence}
    branches = []
    for requirement in required_branches:
        item = by_id.get(requirement.branch_id)
        if item is None:
            branches.append(BranchAssessment(requirement, None, "missing_branch"))
        elif item.requirement != requirement or item.component.identity != requirement.identity:
            branches.append(BranchAssessment(requirement, None, "branch_binding_mismatch"))
        else:
            branches.append(
                BranchAssessment(
                    requirement, classify_component(item.component, policy=policy, as_of=as_of)
                )
            )
    reasons = []
    if not required_branches:
        reasons.append("no_declared_required_branches")
    if l1_exact is None or l1_receipt is None:
        reasons.append("missing_exact_L1_evidence")
    if l1_exact is False:
        state = ProductState.NONEXACT_L1
    elif reasons or any(item.assessment is None for item in branches):
        state = ProductState.UNKNOWN
    elif all(item.assessment.makeable for item in branches):
        state = ProductState.MAKEABLE
    elif any(item.assessment.state is ComponentState.BLOCKED for item in branches):
        state = ProductState.BLOCKED
    else:
        state = ProductState.UNKNOWN
    return ProductAssessment(
        request_id, policy, as_of, l1_exact, l1_receipt, state, tuple(branches), tuple(reasons)
    )
