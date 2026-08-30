"""Fail-closed orchestration of independently allocated Modal experiment requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._runtime.errors import BackendError, SpecError
from experiments._runtime.modal import launch_modal, modal_request_plan
from experiments.catalog import resolve_specification
from forge.core.hashing import is_sha256, sha256_file
from forge.core.io import read_json_object

MODAL_GROUP_SCHEMA_VERSION = "forge.modal_experiment_group.v1"


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise SpecError(
            f"{label} has invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class ModalGroupMember:
    """One independently allocated experiment in a gated Modal group."""

    member_id: str
    experiment: str
    spec_sha256: str
    profile: str
    replicate: int
    device: str | None

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> ModalGroupMember:
        if not isinstance(value, Mapping):
            raise SpecError(f"{label} must be a JSON object")
        _exact_keys(
            value,
            {"id", "experiment", "spec_sha256", "profile", "replicate", "device"},
            label,
        )
        member_id = value["id"]
        experiment = value["experiment"]
        profile = value["profile"]
        replicate = value["replicate"]
        device = value["device"]
        if not isinstance(member_id, str) or not member_id:
            raise SpecError(f"{label}.id must be a non-empty string")
        if not isinstance(experiment, str) or not experiment:
            raise SpecError(f"{label}.experiment must be a non-empty string")
        if not is_sha256(value["spec_sha256"]):
            raise SpecError(f"{label}.spec_sha256 must be a lowercase SHA-256")
        if profile not in {"smoke", "full"}:
            raise SpecError(f"{label}.profile must be 'smoke' or 'full'")
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise SpecError(f"{label}.replicate must be a non-negative integer")
        if device not in {None, "cpu", "cuda"}:
            raise SpecError(f"{label}.device must be null, 'cpu', or 'cuda'")
        return cls(
            member_id=member_id,
            experiment=experiment,
            spec_sha256=str(value["spec_sha256"]),
            profile=profile,
            replicate=replicate,
            device=device,
        )

    def resolve(self, repo: Path) -> Path:
        path = resolve_specification(repo, self.experiment)
        observed = str(sha256_file(path))
        if observed != self.spec_sha256:
            raise BackendError(
                f"Modal group member {self.member_id!r} specification changed: "
                f"expected {self.spec_sha256}, found {observed}"
            )
        return path


@dataclass(frozen=True)
class ModalExperimentGroup:
    """One preflight followed by independently allocated parallel experiments."""

    group_id: str
    description: str
    preflight: ModalGroupMember
    parallel: tuple[ModalGroupMember, ...]

    @classmethod
    def load(cls, path: Path) -> ModalExperimentGroup:
        value = read_json_object(path, error=SpecError, label="Modal experiment group")
        _exact_keys(
            value,
            {"schema_version", "group_id", "description", "preflight", "parallel"},
            "Modal experiment group",
        )
        if value["schema_version"] != MODAL_GROUP_SCHEMA_VERSION:
            raise SpecError(
                f"unsupported Modal experiment group schema: {value['schema_version']!r}"
            )
        group_id = value["group_id"]
        description = value["description"]
        if not isinstance(group_id, str) or not group_id:
            raise SpecError("Modal experiment group.group_id must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise SpecError("Modal experiment group.description must be a non-empty string")
        parallel_value = value["parallel"]
        if isinstance(parallel_value, (str, bytes)) or not isinstance(parallel_value, Sequence):
            raise SpecError("Modal experiment group.parallel must be an array")
        parallel = tuple(
            ModalGroupMember.from_mapping(item, label=f"Modal experiment group.parallel[{index}]")
            for index, item in enumerate(parallel_value)
        )
        if len(parallel) < 2:
            raise SpecError("Modal experiment group.parallel must contain at least two members")
        preflight = ModalGroupMember.from_mapping(
            value["preflight"], label="Modal experiment group.preflight"
        )
        identifiers = [preflight.member_id, *(member.member_id for member in parallel)]
        if len(identifiers) != len(set(identifiers)):
            raise SpecError("Modal experiment group member ids must be unique")
        return cls(
            group_id=group_id,
            description=description,
            preflight=preflight,
            parallel=parallel,
        )


def modal_group_plan(repo: Path, group_path: Path) -> dict[str, Any]:
    """Validate every request and prove that all members use the same source snapshot."""

    group = ModalExperimentGroup.load(group_path)
    members = (group.preflight, *group.parallel)
    requests: dict[str, dict[str, Any]] = {}
    for member in members:
        requests[member.member_id] = modal_request_plan(
            repo,
            member.resolve(repo),
            profile=member.profile,
            replicate=member.replicate,
            device=member.device,
        )
    source_sha256s = {request["source_sha256"] for request in requests.values()}
    if len(source_sha256s) != 1:
        raise BackendError("Modal group members do not share one executable source fingerprint")
    return {
        "schema_version": "forge.modal_experiment_group_plan.v1",
        "group_id": group.group_id,
        "group_sha256": str(sha256_file(group_path)),
        "source_sha256": next(iter(source_sha256s)),
        "preflight": requests[group.preflight.member_id],
        "parallel": {member.member_id: requests[member.member_id] for member in group.parallel},
    }


LaunchFunction = Callable[..., int]


def launch_modal_group(
    repo: Path,
    group_path: Path,
    *,
    resume: bool = False,
    launch: LaunchFunction = launch_modal,
) -> dict[str, Any]:
    """Run the preflight synchronously, then detach every production member in parallel."""

    plan = modal_group_plan(repo, group_path)
    group = ModalExperimentGroup.load(group_path)
    preflight_path = group.preflight.resolve(repo)
    preflight_status = launch(
        repo,
        preflight_path,
        profile=group.preflight.profile,
        replicate=group.preflight.replicate,
        device=group.preflight.device,
        resume=resume,
        detached=False,
    )
    if preflight_status != 0:
        raise BackendError(
            f"Modal group {group.group_id!r} preflight failed with exit code "
            f"{preflight_status}; no production member was launched"
        )

    # The preflight can run for minutes. Refuse to launch production if the working tree, a pinned
    # input, or a member descriptor changed while it was running. This is the launch-time analogue
    # of the remote source check and prevents a passed preflight from blessing different code.
    for member in group.parallel:
        observed = modal_request_plan(
            repo,
            member.resolve(repo),
            profile=member.profile,
            replicate=member.replicate,
            device=member.device,
        )
        expected = plan["parallel"][member.member_id]
        if observed["request_id"] != expected["request_id"]:
            raise BackendError(
                f"Modal group member {member.member_id!r} changed after preflight; "
                "no production member was launched"
            )

    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(group.parallel)) as executor:
        futures = {
            executor.submit(
                launch,
                repo,
                member.resolve(repo),
                profile=member.profile,
                replicate=member.replicate,
                device=member.device,
                resume=resume,
                detached=True,
            ): member.member_id
            for member in group.parallel
        }
        for future in as_completed(futures):
            statuses[futures[future]] = int(future.result())
    failed = {member_id: status for member_id, status in statuses.items() if status != 0}
    if failed:
        raise BackendError(f"Modal group {group.group_id!r} production failures: {failed}")
    return {
        "schema_version": "forge.modal_experiment_group_result.v1",
        "group_id": group.group_id,
        "group_sha256": plan["group_sha256"],
        "source_sha256": plan["source_sha256"],
        "preflight_exit_code": preflight_status,
        "parallel_exit_codes": dict(sorted(statuses.items())),
        "status": "launched",
    }


__all__ = [
    "MODAL_GROUP_SCHEMA_VERSION",
    "ModalExperimentGroup",
    "ModalGroupMember",
    "launch_modal_group",
    "modal_group_plan",
]
