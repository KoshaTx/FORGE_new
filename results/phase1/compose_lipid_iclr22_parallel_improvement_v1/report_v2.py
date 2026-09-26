"""Add the complete component-support census to the frozen campaign snapshot."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def verify(saved):
    path = Path(saved["path"])
    if not path.is_absolute():
        path = ROOT / path
    assert digest(path) == saved["sha256"], str(path)


def main():
    previous = HERE / "result.json"
    assert digest(previous) == "805fb704dc26cf185e3f9ca0f1e241e9c4a5a30798686c5401a8bbd599ed8f73"
    result = read(previous)
    directory = HERE / "b_quality/component_feasibility_v1/run_v2"
    census = read(directory / "result.json")
    protocol = read(directory / "protocol.json")
    review = read(directory / "verification.json")
    review_protocol = read(directory / "verification_protocol.json")
    for saved in [*protocol["inputs"].values(), *review_protocol["inputs"].values()]:
        verify(saved)
    for field in ("protocol", "TRAIN_profiles", "head_witnesses"):
        verify(census[field])
    verify(review["protocol"])
    assert census["complete"] and review["complete"] and review["passed"]
    assert census["CPU_seconds"] <= protocol["CPU_seconds_cap"]
    assert review["CPU_seconds"] <= review_protocol["CPU_cap"]
    assert census["new_molecules_model_source_executor_TEST_network_GPU_calls"] == 0
    assert not review["full_assembly_compatibility_admitted"]
    requests = census["requests"]
    assert len(requests) == len({row["request"] for row in requests}) == 128
    counts = {}
    for family, summary in census["summary"].items():
        rows = [row for row in requests if row["family"] == family]
        assert len(rows) == 64
        assert all(row["all_precursor_roles_have_necessary_support"] for row in rows)
        failures = sum(not row["joint_design"] for row in rows)
        assert failures == summary["joint_design_failures"]
        assert failures == summary["joint_design_failures_with_all_role_support"]
        counts[family] = {"requests": len(rows), "limited_design_failures": failures}
        for row in rows:
            assert row["full_assembly_compatibility"].startswith("unassessed_")
            assert all(part["matching_component_ids"] for part in row["roles"].values())
    ketone = review["role_details"]["ketone_ugi4"]["coupled_ketone"]
    assert ketone["joint_component_observations"] == 62
    assert ketone["joint_role_TRAIN_novel_observations"] == 62
    assert ketone["count_only_upper_bound_on_known_replacement_without_novelty_compensation"] == 0
    assert sum(value["limited_design_failures"] for value in counts.values()) == 33
    for saved in read(HERE / "baseline/result.json")["paper_sources"]:
        verify(saved)
    inputs = [directory / name for name in (
        "result.json", "protocol.json", "verification.json", "verification_protocol.json",
        "TRAIN_profiles.json", "head_witnesses.json", "profile.json",
    )]
    result.update({
        "schema": "forge.iclr22.parallel_improvement_development_snapshot.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": pin(Path(__file__)),
        "previous_snapshot": pin(previous),
        "inputs": [*result["inputs"], *(pin(path) for path in inputs)],
        "component_feasibility": {
            "requests": 128,
            "TRAIN_graphs": census["TRAIN_graphs"],
            "by_family": counts,
            "all_roles_have_necessary_support": True,
            "full_assembly_compatibility_admitted": False,
            "census_CPU_seconds": census["CPU_seconds"],
            "verification_CPU_seconds": review["CPU_seconds"],
            "summary": census["summary"],
            "role_details": review["role_details"],
            "novelty_limit": review["novelty_limit"],
            "interpretation": (
                "Rules out empty necessary TRAIN support for these TRAIN-derived request layouts. "
                "Does not establish independent realism, source-disjoint feasibility, full mixed "
                "assembly compatibility, or causal attribution to training versus decoding."
            ),
        },
        "additional_work_in_progress": (
            "Bounded component census complete. Corrected paid null retry awaits fresh user "
            "authorization; novel-component intervention and independent final evaluation remain open."
        ),
    })
    output = HERE / "result_v2.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"result": pin(output), "joint": result["joint"], "census": counts}))


if __name__ == "__main__":
    main()
