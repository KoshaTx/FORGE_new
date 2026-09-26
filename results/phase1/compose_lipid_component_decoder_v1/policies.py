"""Explicitly attributed conservative ester restriction for generated Ren tails."""

import json

from forge.core.hashing import resolve_pin
from forge.model.compose_lipid_component_policy import compile_component_policies
from results.phase1.compose_lipid_component_decoder_v1.contracts import MIRROR


def load_proposal_domains():
    origin = json.loads(
        (MIRROR / "results/phase1/compose_lipid_ester_thiol_origins_v1/config.json").read_text()
    )
    config = json.loads(
        resolve_pin(
            origin["contracts"]["ester_thiol"], MIRROR, label="ester proposal contract"
        ).read_text()
    )
    registry = json.loads(
        resolve_pin(
            config["inputs"]["registry"], MIRROR, label="ester proposal registry"
        ).read_text()
    )
    domain = registry["ester_thiol_domain"]
    queries = domain["constraints"]["required_queries"]
    assert len(queries) == 1 and queries[0]["smarts"] == domain["ester_query"]
    return dict(retained_ester=[{**q, "whole_component": True} for q in queries])


def compile_for_executor(executor, domains):
    program = executor.get("program")
    if (
        hasattr(program, "specification")
        and program.specification["program_id"] == "source_ren_2_incorporated_arms"
    ):
        # Source Ren tails retain their complete ester through N-alkylation.
        # Reuse a qualified O-ester query only as a construction restriction;
        # neither the Ren admission contract nor any evidence tier is changed.
        executor = {
            **executor,
            "proposal_retained_queries": {"bromoester_arm": domains["retained_ester"]},
        }
    return compile_component_policies(executor)
