from __future__ import annotations

import json
from pathlib import Path

from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]


def test_confirmation_policy_is_frozen_and_hashes_inputs() -> None:
    path = REPO / "configs/model/phase1_ugi_decoration_checkpoint_confirmation_policy_v1.json"
    policy = json.loads(path.read_text())

    assert policy["status"] == "frozen_before_independent_confirmation_sampling"
    assert policy["screen_evidence"]["confirmation_shortlist"] == [1500, 3000, 4000]
    assert policy["reference_policy"]["heldout_fold_used_for_selection"] is False
    assert policy["confirmation_design"]["attempted_products_per_checkpoint"] == 3072
    assert len(policy["confirmation_design"]["paired_replicates"]) == 3
    for specification in (
        *policy["finalists"].values(),
        *policy["implementation"].values(),
        policy["confirmation_design"]["program_schedule"],
        policy["reference_policy"]["training_cache"],
    ):
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]
