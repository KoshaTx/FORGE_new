"""Offline response/identity/stop controls for the exact two-key lookup."""

import json
import resource
import time

import targeted_lookup as run


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.stop = None
        self.calls = 0
        self.urls = []

    def get(self, url, label):
        status, payload = next(self.responses)
        self.calls += 1
        self.urls.append(url)
        if status in (429, 503):
            self.stop = "first_rate_limit_or_server_busy"
        return (
            {
                "curl_exit_code": 0,
                "http_status": status,
                "observed_at_utc": "2026-09-26T02:00:00+00:00",
            },
            {"path": "synthetic_only", "sha256": "0" * 64},
            json.dumps(payload).encode(),
        )


def properties(rows):
    return {"PropertyTable": {"Properties": rows}}


def categories(cid):
    return {
        "SourceCategories": {
            "RecordType": "CID",
            "RecordNumber": cid,
            "Categories": [
                {"Category": "Chemical Vendors", "Sources": [{"SourceName": "synthetic control"}]}
            ],
        }
    }


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    cpu = time.process_time()
    run.validate(require_controls=False)
    identities = list(run.TARGETS)
    good = {"CID": 42, "SMILES": identities[0]}
    wrong = [
        {"CID": 1, "SMILES": "C=CC[OH+]C.[Cl-]"},
        {"CID": 2, "SMILES": "C=CCO[13CH3]"},
        {"CID": 3, "ConnectivitySMILES": identities[0]},
        {"CID": 4, "SMILES": "CC=CCO"},
    ]
    matches, rejected = run.bind_properties(identities[0], properties(wrong + [good]))
    assert matches == [good] and len(rejected) == 4
    try:
        run.bind_category(42, categories(41))
    except ValueError:
        pass
    else:
        raise AssertionError("CID mismatch admitted")
    fake = FakeClient([(200, properties(wrong + [good])), (200, categories(42)), (404, {})])
    rows = run.observe_all(fake, lambda *_: None)
    assert len(rows) == 2 and rows[0]["outcome"] == "success" and rows[1]["outcome"] == "unknown"
    assert fake.calls == 3 and rows[0]["vendor_count"] == 1
    capped = FakeClient(
        [(200, properties([{"CID": n, "SMILES": identities[0]} for n in range(1, 5)])), (404, {})]
    )
    rows = run.observe_all(capped, lambda *_: None)
    assert capped.calls == 2 and rows[0]["status_reason"] == "CID_count_exceeds_frozen_cap_censored"
    assert len(rows[0]["identity_bindings"]) == 4
    for code in (429, 503):
        stopped = FakeClient([(code, {})])
        rows = run.observe_all(stopped, lambda *_: None)
        assert stopped.calls == 1 and len(rows) == 2
        assert rows[1]["status_reason"] == "not_attempted_after_stop"
    spoofed = FakeClient([(200, properties([good])), (200, categories(41)), (404, {})])
    rows = run.observe_all(spoofed, lambda *_: None)
    assert all(row["outcome"] == "unknown" for row in rows)
    invalid = FakeClient(
        [
            (200, {"PropertyTable": {"Properties": [{"CID": True, "SMILES": identities[0]}]}}),
            (404, {}),
        ]
    )
    rows = run.observe_all(invalid, lambda *_: None)
    assert all(row["outcome"] == "unknown" for row in rows)
    literal_urls = [r["url"] for r in run.read(run.OUT / "protocol.json")["property_requests"]]
    assert fake.urls[0] == literal_urls[0] and fake.urls[2] == literal_urls[1]
    checks = [
        "salt/charge rejected",
        "isotope rejected",
        "connectivity-only response rejected",
        "constitution mismatch rejected",
        "all property candidates retained",
        "CID category spoof rejected",
        "positive and404 remain distinct",
        "over-cap candidates retained and censored",
        "429 stops before second target",
        "503 stops before second target",
        "boolean CID rejected",
        "exact two authorized property URLs reproduced",
    ]
    run.write(
        run.OUT / "qualification.json",
        {
            "passed": True,
            "protocol": run.pin(run.OUT / "protocol.json"),
            "producer": run.pin(__file__),
            "runner": run.pin(run.__file__),
            "checks": checks,
            "provider_calls": 0,
            "seed": 0,
            "CPU_seconds": time.process_time() - cpu,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "checks": len(checks),
                "qualification": run.pin(run.OUT / "qualification.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
