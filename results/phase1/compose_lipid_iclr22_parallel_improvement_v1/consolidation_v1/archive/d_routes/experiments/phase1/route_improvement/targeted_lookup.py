"""One-shot, bounded public listing observation; never admits its own evidence."""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

CODE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3].parents[1]
HERE = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
WORK = (
    ROOT
    / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/computational_makeability_v1"
)
OUT = HERE / "targeted_lookup_v1"
API = "https://pubchem.ncbi.nlm.nih.gov/rest"
TARGETS = {
    "C=CCOC": "FASUFOTUSHAIHG-UHFFFAOYSA-N",
    "OCCCCCCl": "DCBJCKDOZLTTDW-UHFFFAOYSA-N",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("new_listing_common", WORK / "evaluation/common_v2.py")
collector = load_module("new_listing_collector", WORK / "vendors/collector.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    with Path(path).open("x") as stream:
        json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze():
    OUT.mkdir(exist_ok=False)
    proposal = read(HERE / "lookup_proposal_v3.json")
    if proposal["transmitted_identifiers"] != TARGETS:
        raise ValueError("Authorized two-identity request changed")
    protocol = {
        "schema": "forge.two_new_route_leaves_listing_lookup.v1",
        "inputs": [
            pin(__file__),
            pin(CODE / "qualify_targeted_lookup.py"),
            pin(HERE / "lookup_proposal_v3.json"),
            pin(HERE / "next_two_listing_proposal.json"),
            pin(WORK / "evaluation/common_v2.py"),
            pin(WORK / "vendors/collector.py"),
            pin("/usr/bin/curl"),
        ],
        "target_keys": TARGETS,
        "property_requests": proposal["property_requests"],
        "maximum_HTTP_requests": 8,
        "maximum_category_candidates_per_target": 3,
        "maximum_response_bytes_each": 1048576,
        "request_interval_seconds": 1,
        "request_timeout_seconds": 35,
        "maximum_wall_seconds": 180,
        "maximum_CPU_seconds": 20,
        "automatic_retries": 0,
        "redirects_allowed": False,
        "recipient": "https://pubchem.ncbi.nlm.nih.gov",
        "authorization": "Root explicitly authorized these two identifiers and the pinned bounded proposal under existing user authorization for vendor-listing work; no third identity or other provider query.",
        "stop_policy": "First HTTP429/503; two other consecutive systemic failures; no retry.",
        "identity_policy": proposal["identity_policy"],
        "as_of_policy": proposal["as_of_policy"],
        "claim_scope": "Candidate directory-listing evidence only; independent admission required. No stock, strict item, experimental synthesis or scientific metric promotion.",
        "created_at_utc": utc(),
        "seed": 0,
    }
    write(OUT / "protocol.json", protocol)
    print(json.dumps({"protocol": pin(OUT / "protocol.json"), "producer": pin(__file__)}))


def validate(require_controls=True):
    protocol = read(OUT / "protocol.json")
    if protocol["target_keys"] != TARGETS:
        raise ValueError("Target key drift")
    for item in protocol["inputs"]:
        if pin(ROOT / item["path"]) != item:
            raise ValueError("Frozen input changed: " + item["path"])
    expected = (8, 3, 1048576, 1, 35, 180, 0, False)
    observed = tuple(
        protocol[key]
        for key in [
            "maximum_HTTP_requests",
            "maximum_category_candidates_per_target",
            "maximum_response_bytes_each",
            "request_interval_seconds",
            "request_timeout_seconds",
            "maximum_wall_seconds",
            "automatic_retries",
            "redirects_allowed",
        ]
    )
    if observed != expected:
        raise ValueError("Request bounds drift")
    if require_controls:
        controls = read(OUT / "qualification.json")
        if controls["passed"] is not True or controls["protocol"] != pin(OUT / "protocol.json"):
            raise ValueError("Offline controls missing or stale")
    return protocol


def bind_properties(identity, payload):
    rows = payload.get("PropertyTable", {}).get("Properties")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing property rows")
    matches, rejected = {}, []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("CID")) is not int or row["CID"] <= 0:
            raise ValueError("Invalid CID property row")
        reported = row.get("SMILES", row.get("IsomericSMILES"))
        if not isinstance(reported, str):
            rejected.append({"row": row, "reason": "whole_isomeric_SMILES_missing"})
            continue
        try:
            canonical = common.canonical(reported)
        except (ValueError, TypeError):
            rejected.append({"row": row, "reason": "invalid_whole_graph"})
            continue
        if canonical != identity:
            rejected.append(
                {"row": row, "observed_identity": canonical, "reason": "whole_graph_mismatch"}
            )
            continue
        if row["CID"] in matches and matches[row["CID"]] != row:
            raise ValueError("Conflicting duplicate CID property rows")
        matches[row["CID"]] = row
    return [matches[cid] for cid in sorted(matches)], rejected


def bind_category(cid, payload):
    category = payload.get("SourceCategories", {})
    if (
        category.get("RecordType") != "CID"
        or type(category.get("RecordNumber")) is not int
        or category["RecordNumber"] != cid
    ):
        raise ValueError("Vendor category response is not bound to the exact requested CID")
    return collector.vendor_sources(payload)


class Client:
    def __init__(self, protocol, output):
        self.protocol = protocol
        self.output = output
        self.calls = 0
        self.started = time.monotonic()
        self.last_request = self.started - 1
        self.stop = None
        self.errors = 0

    def get(self, url, label):
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "pubchem.ncbi.nlm.nih.gov"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Unexpected recipient or URL structure")
        if self.stop or self.calls >= 8:
            raise RuntimeError(self.stop or "HTTP_budget_exhausted")
        delay = max(0, 1 - (time.monotonic() - self.last_request))
        if time.monotonic() - self.started + delay + 35 > 180:
            self.stop = "insufficient_remaining_wall_budget"
            raise RuntimeError(self.stop)
        if delay:
            time.sleep(delay)
        directory = self.output / "http" / f"{self.calls + 1:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        write(
            directory / "started.json",
            {
                "url": url,
                "label": label,
                "sequence": self.calls + 1,
                "requested_at_utc": utc(),
                "protocol": pin(OUT / "protocol.json"),
            },
        )
        self.calls += 1
        self.last_request = time.monotonic()
        command = [
            "/usr/bin/curl",
            "--disable",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--connect-timeout",
            "10",
            "--max-time",
            "35",
            "--max-filesize",
            "1048576",
            "--header",
            "User-Agent: FORGE-vendor-screen/1.0",
            "--dump-header",
            str(directory / "headers.txt"),
            "--output",
            str(directory / "response.bin"),
            "--write-out",
            "%{http_code}\n%{url_effective}\n",
            url,
        ]
        started_at = utc()
        try:
            response = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=35,
                check=False,
            )
            code, stdout, stderr = response.returncode, response.stdout, response.stderr
        except subprocess.TimeoutExpired as error:
            code, stdout, stderr = -1, "", "transport_process_timeout:" + str(error)
        fields = stdout.splitlines()
        status = int(fields[0]) if fields and fields[0].isdigit() else 0
        for filename in ["response.bin", "headers.txt"]:
            path = directory / filename
            if not path.exists():
                path.write_bytes(b"")
        body_path = directory / "response.bin"
        body = body_path.read_bytes()
        if len(body) > 1048576:
            self.stop = "response_limit_violation"
            code = -2
        receipt = {
            "schema": "forge.vendor_http_receipt.v1",
            "url": url,
            "effective_url": fields[1] if len(fields) > 1 else None,
            "started_at_utc": started_at,
            "observed_at_utc": utc(),
            "http_status": status,
            "curl_exit_code": code,
            "stderr": stderr,
            "body_bytes": len(body),
            "body": pin(body_path),
            "headers": pin(directory / "headers.txt"),
            "protocol": pin(OUT / "protocol.json"),
        }
        write(directory / "receipt.json", receipt)
        systemic = code != 0 or status not in (200, 404)
        self.errors = self.errors + 1 if systemic else 0
        if status in (429, 503):
            self.stop = "first_rate_limit_or_server_busy"
        elif self.errors >= 2:
            self.stop = "two_consecutive_systemic_failures"
        return receipt, pin(directory / "receipt.json"), body


def observe_all(client, save_progress):
    observations = []
    for identity, key in TARGETS.items():
        row = {
            "identity": identity,
            "query_inchikey": key,
            "outcome": "unknown",
            "vendor_count": None,
            "observed_at": None,
            "status_reason": "not_attempted",
            "source_receipts": [],
            "identity_bindings": [],
            "rejected_identity_bindings": [],
            "category_observations": [],
            "provider": "PubChem Chemical Vendors",
            "stock_claim": False,
        }
        observations.append(row)
        if client.stop:
            row["status_reason"] = "not_attempted_after_stop"
            continue
        try:
            receipt, reference, body = client.get(
                API + "/pug/compound/inchikey/" + key + "/property/SMILES,InChIKey/JSON",
                "exact_identity_property",
            )
            row["source_receipts"].append(reference)
            row["observed_at"] = receipt["observed_at_utc"]
            row["status_reason"] = "identity_lookup_failed_or_unknown"
            save_progress(observations, client)
            if receipt["curl_exit_code"] or receipt["http_status"] != 200:
                continue
            matches, rejected = bind_properties(identity, json.loads(body))
            row["identity_bindings"], row["rejected_identity_bindings"] = matches, rejected
            if not matches:
                row["status_reason"] = "no_exact_whole_graph_match"
                continue
            if len(matches) > 3:
                row["status_reason"] = "CID_count_exceeds_frozen_cap_censored"
                continue
            row["status_reason"] = "no_observed_vendor_directory_evidence"
            for candidate in matches:
                if client.stop:
                    break
                cid = candidate["CID"]
                receipt, reference, body = client.get(
                    API + f"/pug_view/categories/compound/{cid}/JSON", "exact_CID_vendor_categories"
                )
                row["source_receipts"].append(reference)
                row["observed_at"] = receipt["observed_at_utc"]
                observation = {
                    "CID": cid,
                    "receipt": reference,
                    "observed_at": receipt["observed_at_utc"],
                    "http_status": receipt["http_status"],
                    "vendor_sources": [],
                    "status": "unknown",
                }
                row["category_observations"].append(observation)
                save_progress(observations, client)
                if receipt["curl_exit_code"] or receipt["http_status"] != 200:
                    continue
                try:
                    names = bind_category(cid, json.loads(body))
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    observation["parse_error"] = str(error)
                    continue
                observation.update(
                    vendor_sources=names,
                    status=(
                        "positive_directory_evidence" if names else "no_vendor_category_observed"
                    ),
                )
            positives = [
                r
                for r in row["category_observations"]
                if r["status"] == "positive_directory_evidence"
            ]
            if positives:
                row.update(
                    outcome="success",
                    vendor_count=1,
                    status_reason="candidate_vendor_listing_pending_independent_admission",
                    observed_at=max(r["observed_at"] for r in positives),
                )
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
            row["status_reason"] = "unknown_error:" + str(error)
        finally:
            save_progress(observations, client)
    return observations


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    protocol = validate()
    write(
        OUT / "run_started.json",
        {"pid": os.getpid(), "protocol": pin(OUT / "protocol.json"), "started_at_utc": utc()},
    )
    client = Client(protocol, OUT)
    counter = 0

    def wall_stop(*_):
        client.stop = "wall_budget_exhausted"
        raise RuntimeError(client.stop)

    signal.signal(signal.SIGALRM, wall_stop)
    signal.alarm(180)

    def progress(rows, active):
        nonlocal counter
        counter += 1
        write(
            OUT / f"progress_{counter:03d}.json",
            {
                "observations": rows,
                "HTTP_requests": active.calls,
                "stop_reason": active.stop,
                "at_utc": utc(),
            },
        )

    observations = observe_all(client, progress)
    signal.alarm(0)
    listings = []
    (OUT / "bindings").mkdir(exist_ok=False)
    for row in observations:
        path = OUT / "bindings" / (hashlib.sha256(row["identity"].encode()).hexdigest() + ".json")
        write(
            path,
            {
                "schema": "forge.vendor_listing_binding.v1",
                "protocol": pin(OUT / "protocol.json"),
                **row,
            },
        )
        if row["outcome"] == "success":
            listings.append(
                {
                    key: row[key]
                    for key in ["identity", "outcome", "vendor_count", "observed_at", "provider"]
                }
                | {"receipt": pin(path)}
            )
    write(
        OUT / "candidate_vendor_listings.json",
        {
            "schema": "forge.vendor_listings.v1",
            "candidate_only": True,
            "independently_admitted": False,
            "protocol": pin(OUT / "protocol.json"),
            "listings": listings,
        },
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "schema": "forge.two_leaf_listing_observation_result.v1",
        "protocol": pin(OUT / "protocol.json"),
        "observations": observations,
        "HTTP_requests": client.calls,
        "declared_targets": 2,
        "candidate_positive_identities": len(listings),
        "stop_reason": client.stop,
        "candidate_envelope": pin(OUT / "candidate_vendor_listings.json"),
        "completed_at_utc": utc(),
        "wall_seconds": time.monotonic() - client.started,
        "process_CPU_seconds": usage.ru_utime + usage.ru_stime,
        "transport_CPU_seconds": children.ru_utime + children.ru_stime,
        "new_official_results": 0,
        "independent_admission_required": True,
        "lookup_absence_is_unmakeable": False,
    }
    write(OUT / "result.json", result)


def launch():
    validate()
    write(
        OUT / "launch_intent.json",
        {"protocol": pin(OUT / "protocol.json"), "source": pin(__file__), "created_at_utc": utc()},
    )
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "run"]
    with (OUT / "stdout.log").open("xb") as stdout, (OUT / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write(
        OUT / "launch_receipt.json",
        {
            "pid": process.pid,
            "command": command,
            "detached": True,
            "protocol": pin(OUT / "protocol.json"),
            "qualification": pin(OUT / "qualification.json"),
        },
    )
    print(json.dumps({"pid": process.pid, "receipt": pin(OUT / "launch_receipt.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "run"])
    args = parser.parse_args()
    globals()[args.action]()
