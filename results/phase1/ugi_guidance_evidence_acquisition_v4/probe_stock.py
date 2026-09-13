"""Save public guest stock observations for four already authenticated listings."""

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main():
    destination = OUT / "stock_responses"
    inputs = [Path(__file__), OUT / "bld_request.cjs"]
    for batch in ("batch_01", "batch_02", "batch_03"):
        receipt = OUT / batch / "receipt.json"
        inputs.append(receipt)
        for record in json.loads(receipt.read_text())["records"]:
            source = record.get("response")
            if source and sha256_file(ROOT / source["path"]) != source["sha256"]:
                raise ValueError(f"Acquired source changed: {source['path']}")
    destination.mkdir(exist_ok=False)
    records = []
    for item in ("BD305903", "BD57579", "BD151911", "AAL0129406"):
        record = {"item": item, "requested_at_utc": datetime.now(timezone.utc).isoformat()}
        if item.startswith("BD"):
            endpoint = "https://www.bldpharm.com/webapi/v1/getproductstock"
            body = json.loads(
                subprocess.check_output(["node", str(OUT / "bld_request.cjs"), item], timeout=5)
            )
            data = urllib.parse.urlencode(body).encode()
            mime = "application/x-www-form-urlencoded"
        else:
            endpoint = "https://www.fishersci.com/shop/products/service/availability"
            body = {
                "parts": [{"partNumber": item, "uom": "EA", "quantity": 1}],
                "callerId": ["products-node-ui-SinglePDP-product"],
                "zipCode": "10001",
            }
            data = json.dumps(body).encode()
            mime = "application/json"
            record["reference_zip_only_not_user_address"] = True
        request_path = destination / f"{item}_request.json"
        write_json(request_path, {"url": endpoint, "method": "POST", "body": body})
        record["request"] = pin_record(request_path, ROOT)
        try:
            request = urllib.request.Request(
                endpoint,
                data=data,
                headers={"User-Agent": "Mozilla/5.0", "Content-Type": mime},
                method="POST",
            )
            try:
                response = urllib.request.urlopen(request, timeout=30)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                payload = response.read(2_000_001)
                record.update(http_status=response.status, resolved_url=response.url)
            if len(payload) > 2_000_000:
                raise ValueError("Supplier response exceeded download bound")
            saved = destination / f"{item}_response.txt"
            saved.write_bytes(payload)
            record.update(response=pin_record(saved, ROOT), status="saved_requires_review")
        except Exception as error:
            record.update(status="retrieval_failed", error=str(error))
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
    write_json(
        destination / "receipt.json",
        {
            "schema_version": "forge.public_supplier_stock_probe.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {str(p.relative_to(ROOT)): pin_record(p, ROOT) for p in inputs},
            "records": records,
            "access_mode": "Public guest lookup; no login, account cookie, cart, or order",
            "evidence_admitted": False,
            "purchase_or_contact_actions": 0,
        },
    )
    print(
        json.dumps(
            [{k: r[k] for k in ("item", "status", "http_status") if k in r} for r in records]
        )
    )


if __name__ == "__main__":
    main()
