"""Read the supplier's public guest availability endpoint; never modify a cart."""

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record
from forge.core.io import write_json

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent / "supplier_variant_stock_responses"
out.mkdir(exist_ok=False)
endpoint = "https://www.fishersci.com/shop/products/service/availability"
records = []
for part in ("AC295060050", "AC295060250", "1924225G", "H12951G", "AAL0129403", "5000280479"):
    body = {
        "parts": [{"partNumber": part, "uom": "EA", "quantity": 1}],
        "callerId": ["products-node-ui-SinglePDP-product"],
        "zipCode": "10001",
    }
    request_path = out / f"{part}_request.json"
    write_json(request_path, body)
    record = {
        "part": part,
        "url": endpoint,
        "method": "POST",
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": pin_record(request_path, repo),
        "reference_zip_only_not_user_address": True,
    }
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
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
            raise ValueError("Response too large")
        saved = out / f"{part}_response.txt"
        saved.write_bytes(payload)
        record.update(response=pin_record(saved, repo), status="saved_requires_review")
    except Exception as error:
        record.update(status="retrieval_failed", error=str(error))
    records.append(record)
report = {
    "schema_version": "forge.public_supplier_stock_probe.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "inputs": {
        "script": pin_record(Path(__file__), repo),
        "public_endpoint_definition": pin_record(
            out.parent / "supplier_runtime_probe/plusMinusInputAvailability.js", repo
        ),
    },
    "scope": "Public availability lookup for reference ZIP 10001; not a user destination or order.",
    "records": records,
    "cart_changes": 0,
    "supplier_messages": 0,
    "purchases": 0,
    "evidence_admitted": False,
}
write_json(out / "receipt.json", report)
print(
    json.dumps(
        [{k: r[k] for k in ("part", "status", "http_status", "error") if k in r} for r in records],
        indent=2,
    )
)
