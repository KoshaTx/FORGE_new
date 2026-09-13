"""Summarize authenticated public stock replies without admitting them to guidance."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import write_json

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
result_path = out / "supplier_stock_summary.json"
if result_path.exists():
    raise ValueError("Preserve the existing stock summary")
inputs = {"script": pin_record(Path(__file__), repo)}
rows = []
for directory in ("supplier_stock_responses", "supplier_variant_stock_responses"):
    receipt_path = out / directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    inputs[directory] = pin_record(receipt_path, repo)
    for spec in receipt["inputs"].values():
        if sha256_file(repo / spec["path"]) != spec["sha256"]:
            raise ValueError("Stock probe source changed")
        inputs[spec["path"]] = spec
    for record in receipt["records"]:
        for label in ("request", "response"):
            spec = record[label]
            if sha256_file(repo / spec["path"]) != spec["sha256"]:
                raise ValueError("Stock request or response changed")
            inputs[spec["path"]] = spec
        request = json.loads((repo / record["request"]["path"]).read_text())
        response = json.loads((repo / record["response"]["path"]).read_text())
        if record["http_status"] != 200 or response.get("message") != "success":
            raise ValueError("Supplier stock response was not successful")
        parts = response["priceAndAvailability"]
        if set(parts) != {record["part"]} or len(parts[record["part"]]) != 1:
            raise ValueError("Supplier response does not match the exact requested SKU")
        entry = parts[record["part"]][0]
        if entry["quantity"] != "1" or entry["uom"] != "EA" or request["zipCode"] != "10001":
            raise ValueError("Stock probe scope differs")
        available, backordered = int(entry["availableQuantity"]), int(entry["backOrderedQuantity"])
        display = " ".join(
            re.sub(
                "<[^>]+>", " ", re.sub("<style>.*?</style>", "", entry["availability"], flags=re.S)
            ).split()
        )
        positive = available > 0 and backordered == 0 and "In Stock" in display
        if positive and not entry.get("estimatedShipDate"):
            raise ValueError("Positive stock response lacks the displayed shipping date")
        rows.append(
            {
                "part": record["part"],
                "requested_at_utc": record["requested_at_utc"],
                "reference_zip": request["zipCode"],
                "user_address_used": False,
                "request": record["request"],
                "response": record["response"],
                "stock_observation": (
                    "positive_available_quantity_and_in_stock_label"
                    if positive
                    else "backorder_or_supplier_estimate"
                ),
                "available_quantity": available,
                "backordered_quantity": backordered,
                "display": display,
                "estimated_ship_date": entry.get("estimatedShipDate"),
                "backordered_ship_date": entry.get("backOrderedShipDate"),
                "lead_time_field": entry.get("leadTime"),
                "lead_time_message": entry.get("leadTimeMessage"),
                "stocked_item_field": entry.get("stockedItem"),
                "from_cache": response.get("fromCache"),
                "admitted_to_guidance": False,
            }
        )
result = {
    "schema_version": "forge.public_supplier_stock_summary.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "inputs": inputs,
    "rows": rows,
    "lookup_count": len(rows),
    "positive_stock_responses": sum(
        r["stock_observation"] == "positive_available_quantity_and_in_stock_label" for r in rows
    ),
    "backorder_or_estimate_responses": sum(
        r["stock_observation"] == "backorder_or_supplier_estimate" for r in rows
    ),
    "interpretation": "One protected-propargyl-alcohol pack has a positive supplier-stock response. Three other target materials have only backorder/estimate responses in the checked listings. This is not a survey of all suppliers or proof of chemical impossibility.",
    "qualification": "The positive Fisher observation requires separate source qualification; it is not consumed by the frozen cumulative source.",
    "cart_changes": 0,
    "supplier_contacts": 0,
    "guidance_run_ready": False,
}
write_json(result_path, result)
print(
    json.dumps(
        {
            k: result[k]
            for k in ("lookup_count", "positive_stock_responses", "backorder_or_estimate_responses")
        },
        indent=2,
    )
)
