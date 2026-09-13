"""Exercise only the frozen source loader, without sampling or candidate assessments."""

from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1.synthesis_guidance.current_sampler_preflight import inspect_route_source
from forge.core.hashing import pin_record
from forge.core.io import write_json

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent / "route_source_recheck.json"
if out.exists():
    raise ValueError("Preserve the existing source recheck")
source = repo / "configs/route/phase1_ugi3_source_qualified_cumulative_inputs_v1.json"
historical = repo / "configs/model/phase1_ugi_production_synthesis_guidance_seam_v4.json"
record = inspect_route_source(repo, source, historical)
record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
record["inputs"] = {
    "source": pin_record(source, repo),
    "historical": pin_record(historical, repo),
    "script": pin_record(Path(__file__), repo),
    "loader": pin_record(
        repo / "experiments/phase1/synthesis_guidance/current_sampler_preflight.py", repo
    ),
}
write_json(out, record)
print(record["status"])
for error in record.get("errors", []):
    print(error["message"])
