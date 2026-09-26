"""Resolve documentary locators by exact pinned source digest, never by chemistry."""
from common import ROOT, authenticate, load, pin
from forge.synthesis.assessment.computational_makeability import Receipt

CATALOGS = (
    ROOT / "data/vendor/qualified_zhou_aema_source_program_v1.json",
    ROOT / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json",
)
RESOLUTIONS = {}


def source_assets():
    found = {}
    def walk(value):
        if isinstance(value, dict):
            path = value.get("path", value.get("asset"))
            digest = value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                found.setdefault(digest, []).append(path)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for catalog in CATALOGS:
        walk(load(catalog))
    return found


ASSETS = source_assets()


def resolve(location, digest):
    candidates = [location]
    if location.startswith("/") and "/results/" in location:
        candidates.append("results/" + location.split("/results/", 1)[1])
    candidates.extend(ASSETS.get(digest, []))
    errors = []
    for path in candidates:
        try:
            authenticate(path, digest)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))
            continue
        RESOLUTIONS[(location, digest)] = {"original_locator": location, "sha256": digest,
                                         "authenticated_location": path,
                                         "catalogs": [pin(p) for p in CATALOGS]}
        return Receipt(path, digest)
    raise ValueError(f"No authenticated exact source bytes for {location}: {errors}")
