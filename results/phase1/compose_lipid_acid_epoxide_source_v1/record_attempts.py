"""Preserve primary-source access observations without admitting unseen reaction schemes."""

from datetime import datetime, timezone
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if (HERE / "acquisition.json").exists():
    raise FileExistsError("Source acquisition record is frozen")
logs = {name: pin(ROOT, HERE / name) for name in ("download-attempt1.log", "download-attempt2.log")}
assert "Could not resolve host" in (HERE / "download-attempt1.log").read_text()
assert "403" in (HERE / "download-attempt2.log").read_text()
assert not (HERE / "source-si.pdf").exists()
dump(
    HERE / "acquisition.json",
    {
        "schema_version": "forge.primary_source_access_attempt.v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": logs,
        "implementation": pin(ROOT, Path(__file__)),
        "doi": "10.1002/adhm.202302691",
        "pmid": "37990414",
        "publisher_article": "https://advanced.onlinelibrary.wiley.com/doi/10.1002/adhm.202302691",
        "publisher_supplement": "https://advanced.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002/adhm.202302691&file=adhm202302691-sup-0001-SuppMat.pdf",
        "web_tool_observations": {
            "publisher_link_resolved": True,
            "pdf_text_index_accessible": True,
            "reported_pdf_pages": 30,
            "reaction_locator": "Figure S1, PDF page 2; compound characterization on PDF page 3",
            "source_compound_label": "E12CA1A3",
            "scheme_screenshots": "cache_miss",
            "direct_pdf_open": "inaccessible",
            "browser_surface": "No browser is available",
        },
        "download_return_codes": {"sandbox": 6, "escalated": 56},
        "pdf_bytes_acquired": False,
        "pdf_sha256": None,
        "reaction_scheme_visually_reviewed": False,
        "disposition": "abstain",
        "reason": "Primary reaction-scheme images and hash-pinnable PDF bytes remain unavailable.",
        "new_exact_chemistry_evidence": 0,
        "training_admitted": False,
        "training_calls": 0,
    },
)
