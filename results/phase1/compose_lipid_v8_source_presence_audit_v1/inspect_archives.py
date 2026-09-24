"""Read archive member names only; never extract or execute archived content."""

import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import sha256_file  # noqa: E402


def main():
    inventory = OUTPUT / "local-file-inventory.txt"
    paths = [Path(line) for line in inventory.read_text().splitlines()]
    archives = sorted(
        {
            path
            for path in paths
            if path.name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            and not any(part.startswith((".venv", "venv")) for part in path.parts)
        }
    )
    output = []
    digests = {}
    for number, path in enumerate(archives):
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in digests:
            digests[identity] = str(sha256_file(path))
        record = {"path": str(path), "sha256": digests[identity], "members": []}
        try:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    record["members"] = sorted(archive.namelist())
            else:
                with tarfile.open(path, "r:*") as archive:
                    record["members"] = sorted(member.name for member in archive)
            record["status"] = "member_names_inspected_no_contents_extracted"
        except (zipfile.BadZipFile, tarfile.TarError) as error:
            record["status"] = "not_a_readable_archive"
            record["error"] = str(error)
        output.append(record)
        if number % 25 == 0:
            print(f"Inspected archive {number + 1}/{len(archives)}", flush=True)
    result = {
        "schema_version": "forge.compose_lipid_local_archive_inventory.v1",
        "inputs": {
            "file_inventory": {
                "path": inventory.relative_to(ROOT).as_posix(),
                "sha256": str(sha256_file(inventory)),
            },
            "implementation": {
                "path": Path(__file__).relative_to(ROOT).as_posix(),
                "sha256": str(sha256_file(Path(__file__))),
            },
        },
        "archives": output,
        "seed": 0,
        "random_sampling_used": False,
        "contents_extracted": False,
        "training_calls": 0,
    }
    (OUTPUT / "archive-members.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
