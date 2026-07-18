from __future__ import annotations

import gzip
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any


# Loads plain or gzip-compressed JSON so, unlike the original, no manual extraction is required.
def read_json_auto(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


# Resolves a benchmark data file with or without gzip compression to make input loading robust.
def resolve_data_file(root: Path, stem: str) -> Path:
    candidates = (root / stem, root / f"{stem}.gz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required data file is missing. Expected {candidates[0]} or {candidates[1]}."
    )


# Finds the task's perturbed C or C++ mask so every runner uses the intended benchmark input.
def resolve_mask_file(root: Path, task_id: str) -> Path:
    base = root / "descriptions" / task_id / "mask_desc_perturbed"
    for suffix in (".c", ".cpp"):
        candidate = Path(str(base) + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Task {task_id}: missing perturbed masked source file under {base.parent}"
    )


# Hashes an artifact so experiment inputs and outputs can be verified and reproduced later.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Loads and validates the shared task metadata used by the harness and experiment pipeline.
def load_sample_metadata(root: Path) -> dict[str, Any]:
    metadata = read_json_auto(resolve_data_file(root, "sample_metadata.json"))
    if not isinstance(metadata, dict):
        raise ValueError("SecRepoBench sample metadata must be a JSON object.")
    return metadata


# Loads the project-to-URL mapping once so all runners resolve repositories consistently.
def load_repository_urls(root: Path) -> dict[str, str]:
    rows = read_json_auto(root / "github_repos.json")
    if not isinstance(rows, list):
        raise ValueError("SecRepoBench repository metadata must be a JSON array.")
    return {
        str(row["project"]): str(row["repo_addr"])
        for row in rows
        if isinstance(row, dict) and row.get("project") and row.get("repo_addr")
    }


# Stops a process and its children so timeouts or interruptions leave no work running.
def terminate_process_group(
    process: subprocess.Popen[Any], grace_seconds: int = 10
) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            return
