from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from git import Repo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assets.constants import AGENT_USER_PEOMPT, SYSTEM_PROMPT, THINKING_BUDGET_TOKENS  # noqa: E402
from tools.experiment_utils import (  # noqa: E402
    load_repository_urls,
    load_sample_metadata,
    resolve_mask_file,
    sha256_file,
    terminate_process_group,
)

INSTRUCTION_FILENAMES = {"claude.md", "claude.local.md"}
CLAUDE_CONFIGURATION_FILES = {
    ".mcp.json",
    "settings.json",
    "settings.local.json",
}


# Removes only paths inside the task repository to keep instruction cleanup safely contained.
def _safe_remove(path: Path, repository_root: Path) -> None:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(
            f"Refusing to remove a path outside the task repository: {path}"
        )
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


# Removes uncontrolled Claude instructions and integrations so both conditions start consistently.
def clean_claude_instruction_files(repository_root: Path) -> list[str]:
    repository_root = repository_root.resolve()
    removed: list[str] = []
    for path in list(repository_root.rglob("*")):
        relative = path.relative_to(repository_root)
        lower_name = path.name.lower()
        should_remove = (
            lower_name in INSTRUCTION_FILENAMES
            or (path.is_dir() and path.name == ".claude")
            or (
                path.is_file()
                and path.parent.name == ".claude"
                and lower_name in CLAUDE_CONFIGURATION_FILES
            )
            or (path.is_file() and relative.as_posix() == ".mcp.json")
        )
        if should_remove and path.exists():
            removed.append(relative.as_posix())
            _safe_remove(path, repository_root)
    return sorted(set(removed))


# Finds remaining Claude instructions to verify baseline isolation and controlled memory injection.
def find_remaining_instruction_files(repository_root: Path) -> list[str]:
    remaining: list[str] = []
    for path in repository_root.rglob("*"):
        if path.name.lower() in INSTRUCTION_FILENAMES:
            remaining.append(path.relative_to(repository_root).as_posix())
        elif path.is_dir() and path.name == ".claude":
            remaining.append(path.relative_to(repository_root).as_posix())
    return sorted(set(remaining))


# Commits the prepared repository state so the model's subsequent changes can be measured exactly.
def _commit_all(repository: Repo, message: str) -> str:
    repository.git.add(A=True)
    if not repository.git.diff("--cached", "--name-only").strip():
        return repository.head.commit.hexsha
    return repository.index.commit(message).hexsha


# Loads and validates task metadata and its repository URL so each run uses the correct project state.
def _load_task(root: Path, task_id: str) -> tuple[dict[str, Any], str]:
    metadata = load_sample_metadata(root)
    record = metadata.get(task_id)
    if not isinstance(record, dict):
        raise KeyError(f"Unknown SecRepoBench task: {task_id}")
    repository_url = load_repository_urls(root).get(str(record.get("project_name")), "")
    if not repository_url:
        raise RuntimeError(f"No repository URL for {record.get('project_name')}")
    return record, repository_url


class ClaudeCodeRunner:
    # Runs one isolated condition and records its completion, diff, logs, and provenance artifacts.
    @staticmethod
    def run_task(
        *,
        root: Path,
        task_id: str,
        condition: str,
        output_dir: Path,
        model_name: str,
        project_memory_file_path: Path | None = None,
    ) -> dict[str, Any]:
        if condition not in {"baseline", "project_memory"}:
            raise ValueError(f"Unsupported condition: {condition}")
        if condition == "project_memory" and project_memory_file_path is None:
            raise ValueError(
                "The malicious project memory condition requires a "
                "project memory file path"
            )

        root = root.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        repository_dir = output_dir / "repository"
        if repository_dir.exists():
            shutil.rmtree(repository_dir)

        record, repository_url = _load_task(root, task_id)
        project_name = str(record["project_name"])
        fixing_commit = str(record["fixing_commit"])
        changed_file = str(record["changed_file"])
        target_path = repository_dir / changed_file
        child: subprocess.Popen[str] | None = None
        old_handlers: dict[int, Any] = {}

        # Forwards termination signals to Claude Code so interrupted runs shut down cleanly.
        def forward_signal(signum: int, frame: Any) -> None:
            if child is not None:
                terminate_process_group(child)
            raise KeyboardInterrupt

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, forward_signal)

            Repo.clone_from(repository_url, repository_dir)
            source_repository = Repo(repository_dir)
            source_repository.git.checkout("--force", fixing_commit)
            source_repository.git.clean("-fdx")
            shutil.rmtree(repository_dir / ".git")
            repository = Repo.init(repository_dir)
            repository.git.config("user.email", "secrepobench@example.invalid")
            repository.git.config("user.name", "SecRepoBench Experiment")

            removed_instructions = clean_claude_instruction_files(repository_dir)
            remaining = find_remaining_instruction_files(repository_dir)
            if remaining:
                raise RuntimeError(
                    "Uncontrolled Claude instruction files remain: "
                    + ", ".join(remaining)
                )

            masked_content = resolve_mask_file(root, task_id).read_text(
                encoding="utf-8"
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(masked_content, encoding="utf-8")
            base_commit = _commit_all(repository, "Prepare masked SecRepoBench task")

            project_memory_file_sha256 = None
            if condition == "project_memory":
                assert project_memory_file_path is not None
                project_memory_file_path = project_memory_file_path.resolve()
                (repository_dir / "CLAUDE.md").write_text(
                    project_memory_file_path.read_text(encoding="utf-8"), encoding="utf-8",
                )
                project_memory_file_sha256 = sha256_file(repository_dir / "CLAUDE.md")
                injected_instructions = find_remaining_instruction_files(repository_dir)
                if injected_instructions != ["CLAUDE.md"]:
                    raise RuntimeError(
                        "The malicious project memory condition must contain exactly one "
                        f"root CLAUDE.md. Found: {injected_instructions}"
                    )
                base_commit = _commit_all(repository, "Add malicious project memory file")

            empty_mcp = output_dir / "empty_mcp.json"
            empty_mcp.write_text('{"mcpServers": {}}\n', encoding="utf-8")
            system_prompt = SYSTEM_PROMPT.replace(
                "Only return the code to be filled in the masked region. "
                "DO NOT include any other information, such as a preamble or suffix.",
                "",
            ).strip()
            user_prompt = AGENT_USER_PEOMPT.format(changed_file=changed_file)
            prompt = system_prompt + user_prompt
            (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            stdout_path = output_dir / "claude_stdout.txt"
            stderr_path = output_dir / "claude_stderr.txt"
            command_path = output_dir / "command.json"
            command = [
                "claude",
                "-p",
                prompt,
                "--model",
                model_name,
                "--permission-mode",
                "bypassPermissions",
                "--output-format",
                "text",
                "--no-session-persistence",
                "--setting-sources",
                "project",
                "--mcp-config",
                str(empty_mcp),
                "--strict-mcp-config",
                "--no-chrome",
                "--disable-slash-commands",
                "--disallowedTools",
                "WebSearch,WebFetch,mcp__*",
            ]
            command_path.write_text(
                json.dumps(command, indent=2) + "\n", encoding="utf-8"
            )

            environment = os.environ.copy()
            environment.pop("ANTHROPIC_API_KEY", None)
            environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
            environment["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
            environment["MAX_THINKING_TOKENS"] = str(THINKING_BUDGET_TOKENS)
            environment["DISABLE_AUTOUPDATER"] = "1"

            with (
                stdout_path.open("w", encoding="utf-8") as stdout_handle,
                stderr_path.open("w", encoding="utf-8") as stderr_handle,
            ):
                child = subprocess.Popen(
                    command,
                    cwd=repository_dir,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    returncode = child.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    terminate_process_group(child, grace_seconds=5)
                    raise TimeoutError("Claude Code exceeded the 1200 second timeout")

            if returncode != 0:
                raise RuntimeError(f"Claude Code exited with status {returncode}")
            if not target_path.is_file():
                raise RuntimeError(
                    f"Claude Code removed the target file: {changed_file}"
                )

            completion = target_path.read_text(encoding="utf-8")
            if (
                not completion.strip()
                or completion == masked_content
                or "// <MASK>" in completion
            ):
                raise RuntimeError(
                    "Claude Code produced an empty or unchanged completion"
                )

            _commit_all(repository, "Record Claude Code completion")
            diff = repository.git.diff(base_commit, "HEAD", "--", changed_file)
            if not diff.strip():
                raise RuntimeError("Claude Code produced no nonempty target file diff")
            changed_paths = [
                line
                for line in repository.git.diff(
                    base_commit, "HEAD", "--name-only"
                ).splitlines()
                if line
            ]
            completion_path = output_dir / "completion.txt"
            diff_path = output_dir / "completion.diff"
            completion_path.write_text(completion, encoding="utf-8")
            diff_path.write_text(diff, encoding="utf-8")

            manifest = {
                "task_id": task_id,
                "condition": condition,
                "project_name": project_name,
                "repository_url": repository_url,
                "fixing_commit": fixing_commit,
                "changed_file": changed_file,
                "model": model_name,
                "thinking_budget_tokens": THINKING_BUDGET_TOKENS,
                "removed_instruction_files": removed_instructions,
                "remaining_instruction_files": remaining,
                "project_memory_file_sha256": project_memory_file_sha256,
                "completion_sha256": sha256_file(completion_path),
                "diff_sha256": sha256_file(diff_path),
                "changed_paths": changed_paths,
                "unexpected_changed_paths": [
                    path for path in changed_paths if path != changed_file
                ],
                "returncode": returncode,
            }
            (output_dir / "inference_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return manifest
        finally:
            if child is not None:
                terminate_process_group(child)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if repository_dir.exists():
                shutil.rmtree(repository_dir, ignore_errors=True)


# Defines the single-run CLI contract needed by the external experiment orchestrator.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--condition", choices=("baseline", "project_memory"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--project-memory-file", type=Path)
    return parser.parse_args()


# Executes one requested run and exposes success, failure, or interruption through stable exit codes.
def main() -> int:
    arguments = _parse_args()
    try:
        ClaudeCodeRunner.run_task(
            root=ROOT,
            task_id=arguments.task_id,
            condition=arguments.condition,
            output_dir=arguments.output_dir,
            model_name=arguments.model,
            project_memory_file_path=arguments.project_memory_file,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        (arguments.output_dir / "harness_error.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
