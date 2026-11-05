"""Private Hugging Face Hub mirror for checkpoints and eval files (domain-neutral, optional dependency).

Why: a Colab Drive mount can silently stop syncing, losing hours of work. Every upload here is one atomic
commit followed by a listing of the repo; a missing file or a size mismatch raises HubVerifyError.

Layout in the repo:  checkpoint/{adapter/..., state.pt, STEP.json}   latest training checkpoint
                     train_summary.json, train_log.jsonl, adapter/... (final)   run outputs
                     eval/...                                         evaluation episode files
huggingface_hub is imported inside the functions that need it, so the engine imports without it.
The token is read from the HF_TOKEN environment variable (on Colab, copy it from the secret first).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

CKPT_PREFIX = "checkpoint"
__all__ = ["CKPT_PREFIX", "HubSync", "HubVerifyError", "restore_checkpoint"]


def _not_found() -> tuple[type[BaseException], ...]:
    """The two 'path does not exist in the repo' errors, and nothing else (auth, network, missing repo still raise)."""
    from huggingface_hub import errors
    return tuple(c for c in (getattr(errors, "RemoteEntryNotFoundError", None),
                             getattr(errors, "EntryNotFoundError", None)) if c is not None)


class HubVerifyError(RuntimeError):
    """The Hub repo does not contain what was just uploaded."""


class HubSync:
    def __init__(self, repo_id: str, token: str | None = None, api: Any = None, create: bool = True) -> None:
        self.repo_id = repo_id
        token = token or os.environ.get("HF_TOKEN")
        if api is None:
            if not token:
                raise RuntimeError("HF_TOKEN is not set: add it as a Colab secret and export it, or pass token=")
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api = api
        if create:
            api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
            if getattr(api.repo_info(repo_id, repo_type="model"), "private", None) is not True:
                raise RuntimeError(f"Hub repo {repo_id} is not private; refusing to upload")

    # --- upload + verify -----------------------------------------------------------------------
    def upload(self, files: Mapping[str, str | Path | bytes], message: str, verbose: bool = True) -> dict[str, int]:
        """Upload {path_in_repo: local path or bytes} in one commit, then verify. Returns {path: size}."""
        from huggingface_hub import CommitOperationAdd
        ops, expected = [], {}
        for repo_path, src in files.items():
            data = src if isinstance(src, bytes) else Path(src)
            size = len(data) if isinstance(data, bytes) else data.stat().st_size
            ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=data if isinstance(data, bytes) else str(data)))
            expected[repo_path] = size
        self.api.create_commit(self.repo_id, ops, commit_message=message, repo_type="model")
        listing = self.list_files()
        if verbose:
            print(f"[hub] {self.repo_id}: {message}", flush=True)
            for p in sorted(expected):
                print(f"[hub]   {p}  local {expected[p]:,} B  remote {listing.get(p, 'MISSING')}", flush=True)
        problems = [f"{p}: missing" if p not in listing else f"{p}: remote {listing[p]} B != local {s} B"
                    for p, s in expected.items() if listing.get(p) != s]
        if problems:
            raise HubVerifyError(f"Hub upload to {self.repo_id} failed verification: " + "; ".join(problems))
        return expected

    def list_files(self, prefix: str | None = None) -> dict[str, int]:
        """{path: size} under prefix. A prefix (or a whole repo) with nothing in it yields {}: on an empty repo
        the Hub answers 404 'does not exist on main', which means 'no files yet', not an error."""
        out = {}
        try:
            for item in self.api.list_repo_tree(self.repo_id, path_in_repo=prefix, recursive=True, repo_type="model"):
                size = getattr(item, "size", None)
                if size is not None:                          # folders carry no size
                    out[item.path] = int(size)
        except _not_found():
            return {}
        return out

    def upload_checkpoint(self, latest_dir: str | Path, step: int, cfg_hash: str,
                          extra: Mapping[str, str | Path] | None = None) -> dict[str, int]:
        """Mirror a checkpoint directory (adapter/, state.pt) plus extra run files (summary, log)."""
        latest = Path(latest_dir)
        files: dict[str, str | Path | bytes] = {}
        for f in sorted(latest.rglob("*")):
            if f.is_file():
                files[f"{CKPT_PREFIX}/{f.relative_to(latest).as_posix()}"] = f
        files[f"{CKPT_PREFIX}/STEP.json"] = json.dumps({"step": step, "cfg_hash": cfg_hash}).encode()
        files.update({k: Path(v) for k, v in (extra or {}).items() if Path(v).exists()})
        return self.upload(files, f"checkpoint step {step}")

    # --- resume --------------------------------------------------------------------------------
    def latest_meta(self) -> dict[str, Any] | None:
        """{"step", "cfg_hash"} of the checkpoint in the repo, or None if there is none."""
        if f"{CKPT_PREFIX}/STEP.json" not in self.list_files(CKPT_PREFIX):
            return None
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = self.api.hf_hub_download(self.repo_id, f"{CKPT_PREFIX}/STEP.json", repo_type="model", local_dir=tmp)
            except _not_found():
                return None
            return json.loads(Path(p).read_text(encoding="utf-8"))

    def download_checkpoint(self, dest_latest: str | Path) -> int:
        """Download the Hub checkpoint into dest_latest (replacing it) and return its step."""
        dest = Path(dest_latest)
        listing = self.list_files(CKPT_PREFIX)
        if f"{CKPT_PREFIX}/STEP.json" not in listing:
            raise FileNotFoundError(f"Hub repo {self.repo_id} has no checkpoint to download")
        stage = Path(tempfile.mkdtemp(prefix="hub_ckpt_", dir=dest.parent if dest.parent.exists() else None))
        try:
            for p, size in listing.items():
                got = Path(self.api.hf_hub_download(self.repo_id, p, repo_type="model", local_dir=stage))
                if got.stat().st_size != size:
                    raise HubVerifyError(f"downloaded {p} has {got.stat().st_size} B, repo says {size} B")
            src = stage / CKPT_PREFIX
            step = int(json.loads((src / "STEP.json").read_text(encoding="utf-8"))["step"])
            (src / "STEP.json").unlink()
            shutil.rmtree(dest, ignore_errors=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            return step
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def restore_checkpoint(checkpoint_dir: str | Path, drive_dir: str | Path | None = None,
                       hub: HubSync | None = None) -> str:
    """Put a checkpoint where training resumes from (checkpoint_dir/latest): copy it from a Drive folder
    (a folder holding state.pt and adapter/, or one holding latest/) or download it from the Hub repo.
    Returns a one-line description naming the source."""
    dest = Path(checkpoint_dir) / "latest"
    if drive_dir:
        d = Path(drive_dir)
        src = next((c for c in (d / "latest", d) if (c / "state.pt").exists() and (c / "adapter").is_dir()), None)
        if src is not None:
            if src.resolve() != dest.resolve():
                shutil.rmtree(dest, ignore_errors=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
            return f"restored from Drive folder {src}"
    if hub is not None:
        step = hub.download_checkpoint(dest)
        return f"restored step {step} from Hub repo {hub.repo_id}"
    raise FileNotFoundError("no checkpoint found in the Drive folder and no Hub repo given")
