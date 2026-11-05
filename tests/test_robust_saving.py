"""Robust saving: Hub mirror (mocked), resume from the Hub, resumable evaluation, partial aggregation.
No network: FakeApi stands in for huggingface_hub.HfApi."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verispace_engine.hub import HubSync, HubVerifyError, restore_checkpoint

ROOT = Path(__file__).parent.parent


class FakeApi:
    """In-memory Hub repo. `lose` names files the 'Hub' silently drops; `shrink` truncates them."""

    def __init__(self, private=True, lose=(), shrink=(), fail_with=None):
        self.fail_with = fail_with
        self.files: dict[str, bytes] = {}
        self.private, self.lose, self.shrink = private, set(lose), set(shrink)
        self.commits: list[str] = []

    def create_repo(self, repo_id, private=None, exist_ok=False, repo_type=None):
        assert private is True                      # the repo must be requested private

    def repo_info(self, repo_id, repo_type=None):
        return SimpleNamespace(private=self.private)

    def create_commit(self, repo_id, operations, commit_message, repo_type=None):
        self.commits.append(commit_message)
        for op in operations:
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            if op.path_in_repo in self.lose:
                continue
            self.files[op.path_in_repo] = data[:-1] if op.path_in_repo in self.shrink else data

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, repo_type=None):
        if self.fail_with is not None:
            raise self.fail_with
        if path_in_repo is None and not self.files:
            raise _real_404()                        # the real Hub: 404 on an empty repo
        if path_in_repo is not None and not any(p.startswith(path_in_repo + "/") for p in self.files):
            raise _real_404()                        # ... and on a path that does not exist (yet)
        yield SimpleNamespace(path="a_folder")      # folders carry no size
        for p, d in sorted(self.files.items()):
            if path_in_repo is None or p.startswith(path_in_repo + "/"):
                yield SimpleNamespace(path=p, size=len(d))

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[filename])
        return str(out)


def _real_404(name="RemoteEntryNotFoundError"):
    """Build the real huggingface_hub exception (as the Hub raises it on an empty repo)."""
    from huggingface_hub import errors
    cls = getattr(errors, name)
    try:
        return cls("404 Client Error: checkpoint does not exist on main")
    except TypeError:                                # HfHubHTTPError wants a response object
        try:
            import httpx2 as httpx                   # the HTTP module this huggingface_hub version uses
        except ImportError:
            import httpx
        req = httpx.Request("GET", "https://huggingface.co/api/models/me/x/tree/main/checkpoint")
        return cls("404: checkpoint does not exist on main", response=httpx.Response(404, request=req))


def hub(**kw):
    return HubSync("me/helio-test", api=FakeApi(**kw))


# --- Hub upload and verification (mocked) -------------------------------------------------------------
def test_upload_lists_the_repo_and_returns_sizes(tmp_path, capsys):
    f = tmp_path / "x.json"
    f.write_text("hello")
    h = hub()
    assert h.upload({"a/x.json": f, "b.bin": b"12345"}, "msg") == {"a/x.json": 5, "b.bin": 5}
    out = capsys.readouterr().out
    assert "a/x.json" in out and "b.bin" in out and "remote 5" in out
    assert h.list_files() == {"a/x.json": 5, "b.bin": 5}


def test_a_missing_file_after_upload_fails_loudly():
    with pytest.raises(HubVerifyError, match="missing"):
        hub(lose={"state.pt"}).upload({"state.pt": b"abc", "ok": b"1"}, "m")


def test_a_size_mismatch_after_upload_fails_loudly():
    with pytest.raises(HubVerifyError, match="remote 2 B != local 3 B"):
        hub(shrink={"state.pt"}).upload({"state.pt": b"abc"}, "m")


def test_a_public_repo_is_refused():
    with pytest.raises(RuntimeError, match="not private"):
        hub(private=False)


def test_missing_token_fails_at_startup(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        HubSync("me/x")


def test_checkpoint_round_trip_through_the_mock_hub(tmp_path):
    latest = tmp_path / "latest"
    (latest / "adapter").mkdir(parents=True)
    (latest / "state.pt").write_bytes(b"state")
    (latest / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    summ = tmp_path / "train_summary.json"
    summ.write_text("{}")
    h = hub()
    assert h.latest_meta() is None
    h.upload_checkpoint(latest, 22, "hash1", {"train_summary.json": summ, "train_log.jsonl": tmp_path / "absent"})
    assert set(h.list_files()) == {"checkpoint/state.pt", "checkpoint/adapter/adapter_model.safetensors",
                                   "checkpoint/STEP.json", "train_summary.json"}
    assert h.latest_meta() == {"step": 22, "cfg_hash": "hash1"}
    dest = tmp_path / "fresh" / "latest"
    assert h.download_checkpoint(dest) == 22
    assert (dest / "state.pt").read_bytes() == b"state"
    assert (dest / "adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert not (dest / "STEP.json").exists()


def test_restore_checkpoint_from_drive_folder_or_hub(tmp_path):
    drive = tmp_path / "drive_ckpts"
    (drive / "latest" / "adapter").mkdir(parents=True)
    (drive / "latest" / "state.pt").write_bytes(b"s22")
    ck = tmp_path / "local"
    assert "Drive folder" in restore_checkpoint(ck, drive_dir=drive)
    assert (ck / "latest" / "state.pt").read_bytes() == b"s22"
    h = hub()
    h.upload_checkpoint(ck / "latest", 22, "h")
    ck2 = tmp_path / "local2"
    assert "step 22 from Hub" in restore_checkpoint(ck2, drive_dir=tmp_path / "nothing_here", hub=h)
    assert (ck2 / "latest" / "state.pt").read_bytes() == b"s22"
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(tmp_path / "x", drive_dir=tmp_path / "none")


# --- empty repo: the real Hub raises 404 for a path that does not exist yet --------------------------------
def test_empty_repo_listing_and_meta_treat_the_404_as_no_checkpoint():
    h = hub()
    with pytest.raises(type(_real_404())):                       # the fake really raises the Hub's exception
        list(h.api.list_repo_tree("me/x", path_in_repo="checkpoint"))
    assert h.list_files("checkpoint") == {} and h.list_files() == {}
    assert h.latest_meta() is None


@pytest.mark.parametrize("name", ["RemoteEntryNotFoundError", "EntryNotFoundError"])
def test_both_not_found_errors_are_treated_as_missing(name):
    h = hub()
    h.api.fail_with = _real_404(name)
    assert h.list_files("checkpoint") == {} and h.latest_meta() is None


@pytest.mark.parametrize("name", ["RepositoryNotFoundError", "HfHubHTTPError", "RevisionNotFoundError"])
def test_other_hub_errors_still_raise(name):
    h = hub()
    h.api.fail_with = _real_404(name)
    with pytest.raises(type(h.api.fail_with)):
        h.list_files("checkpoint")
    with pytest.raises(type(h.api.fail_with)):
        h.latest_meta()
    for exc in (PermissionError("401"), ConnectionError("down"), TimeoutError()):
        h.api.fail_with = exc
        with pytest.raises(type(exc)):
            h.list_files()


def test_a_404_on_the_step_file_download_means_no_checkpoint_other_errors_raise():
    h = hub()
    h.upload({"checkpoint/STEP.json": b"{}"}, "m", verbose=False)

    def boom(exc):
        def f(*a, **k):
            raise exc
        return f
    h.api.hf_hub_download = boom(_real_404("EntryNotFoundError"))
    assert h.latest_meta() is None
    h.api.hf_hub_download = boom(ConnectionError("down"))
    with pytest.raises(ConnectionError):
        h.latest_meta()


def test_downloading_from_an_empty_repo_is_a_clear_error_and_restore_reports_it(tmp_path):
    h = hub()
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        h.download_checkpoint(tmp_path / "latest")
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(tmp_path / "ck", drive_dir=tmp_path / "none", hub=h)
    assert not (tmp_path / "ck" / "latest").exists()


def test_first_upload_to_an_empty_repo_verifies():
    h = hub()
    assert h.upload({"checkpoint/STEP.json": b"{}", "x": b"1"}, "first", verbose=False) == {
        "checkpoint/STEP.json": 2, "x": 1}


def test_empty_repo_then_first_upload_then_resume_through_training(tmp_path):
    pytest.importorskip("torch")
    from test_sft import synthetic_rows, tiny_build, tiny_cfg
    from verispace_engine.sft import train
    rows, h = synthetic_rows(24), hub()
    assert h.latest_meta() is None                               # empty repo on start
    first = train(tiny_cfg(tmp_path, "a"), rows, tiny_build, stop_after_steps=3, hub=h)   # from step 0, first upload
    assert first["resumed_from"] == 0 and h.latest_meta()["step"] == 3
    second = train(tiny_cfg(tmp_path, "fresh"), rows, tiny_build, hub=h)                  # fresh runtime resumes
    assert second["resumed_from"] == 3 and second["finished"]


def test_eval_on_an_empty_repo_uploads_and_verifies(tmp_path):
    from helioguard.sft_eval import evaluate
    h = hub()
    evaluate(GarbageBackend(), n=2, out_dir=tmp_path, tag="t", greedy_only=True, hub=h, upload_every=1,
             baseline_dir=tmp_path / "nobase")
    assert h.list_files("eval") and h.latest_meta() is None


# --- training: mirror to the Hub, resume on a fresh runtime from the Hub only --------------------------
def test_training_mirrors_every_checkpoint_and_resumes_from_the_hub_alone(tmp_path, capsys):
    pytest.importorskip("torch")
    from test_sft import synthetic_rows, tiny_build, tiny_cfg
    from verispace_engine.sft import train
    rows, h = synthetic_rows(24), hub()
    first = train(tiny_cfg(tmp_path, "a"), rows, tiny_build, stop_after_steps=3, hub=h)
    assert first["steps_done"] == 3
    out = capsys.readouterr().out
    assert '"event": "checkpoint", "step": 3, "verified_step_in_state_pt": 3' in out    # step read back and printed
    assert h.latest_meta()["step"] == 3 and {"train_summary.json", "train_log.jsonl"} <= set(h.list_files())
    # a fresh runtime: new local dirs (the "Drive" is gone), only the Hub survives
    second = train(tiny_cfg(tmp_path, "fresh"), rows, tiny_build, hub=h)
    assert second["resumed_from"] == 3 and second["finished"] and second["steps_done"] == 6
    assert "downloaded step 3 from Hub" in capsys.readouterr().out
    files = h.list_files()
    assert h.latest_meta()["step"] == 6
    assert "adapter/adapter_model.safetensors" in files and "checkpoint/state.pt" in files
    assert json.loads(h.api.files["train_summary.json"])["finished"] is True
    steps = [json.loads(line)["step"] for line in h.api.files["train_log.jsonl"].decode().splitlines()]
    assert steps == [1, 2, 3, 4, 5, 6]               # the earlier steps came back with the checkpoint
    # and it equals a run that never stopped
    straight = train(tiny_cfg(tmp_path, "straight"), rows, tiny_build)
    a = [json.loads(line)["loss"] for line in (tmp_path / "straight" / "train_log.jsonl").read_text().splitlines()]
    b = [json.loads(line)["loss"] for line in (tmp_path / "fresh" / "train_log.jsonl").read_text().splitlines()]
    assert a == pytest.approx(b, rel=1e-5, abs=1e-6) and straight["finished"]


def test_local_checkpoint_wins_when_it_is_as_new_as_the_hub(tmp_path, capsys):
    pytest.importorskip("torch")
    from test_sft import synthetic_rows, tiny_build, tiny_cfg
    from verispace_engine.sft import train
    rows, h = synthetic_rows(24), hub()
    train(tiny_cfg(tmp_path, "a"), rows, tiny_build, stop_after_steps=3, hub=h)
    capsys.readouterr()
    second = train(tiny_cfg(tmp_path, "a"), rows, tiny_build, hub=h)
    assert second["resumed_from"] == 3
    assert "at least as new" in capsys.readouterr().out


def test_a_failed_hub_verification_stops_training_loudly(tmp_path):
    pytest.importorskip("torch")
    from test_sft import synthetic_rows, tiny_build, tiny_cfg
    from verispace_engine.sft import train
    with pytest.raises(HubVerifyError):
        train(tiny_cfg(tmp_path, "a"), synthetic_rows(24), tiny_build, stop_after_steps=3,
              hub=hub(lose={"checkpoint/state.pt"}))


def test_hub_checkpoint_from_another_config_is_refused(tmp_path):
    pytest.importorskip("torch")
    from test_sft import synthetic_rows, tiny_build, tiny_cfg
    from verispace_engine.sft import train
    rows, h = synthetic_rows(24), hub()
    train(tiny_cfg(tmp_path, "a"), rows, tiny_build, stop_after_steps=3, hub=h)
    with pytest.raises(ValueError, match="different training config"):
        train(tiny_cfg(tmp_path, "b", lr=1e-3), rows, tiny_build, hub=h)


def test_cfg_hash_is_unchanged_by_this_work():
    """The step-22 checkpoint from the lost run must still load: the hash must not depend on any new field."""
    from verispace_engine.sft import SFTConfig, _cfg_hash
    from dataclasses import fields
    assert [f.name for f in fields(SFTConfig)] == [
        "model", "data", "out_dir", "checkpoint_dir", "max_seq_len", "lora_r", "lora_alpha", "lora_dropout",
        "target_modules", "lr", "weight_decay", "epochs", "grad_accum", "warmup_steps", "max_steps",
        "max_grad_norm", "seed", "checkpoint_minutes", "log_every", "dtype", "trainer",
        "gradient_checkpointing", "extra"]
    assert _cfg_hash(SFTConfig()) == _cfg_hash(SFTConfig(out_dir="elsewhere", max_steps=5))


# --- evaluation: append-per-episode, skip finished, upload every N, aggregate from files ----------------
class Crash(Exception):
    pass


class GarbageBackend:
    """Every reply fails the strict parser, so each episode is one call and a failure (no model needed)."""
    name = "garbage"

    def __init__(self, crash_on_call=None):
        self.calls, self.crash_on_call = 0, crash_on_call

    def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None):
        self.calls += 1
        if self.crash_on_call and self.calls == self.crash_on_call:
            raise Crash
        return "not json"


def _lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def test_eval_appends_each_episode_immediately_and_skips_finished_ones_on_restart(tmp_path):
    from helioguard.sft_eval import evaluate, load_episodes
    b1 = GarbageBackend(crash_on_call=4)
    with pytest.raises(Crash):
        evaluate(b1, n=5, out_dir=tmp_path, tag="t", greedy_only=True, baseline_dir=tmp_path / "nobase")
    greedy = tmp_path / "t_greedy.jsonl"
    assert len(load_episodes(greedy)) == 3                       # saved before the crash, no final summary needed
    assert not (tmp_path / "t_eval_summary.json").exists()
    b2 = GarbageBackend()
    s = evaluate(b2, n=5, out_dir=tmp_path, tag="t", greedy_only=True, baseline_dir=tmp_path / "nobase")
    assert b2.calls == 2                                         # only the 2 unfinished episodes ran
    eps = load_episodes(greedy)
    assert len(eps) == 5 and len({e.task_id for e in eps}) == 5
    assert s["complete"] is True and s["greedy"]["episodes"] == 5
    b3 = GarbageBackend()
    evaluate(b3, n=5, out_dir=tmp_path, tag="t", greedy_only=True, baseline_dir=tmp_path / "nobase")
    assert b3.calls == 0 and len(load_episodes(greedy)) == 5     # nothing re-run, nothing duplicated


def test_a_half_finished_episode_is_rerun_and_leaves_no_stray_records(tmp_path):
    from helioguard.sft_eval import evaluate, load_episodes
    evaluate(GarbageBackend(), n=2, out_dir=tmp_path, tag="t", greedy_only=True, baseline_dir=tmp_path / "nobase")
    g = tmp_path / "t_greedy.jsonl"
    last = g.read_text().splitlines()[-1]                        # drop the last episode record, add a torn line
    g.write_text("\n".join(g.read_text().splitlines()[:-1]) + "\n" + last[:20])
    b = GarbageBackend()
    evaluate(b, n=2, out_dir=tmp_path, tag="t", greedy_only=True, baseline_dir=tmp_path / "nobase")
    assert b.calls == 1 and len(load_episodes(g)) == 2
    assert all(r.get("event") in ("turn", "episode") for r in _lines(g))


def test_greedy_then_temperature_seeds_go_to_separate_files(tmp_path):
    from helioguard.sft_eval import evaluate, load_episodes
    b = GarbageBackend()
    s = evaluate(b, n=2, out_dir=tmp_path, tag="t", baseline_dir=tmp_path / "nobase")
    names = {p.name for p in tmp_path.glob("t_*.jsonl")}
    assert names == {"t_greedy.jsonl", "t_t0.7_seed0.jsonl", "t_t0.7_seed1.jsonl", "t_t0.7_seed2.jsonl"}
    assert b.calls == 2 * 4
    assert all(len(load_episodes(tmp_path / n)) == 2 for n in names)
    assert s["temperature_0.7_3_seeds"]["episodes"] == 6 and s["greedy"]["episodes"] == 2


def test_eval_uploads_every_n_episodes_and_verifies(tmp_path):
    from helioguard.sft_eval import evaluate
    h = hub()
    evaluate(GarbageBackend(), n=5, out_dir=tmp_path, tag="t", greedy_only=True, hub=h, upload_every=2,
             baseline_dir=tmp_path / "nobase")
    # 5 episodes: uploads after 2 and 4, a closing upload for the 5th, then the summary
    assert [c.split(": ", 1)[1] for c in h.api.commits] == ["greedy 2/5", "greedy 4/5", "greedy finished", "summary"]
    assert set(h.list_files("eval")) == {"eval/t_greedy.jsonl", "eval/t_eval_summary.json"}
    assert h.api.files["eval/t_greedy.jsonl"] == (tmp_path / "t_greedy.jsonl").read_bytes()


def test_eval_upload_failure_is_loud(tmp_path):
    from helioguard.sft_eval import evaluate
    with pytest.raises(HubVerifyError):
        evaluate(GarbageBackend(), n=3, out_dir=tmp_path, tag="t", greedy_only=True, upload_every=1,
                 hub=hub(lose={"eval/t_greedy.jsonl"}), baseline_dir=tmp_path / "nobase")


def test_partial_aggregation_from_saved_files_issues_no_gates(tmp_path):
    from helioguard.harness_env import load
    from helioguard.sft_eval import aggregate, evaluate
    with pytest.raises(Crash):
        evaluate(GarbageBackend(crash_on_call=3), n=4, out_dir=tmp_path, tag="t", baseline_dir=tmp_path / "nobase")
    env, ids = load("eval", 4)
    tasks = [env._by_id[i] for i in ids]
    part = aggregate(tmp_path, "t", tasks, baseline_dir=tmp_path / "nobase")      # no model, files only
    assert part["complete"] is False and part["gates"] is None and "partial" in part["note"]
    assert part["progress"]["greedy"] == {"episodes": 2, "expected": 4}
    assert part["progress"]["t0.7_seed0"]["episodes"] == 0 and "temperature_0.7_3_seeds" not in part
    assert part["greedy"]["episodes"] == 2
    evaluate(GarbageBackend(), n=4, out_dir=tmp_path, tag="t", baseline_dir=tmp_path / "nobase")
    full = aggregate(tmp_path, "t", tasks, baseline_dir=tmp_path / "nobase")
    assert full["complete"] is True and full["gates"]["gate_A_format"]["pass"] is False   # garbage replies fail gate A
    assert full["temperature_0.7_3_seeds"]["episodes"] == 12


def test_aggregate_with_no_files_does_not_crash(tmp_path):
    from helioguard.harness_env import load
    from helioguard.sft_eval import aggregate
    env, ids = load("eval", 3)
    s = aggregate(tmp_path, "t", [env._by_id[i] for i in ids], baseline_dir=tmp_path / "nobase")
    assert s["complete"] is False and s["gates"] is None and "greedy" not in s


# --- notebook -----------------------------------------------------------------------------------------
def test_notebook_has_hub_cells_and_a_restore_cell_and_no_secrets():
    raw = (ROOT / "notebooks" / "phase6_sft_colab.ipynb").read_text(encoding="utf-8")
    nb = json.loads(raw)
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any('userdata.get("HF_TOKEN")' in c for c in code)
    train_cell = next(c for c in code if "train_sft.py" in c)
    assert "--hub-repo" in train_cell
    restore = next(c for c in code if "DRIVE_CKPT_FOLDER" in c)
    assert "DRIVE_CKPT_FOLDER" in restore and "HUB_REPO" in restore
    assert code.index(restore) < code.index(train_cell)
    assert "--hub-repo" in next(c for c in code if "sft_eval" in c)
    assert "hf_" not in raw.replace("hf_hub", "").replace("hf_token", "")     # no token literal
