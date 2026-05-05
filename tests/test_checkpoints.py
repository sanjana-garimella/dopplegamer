from __future__ import annotations

from pathlib import Path

from agents import RLAgent, SFTAgent
from agents.checkpoints import checkpoint_status, resolve_checkpoint
from dashboard.pages.live_game import _agent_runtime_status


def test_resolve_checkpoint_prefers_real_ppo(tmp_path, monkeypatch):
    import agents.checkpoints as ckpt

    monkeypatch.setattr(ckpt, "CHECKPOINTS_DIR", tmp_path)
    (tmp_path / "ppo_real.zip").write_text("stub")
    assert resolve_checkpoint("rl") == tmp_path / "ppo_real"


def test_checkpoint_status_reports_absence(tmp_path, monkeypatch):
    import agents.checkpoints as ckpt

    monkeypatch.setattr(ckpt, "CHECKPOINTS_DIR", tmp_path)
    found, path = checkpoint_status("sft")
    assert found is False
    assert path is None


def test_runtime_status_includes_checkpoint_path():
    class Dummy:
        _model = object()
        checkpoint_path = Path("checkpoints/ppo_real")

    label, detail = _agent_runtime_status("rl", Dummy())
    assert label == "Checkpoint loaded"
    assert "checkpoints/ppo_real" in detail


def test_agent_records_resolved_checkpoint_path(tmp_path, monkeypatch):
    import agents.checkpoints as ckpt

    monkeypatch.setattr(ckpt, "CHECKPOINTS_DIR", tmp_path)
    (tmp_path / "ppo_real.zip").write_text("stub")
    (tmp_path / "sft_real").mkdir()

    rl_agent = RLAgent()
    sft_agent = SFTAgent()

    assert rl_agent.checkpoint_path == tmp_path / "ppo_real"
    assert sft_agent.checkpoint_path == tmp_path / "sft_real"
