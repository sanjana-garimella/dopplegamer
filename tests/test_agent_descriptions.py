from __future__ import annotations

from dashboard.pages.live_game import _agent_description


def test_profile_counter_description_mentions_history():
    description = _agent_description("profile_counter")
    assert "saved history" in description
    assert "personalized" in description


def test_alias_agent_descriptions_are_present():
    assert "alias" in _agent_description("ppo").lower()
    assert "alias" in _agent_description("bcrl").lower()
