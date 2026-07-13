"""
Integration tests for the agent glue (`conversation.py`).

Coverage:
    - passthrough (no fuzzy match -> forward original text unchanged)
    - fuzzy hit (no slots -> canonical sentence forwarded)
    - slot extraction (resolver-backed slot resolution)
    - sibling fallback (best-scored expansion not extractable)
    - no-match below threshold
    - registry-change rebuild (cleared cache, fresh slot_values)
    - per-language pool isolation
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# re-confirm sys.path extension in case of nondeterministic load order
PKG_DIR = Path(__file__).resolve().parent.parent / "custom_components" / "closest_intent"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


import conversation as agent_module  # type: ignore  # noqa: E402
from const import (  # type: ignore  # noqa: E402
    DEFAULT_STARTUP_SELF_CHECK,
    DOMAIN,
    KEY_CONVERSATION_EXPANSION_RULES,
    KEY_CONVERSATION_INTENTS,
    KEY_CONVERSATION_LISTS,
)
from conversation import ClosestIntentAgent  # type: ignore  # noqa: E402


class FakeBus:
    def __init__(self) -> None:
        self.listeners: dict[str, list] = {}

    def async_listen(self, event_name: str, cb):
        import contextlib

        self.listeners.setdefault(event_name, []).append(cb)

        def _unsub() -> None:
            with contextlib.suppress(ValueError):
                self.listeners[event_name].remove(cb)

        return _unsub

    def fire(self, event_name: str, data: dict | None = None) -> None:
        for cb in list(self.listeners.get(event_name, [])):
            cb(SimpleNamespace(data=data or {}, event_type=event_name))


class FakeStates:
    def __init__(self) -> None:
        self._states: dict[str, Any] = {}

    def set(self, entity_id: str, friendly_name: str) -> None:
        self._states[entity_id] = SimpleNamespace(
            attributes={"friendly_name": friendly_name},
            name=friendly_name,
        )

    def get(self, entity_id: str):
        return self._states.get(entity_id)


class FakeServices:
    def __init__(self) -> None:
        self._svcs: dict[tuple[str, str], Any] = {}

    def has_service(self, domain: str, name: str) -> bool:
        return (domain, name) in self._svcs

    def async_register(self, domain: str, name: str, fn) -> None:
        self._svcs[(domain, name)] = fn


def _make_hass(tmp_path: Path, language: str = "de") -> SimpleNamespace:
    """
    Build a minimal fake hass.

    `async_add_executor_job(fn, *args)` returns an already-fulfilled Future rather than threading,
    so tests stay deterministic and hot paths run on the event loop.
    """

    async def _run_in_executor(fn, *args):
        return fn(*args)

    from conftest import _async_create_background_task  # type: ignore

    hass = SimpleNamespace(
        data={},
        bus=FakeBus(),
        states=FakeStates(),
        services=FakeServices(),
        _scheduled_actions=[],
        async_add_executor_job=_run_in_executor,
        async_create_background_task=_async_create_background_task,
    )
    hass.config = SimpleNamespace(
        language=language,
        path=lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    return hass


def _make_agent(
    hass: SimpleNamespace,
    *,
    threshold: int = 70,
    slot_threshold: int | None = None,
    expansion_cap: int = 16,
    denylist=None,
    include_builtins: bool = False,
    builtin_allowlist=None,
    slot_extraction: bool = True,
    fallback_agent_id: str = "conversation.home_assistant",
    startup_self_check: bool = DEFAULT_STARTUP_SELF_CHECK,
) -> ClosestIntentAgent:
    return ClosestIntentAgent(
        hass,
        threshold=threshold,
        slot_threshold=slot_threshold,
        expansion_cap=expansion_cap,
        denylist=denylist,
        include_builtins=include_builtins,
        builtin_allowlist=builtin_allowlist,
        slot_extraction=slot_extraction,
        fallback_agent_id=fallback_agent_id,
        startup_self_check=startup_self_check,
        entry_id="TESTENTRY",
    )


def _conversation_input(
    text: str,
    language: str = "de",
    *,
    device_id: str | None = None,
    satellite_id: str | None = None,
    extra_system_prompt: str | None = None,
):
    """Build a stubbed ConversationInput."""
    from homeassistant.components.conversation import ConversationInput  # type: ignore

    return ConversationInput(
        text=text,
        language=language,
        device_id=device_id,
        satellite_id=satellite_id,
        extra_system_prompt=extra_system_prompt,
    )


@pytest.fixture
def hass(tmp_path: Path) -> SimpleNamespace:
    return _make_hass(tmp_path)


@pytest.fixture(autouse=True)
def _capture_async_converse(monkeypatch):
    """
    Replace the stub's async_converse with a capturing version.

    Tests assert on ``last_call`` to verify what the agent forwarded.
    """
    captured: dict = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        from homeassistant.components.conversation import ConversationResult  # type: ignore

        return ConversationResult(response={"forwarded": kwargs["text"]})

    monkeypatch.setattr("homeassistant.components.conversation.async_converse", _fake)
    yield captured


@pytest.mark.asyncio
async def test_passthrough_when_no_match(hass, _capture_async_converse):
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }

    agent = _make_agent(hass, threshold=80)
    await agent.async_added_to_hass()

    user_input = _conversation_input("erzähl mir einen witz")
    await agent.async_process(user_input)

    # Below-threshold: agent forwards user's original text unchanged.
    assert _capture_async_converse["text"] == "erzähl mir einen witz"
    assert _capture_async_converse["agent_id"] == "conversation.home_assistant"


@pytest.mark.asyncio
async def test_fuzzy_hit_no_slots(hass, _capture_async_converse):
    """Fuzzy match -> canonical sentence (case-preserving) forwarded."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("pumpr an"))
    assert _capture_async_converse["text"] == "Pumpe an"


@pytest.mark.asyncio
async def test_slot_extraction_and_resolution(hass, _capture_async_converse):
    """Slot pattern matched + captured text fuzz-resolved against slot list values."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS] = {
        "area": {"values": ["Wohnzimmer", "Büro"]},
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("test zwei im wohnzma"))
    assert _capture_async_converse["text"] == "Test zwei im Wohnzimmer"


@pytest.mark.asyncio
async def test_sibling_fallback(hass, _capture_async_converse):
    """If best-scored expansion can't extract, fall through to a sibling."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Einkauf_Add": [
            "(setze|tu|pack) {item} auf die einkaufsliste",
        ],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("setze brot auf die einkaufsliste"))
    forwarded = _capture_async_converse["text"]
    assert "brot" in forwarded
    assert "auf die einkaufsliste" in forwarded


@pytest.mark.asyncio
async def test_device_and_satellite_ids_propagate_to_hassil(hass, _capture_async_converse):
    """
    The hassil forward must carry device_id/satellite_id so the default
    agent can derive ``preferred_area_id`` from the invoking satellite.
    """
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(
        _conversation_input(
            "pumpr an",
            device_id="dev-abc",
            satellite_id="sat-xyz",
            extra_system_prompt="be terse",
        )
    )

    assert _capture_async_converse["agent_id"] == "conversation.home_assistant"
    assert _capture_async_converse["device_id"] == "dev-abc"
    assert _capture_async_converse["satellite_id"] == "sat-xyz"
    assert _capture_async_converse["extra_system_prompt"] == "be terse"


@pytest.mark.asyncio
async def test_device_and_satellite_ids_propagate_to_fallback(
    hass, _capture_async_converse, monkeypatch
):
    """
    When the hassil forward returns an error result, the configured
    fallback agent must also receive device_id/satellite_id.
    """
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70, fallback_agent_id="conversation.my_llm")
    await agent.async_added_to_hass()

    calls: list[dict] = []

    async def _fake(**kwargs):
        from homeassistant.components.conversation import (  # type: ignore
            ConversationResult,
        )

        calls.append(kwargs)
        if kwargs["agent_id"] == "conversation.home_assistant":
            response = SimpleNamespace(error_code="no_intent_match")
            return ConversationResult(response=response)
        return ConversationResult(response={"text": kwargs["text"]})

    monkeypatch.setattr("homeassistant.components.conversation.async_converse", _fake)

    await agent.async_process(
        _conversation_input(
            "pumpr an",
            device_id="dev-abc",
            satellite_id="sat-xyz",
            extra_system_prompt="be terse",
        )
    )

    assert [c["agent_id"] for c in calls] == [
        "conversation.home_assistant",
        "conversation.my_llm",
    ]
    for call in calls:
        assert call["device_id"] == "dev-abc"
        assert call["satellite_id"] == "sat-xyz"
        assert call["extra_system_prompt"] == "be terse"


@pytest.mark.asyncio
async def test_registry_change_triggers_rebuild(hass, _capture_async_converse):
    """
    Firing area_registry_updated invalidates the cached pool and a fresh
    one is built from the (now-updated) slot list stash.
    """
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS] = {
        "area": {"values": ["Wohnzimmer"]},
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    pool = agent._pools["de"]
    assert pool[0].slot_values.get("area") == ["Wohnzimmer"]

    # Simulate a registry change.
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS]["area"]["values"] = [
        "Wohnzimmer",
        "Küche",
    ]
    hass.bus.fire("area_registry_updated", {})
    assert hass._scheduled_actions, "expected debounced rebuild to be scheduled"
    _, scheduled_action = hass._scheduled_actions.pop()
    await scheduled_action(None)

    pool = agent._pools["de"]
    assert pool[0].slot_values.get("area") == ["Küche", "Wohnzimmer"]


@pytest.mark.asyncio
async def test_per_language_pools(hass, _capture_async_converse):
    """Different ``user_input.language`` values must yield independent pools."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    assert "de" in agent._pools

    await agent.async_process(_conversation_input("pumpe an", language="en"))
    assert "en" in agent._pools


@pytest.mark.asyncio
async def test_slot_extraction_disabled_falls_back_to_passthrough(hass, _capture_async_converse):
    """slot_extraction=false: matches with slots are skipped, original text forwarded."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    agent = _make_agent(hass, threshold=70, slot_extraction=False)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("test zwei im büro"))
    assert _capture_async_converse["text"] == "test zwei im büro"


@pytest.mark.asyncio
async def test_custom_sentences_loaded(hass, _capture_async_converse, tmp_path):
    """Files under ``custom_sentences/<lang>/*.yaml`` are picked up."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("PyYAML not available")

    cs_dir = tmp_path / "custom_sentences" / "de"
    cs_dir.mkdir(parents=True)
    (cs_dir / "einkauf.yaml").write_text(
        "language: de\n"
        "intents:\n"
        "  Einkauf_Add:\n"
        "    data:\n"
        "      - sentences:\n"
        "          - 'schreib {item} auf die einkaufsliste'\n"
        "lists:\n"
        "  item:\n"
        "    wildcard: true\n",
        encoding="utf-8",
    )

    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("schreib salami auf die einkaufsliste"))
    forwarded = _capture_async_converse["text"]
    assert "salami" in forwarded
    assert "einkaufsliste" in forwarded


@pytest.mark.asyncio
async def test_slot_threshold_override_resolves_borderline_capture(hass, _capture_async_converse):
    """Test that lowering the slot threshold improves slot matching."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS] = {
        "area": {"values": ["Wohnzimmer", "Büro"]},
    }

    # "wozim" vs "Wohnzimmer" scores well under 70
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("test zwei im wozim"))
    assert _capture_async_converse["text"] == "Test zwei im wozim"

    agent.apply_options(
        threshold=70,
        slot_threshold=60,
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        builtin_allowlist=None,
        slot_extraction=True,
        fallback_agent_id="conversation.home_assistant",
    )

    await agent.async_process(_conversation_input("test zwei im wozim"))
    assert _capture_async_converse["text"] == "Test zwei im Wohnzimmer"


@pytest.mark.asyncio
async def test_builtin_allowlist_picks_specific_intents(hass, _capture_async_converse, monkeypatch):
    """
    With include_builtins=False but a non-empty builtin_allowlist, only the
    named builtin intents enter the candidate pool.
    """
    user_intents = {"UserPump": ["pumpe an"]}
    builtin_intents = {
        "HassTurnOn": ["turn on {name}"],
        "HassGetWeather": ["how is the weather"],
        "HassSomethingElse": ["activate scene {name}"],
    }

    async def _fake_collect(self, language):
        return ({}, {}, dict(user_intents), dict(builtin_intents))

    monkeypatch.setattr(ClosestIntentAgent, "_async_collect_ha_intents_data", _fake_collect)

    agent = _make_agent(
        hass,
        threshold=70,
        include_builtins=False,
        builtin_allowlist=["HassTurnOn", "HassGetWeather"],
    )
    await agent.async_added_to_hass()

    _, _, builtin_candidates = agent._pools["de"]
    seen_intents = {c.intent for c in builtin_candidates}
    assert seen_intents == {"HassTurnOn", "HassGetWeather"}


@pytest.mark.asyncio
async def test_apply_options_clears_pools(hass, _capture_async_converse):
    """Live option changes must invalidate cached pools."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    assert agent._pools, "pre-warm should populate the cache"
    agent.apply_options(
        threshold=80,
        slot_threshold=None,
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        builtin_allowlist=None,
        slot_extraction=True,
        fallback_agent_id="conversation.home_assistant",
    )
    assert agent._pools == {}


@pytest.fixture
def _issue_registry_state():
    """Reset and expose the conftest issue-registry stub state."""
    from homeassistant.helpers import issue_registry as ir  # type: ignore

    ir._state.clear()
    yield ir._state
    ir._state.clear()


def _find_self_check_issue(state, lang: str = "de", entry_id: str = "TESTENTRY"):
    """Locate the current self-check issue for ``lang``, regardless of count suffix."""
    prefix = f"self_check_{entry_id}_{lang}_"
    for (domain, issue_id), payload in state.items():
        if domain == DOMAIN and issue_id.startswith(prefix):
            return issue_id, payload
    return None


@pytest.mark.asyncio
async def test_self_check_clean_creates_no_issue(
    hass, _capture_async_converse, _issue_registry_state
):
    """Non-clashing intents must not raise a repair issue."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
        "PumpeAus": ["Pumpe aus"],
    }
    agent = _make_agent(hass, threshold=70, startup_self_check=True)
    await agent.async_added_to_hass()

    assert _issue_registry_state == {}


@pytest.mark.asyncio
async def test_self_check_detects_clash(hass, _capture_async_converse, _issue_registry_state):
    """A pattern that consistently routes to a wrong intent must surface as an issue."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        # Two intents with the same canonical sentence guarantee a clash.
        "IntentA": ["Foo bar"],
        "IntentB": ["Foo bar"],
    }
    agent = _make_agent(hass, threshold=70, startup_self_check=True)
    await agent.async_added_to_hass()

    found = _find_self_check_issue(_issue_registry_state)
    assert found is not None, "expected a self-check repair issue to be raised"
    issue_id, issue = found
    # Count encoded in the id (suffix after the language).
    assert issue_id.rsplit("_", 1)[-1].isdigit()
    assert issue["translation_key"] == "self_check_clashes"
    details = issue["translation_placeholders"]["details"]
    # Group heading present, code-fenced source intent name.
    assert "### `IntentA`" in details or "### `IntentB`" in details
    assert "`IntentA`" in details and "`IntentB`" in details
    assert "is matched as" in details


@pytest.mark.asyncio
async def test_self_check_pretty_pattern_restores_slot_names(
    hass, _capture_async_converse, _issue_registry_state
):
    """When a slot-bearing intent clashes, the details render `{slot_name}` not sentinels."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        # Two intents with the same fixed surface but different slot names.
        "Add_Shopping": ["Add {item} to the list"],
        "Add_Todo": ["Add {task} to the list"],
    }
    agent = _make_agent(hass, threshold=70, startup_self_check=True)
    await agent.async_added_to_hass()

    found = _find_self_check_issue(_issue_registry_state)
    assert found is not None, "expected a self-check repair issue to be raised"
    _, issue = found
    details = issue["translation_placeholders"]["details"]
    # Slot names render in {curly braces}, not as the SLOT_WILDCARD sentinel.
    assert "{item}" in details or "{task}" in details
    # The internal sentinel must not leak through.
    assert "zqzqxslotx" not in details
    assert "\x00slot\x00" not in details


@pytest.mark.asyncio
async def test_self_check_clears_prior_issue_on_rebuild(
    hass, _capture_async_converse, _issue_registry_state
):
    """A rebuild that resolves all clashes must delete the previous repair card."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "IntentA": ["Foo bar"],
        "IntentB": ["Foo bar"],
    }
    agent = _make_agent(hass, threshold=70, startup_self_check=True)
    await agent.async_added_to_hass()

    found = _find_self_check_issue(_issue_registry_state)
    assert found is not None
    prior_id, _ = found

    # User removes the duplicate, then we trigger a rebuild via apply_options.
    hass.data[DOMAIN][KEY_CONVERSATION_INTENTS] = {"IntentA": ["Foo bar"]}
    agent.apply_options(
        threshold=70,
        slot_threshold=None,
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        builtin_allowlist=None,
        slot_extraction=True,
        fallback_agent_id="conversation.home_assistant",
        startup_self_check=True,
    )
    # Force a fresh pool build (and therefore a fresh self-check run).
    await agent._async_get_pool("de")

    # Prior id must be gone, no new id created (no clashes).
    assert (DOMAIN, prior_id) not in _issue_registry_state
    assert _find_self_check_issue(_issue_registry_state) is None


@pytest.mark.asyncio
async def test_self_check_replaces_prior_issue_when_count_changes(
    hass, _capture_async_converse, _issue_registry_state
):
    """When the clash count changes, the prior id is deleted before the new one is written."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "IntentA": ["Foo bar"],
        "IntentB": ["Foo bar"],
        "IntentC": ["Foo bar"],
    }
    agent = _make_agent(hass, threshold=70, startup_self_check=True)
    await agent.async_added_to_hass()

    found = _find_self_check_issue(_issue_registry_state)
    assert found is not None
    prior_id, _ = found
    prior_count = int(prior_id.rsplit("_", 1)[-1])

    # Drop one clashing intent -> count decreases on rebuild.
    hass.data[DOMAIN][KEY_CONVERSATION_INTENTS] = {
        "IntentA": ["Foo bar"],
        "IntentB": ["Foo bar"],
    }
    agent.apply_options(
        threshold=70,
        slot_threshold=None,
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        builtin_allowlist=None,
        slot_extraction=True,
        fallback_agent_id="conversation.home_assistant",
        startup_self_check=True,
    )
    await agent._async_get_pool("de")

    found = _find_self_check_issue(_issue_registry_state)
    assert found is not None
    new_id, _ = found
    new_count = int(new_id.rsplit("_", 1)[-1])

    assert new_id != prior_id
    assert new_count != prior_count
    # The old id is gone.
    assert (DOMAIN, prior_id) not in _issue_registry_state


@pytest.mark.asyncio
async def test_self_check_disabled(
    hass, _capture_async_converse, _issue_registry_state, monkeypatch
):
    """With startup_self_check=False, no issue is created even for clashes."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "IntentA": ["Foo bar"],
        "IntentB": ["Foo bar"],
    }
    agent = ClosestIntentAgent(
        hass,
        threshold=70,
        slot_threshold=None,
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        builtin_allowlist=None,
        slot_extraction=True,
        fallback_agent_id="conversation.home_assistant",
        startup_self_check=False,
        entry_id="TESTENTRY",
    )
    await agent.async_added_to_hass()

    assert _issue_registry_state == {}


@pytest.mark.asyncio
async def test_region_language_resolves_to_base_variant(hass, monkeypatch):
    """
    A region variant like ``en-AU`` must resolve to the supported intents variant (``en``)
    the way HA's DefaultAgent does (issue #21).
    """
    hai = types.ModuleType("home_assistant_intents")
    hai.get_languages = lambda: ["en", "de", "fr"]
    monkeypatch.setitem(sys.modules, "home_assistant_intents", hai)

    util = types.ModuleType("homeassistant.util")
    lang_mod = types.ModuleType("homeassistant.util.language")

    def _matches(target, supported, country=None):
        base = target.split("-")[0]
        return [base] if base in supported else []

    lang_mod.matches = _matches
    util.language = lang_mod
    monkeypatch.setitem(sys.modules, "homeassistant.util", util)
    monkeypatch.setitem(sys.modules, "homeassistant.util.language", lang_mod)

    agent = _make_agent(hass, threshold=70)
    assert agent._resolve_language("en-AU") == "en"
    assert agent._resolve_language("de-CH") == "de"
    assert agent._resolve_language("en") == "en"
    assert agent._resolve_language("xx-YY") == "xx-YY"

    await agent._async_get_pool("en-AU")
    assert "en" in agent._pools
    assert "en-AU" not in agent._pools
    pool_en = await agent._async_get_pool("en")
    assert pool_en is agent._pools["en"]


@pytest.mark.asyncio
async def test_self_check_slot_sentinel_routes_back(
    hass, _capture_async_converse, _issue_registry_state
):
    """Slot-bearing intents materialise to sentinels and must round-trip cleanly."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS] = {
        "area": {"values": ["Wohnzimmer", "Büro"]},
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    assert _issue_registry_state == {}


# Allow tests to run without the pytest-asyncio plugin
def pytest_collection_modifyitems(config, items):  # pragma: no cover
    if config.pluginmanager.hasplugin("asyncio"):
        return
    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.usefixtures("_run_async"))


@pytest.fixture
def _run_async():  # pragma: no cover
    yield


def pytest_pyfunc_call(pyfuncitem):  # pragma: no cover
    """Fallback runner: execute async test functions on a fresh loop."""
    func = pyfuncitem.obj
    if not asyncio.iscoroutinefunction(func):
        return None
    fn_args = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    asyncio.run(func(**fn_args))
    return True
