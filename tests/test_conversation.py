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

    hass = SimpleNamespace(
        data={},
        bus=FakeBus(),
        states=FakeStates(),
        services=FakeServices(),
        _scheduled_actions=[],
        async_add_executor_job=_run_in_executor,
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
    expansion_cap: int = 16,
    denylist=None,
    include_builtins: bool = False,
    slot_extraction: bool = True,
    base_agent_id: str = "conversation.home_assistant",
) -> ClosestIntentAgent:
    return ClosestIntentAgent(
        hass,
        threshold=threshold,
        expansion_cap=expansion_cap,
        denylist=denylist,
        include_builtins=include_builtins,
        slot_extraction=slot_extraction,
        base_agent_id=base_agent_id,
        entry_id="TESTENTRY",
    )


def _conversation_input(text: str, language: str = "de"):
    """Build a stubbed ConversationInput."""
    from homeassistant.components.conversation import ConversationInput  # type: ignore

    return ConversationInput(text=text, language=language)


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
    """Fuzzy match -> canonical sentence (lowercased, normalised) forwarded."""
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "PumpeAn": ["Pumpe an"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("pumpr an"))
    assert _capture_async_converse["text"] == "pumpe an"


@pytest.mark.asyncio
async def test_slot_extraction_and_resolution(hass, _capture_async_converse):
    """Slot pattern matched + captured text fuzz-resolved against registry."""
    import sys as _sys

    fake_ar = _sys.modules.setdefault(
        "homeassistant.helpers.area_registry", type(_sys)("homeassistant.helpers.area_registry")
    )
    fake_fr = _sys.modules.setdefault(
        "homeassistant.helpers.floor_registry", type(_sys)("homeassistant.helpers.floor_registry")
    )
    fake_er = _sys.modules.setdefault(
        "homeassistant.helpers.entity_registry", type(_sys)("homeassistant.helpers.entity_registry")
    )

    class _Area:
        def __init__(self, name: str, aliases=None) -> None:
            self.name = name
            self.aliases = aliases or []

    fake_ar.async_get = lambda hass: SimpleNamespace(
        async_list_areas=lambda: [_Area("Wohnzimmer"), _Area("Büro")]
    )
    fake_fr.async_get = lambda hass: SimpleNamespace(async_list_floors=lambda: [])
    fake_er.async_get = lambda hass: SimpleNamespace(entities={})

    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    await agent.async_process(_conversation_input("test zwei im wohnzma"))
    assert _capture_async_converse["text"] == "test zwei im wohnzimmer"


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
async def test_registry_change_triggers_rebuild(hass, _capture_async_converse):
    """Firing area_registry_updated invalidates pools and new areas show up."""
    import sys as _sys

    areas = [SimpleNamespace(name="Wohnzimmer", aliases=[])]
    fake_ar = _sys.modules.setdefault(
        "homeassistant.helpers.area_registry", type(_sys)("homeassistant.helpers.area_registry")
    )
    fake_fr = _sys.modules.setdefault(
        "homeassistant.helpers.floor_registry", type(_sys)("homeassistant.helpers.floor_registry")
    )
    fake_er = _sys.modules.setdefault(
        "homeassistant.helpers.entity_registry", type(_sys)("homeassistant.helpers.entity_registry")
    )
    fake_ar.async_get = lambda hass: SimpleNamespace(async_list_areas=lambda: list(areas))
    fake_fr.async_get = lambda hass: SimpleNamespace(async_list_floors=lambda: [])
    fake_er.async_get = lambda hass: SimpleNamespace(entities={})

    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = {
        "Test_Area": ["Test zwei im {area}"],
    }
    agent = _make_agent(hass, threshold=70)
    await agent.async_added_to_hass()

    pool = agent._pools["de"]
    assert pool[0].slot_values.get("area") == ["Wohnzimmer"]

    areas.append(SimpleNamespace(name="Küche", aliases=[]))
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
        expansion_cap=16,
        denylist=None,
        include_builtins=False,
        slot_extraction=True,
        base_agent_id="conversation.home_assistant",
    )
    assert agent._pools == {}


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
