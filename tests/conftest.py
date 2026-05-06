"""
Stub Home Assistant modules so tests can import conversation.py.

The real HA package isn't in the dev shell. Tests only need the surface
area conversation.py imports, so we insert stubs into ``sys.modules``.

This conftest also extends ``sys.path`` with the package directory so
``from conversation import ...`` works in tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

PKG_DIR = Path(__file__).resolve().parent.parent / "custom_components" / "closest_intent"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    # Wire as attribute on parent so getattr-based lookups traverse correctly.
    if "." in name:
        parent_name, _, child = name.rpartition(".")
        parent = _ensure_module(parent_name)
        setattr(parent, child, mod)
    return mod


ha = _ensure_module("homeassistant")
ha_components = _ensure_module("homeassistant.components")
ha_components_conversation = _ensure_module("homeassistant.components.conversation")
ha_config_entries = _ensure_module("homeassistant.config_entries")
ha_const = _ensure_module("homeassistant.const")
ha_core = _ensure_module("homeassistant.core")
ha_helpers = _ensure_module("homeassistant.helpers")
ha_helpers_intent = _ensure_module("homeassistant.helpers.intent")
ha_helpers_entity_platform = _ensure_module("homeassistant.helpers.entity_platform")
ha_helpers_event = _ensure_module("homeassistant.helpers.event")
ha_helpers_typing = _ensure_module("homeassistant.helpers.typing")
ha_helpers_cv = _ensure_module("homeassistant.helpers.config_validation")
ha_helpers_selector = _ensure_module("homeassistant.helpers.selector")

_ensure_module("homeassistant.helpers.area_registry").async_get = lambda hass: (
    types.SimpleNamespace(async_list_areas=lambda: [])
)
_ensure_module("homeassistant.helpers.floor_registry").async_get = lambda hass: (
    types.SimpleNamespace(async_list_floors=lambda: [])
)
_ensure_module("homeassistant.helpers.entity_registry").async_get = lambda hass: (
    types.SimpleNamespace(entities={})
)


class _ConversationEntityFeature:
    CONTROL = 1


class _ConversationEntity:
    """Minimal stand-in for HA's ConversationEntity."""

    hass = None

    async def async_added_to_hass(self) -> None:
        return None

    async def async_will_remove_from_hass(self) -> None:
        return None


class _ConversationInput:
    def __init__(
        self,
        *,
        text: str,
        language: str | None = None,
        conversation_id: str | None = None,
        context=None,
    ) -> None:
        self.text = text
        self.language = language
        self.conversation_id = conversation_id
        self.context = context


class _ConversationResult:
    def __init__(self, response=None, conversation_id=None) -> None:
        self.response = response
        self.conversation_id = conversation_id


async def _async_converse(*, text: str, **kwargs):
    """Default stub. Overridden per-test via monkeypatch."""
    return _ConversationResult(response={"text": text, "kwargs": kwargs})


ha_components_conversation.ConversationEntity = _ConversationEntity
ha_components_conversation.ConversationEntityFeature = _ConversationEntityFeature
ha_components_conversation.ConversationInput = _ConversationInput
ha_components_conversation.ConversationResult = _ConversationResult
ha_components_conversation.MATCH_ALL = "*"
ha_components_conversation.async_converse = _async_converse


ha_components_conversation_const = _ensure_module("homeassistant.components.conversation.const")
ha_components_conversation_const.DOMAIN = "conversation"


class _IntentResponseErrorCode:
    NO_INTENT_MATCH = "no_intent_match"


class _IntentResponse:
    def __init__(self, language: str | None = None) -> None:
        self.language = language
        self.error_code = None
        self.error_message = None

    def async_set_error(self, code, message: str) -> None:
        self.error_code = code
        self.error_message = message


ha_helpers_intent.IntentResponse = _IntentResponse
ha_helpers_intent.IntentResponseErrorCode = _IntentResponseErrorCode


def _async_call_later(hass, delay, action):
    """Return a no-op cancel callable. Tests drive rebuilds explicitly."""
    hass._scheduled_actions.append((delay, action))

    def _cancel() -> None:
        return None

    return _cancel


ha_helpers_event.async_call_later = _async_call_later


ha_helpers_entity_platform.AddEntitiesCallback = MagicMock
ha_helpers_typing.ConfigType = dict
ha_helpers_cv.string = str
ha_helpers_cv.boolean = bool


def _callback_passthrough(fn):
    return fn


ha_core.callback = _callback_passthrough


class _HomeAssistant:
    """Minimal hass stand-in for the agent's executor + data needs."""


ha_core.HomeAssistant = _HomeAssistant


class _ConfigEntry:
    def __init__(
        self,
        *,
        entry_id: str = "TESTENTRY",
        data: dict | None = None,
        options: dict | None = None,
    ) -> None:
        self.entry_id = entry_id
        self.data = data or {}
        self.options = options or {}

    def async_on_unload(self, _func) -> None:
        return None

    def add_update_listener(self, _listener):
        def _unsub() -> None:
            return None

        return _unsub


ha_config_entries.ConfigEntry = _ConfigEntry
ha_config_entries.SOURCE_IMPORT = "import"


class _ConfigFlow:  # pragma: no cover
    def __init_subclass__(cls, **kwargs) -> None:
        kwargs.pop("domain", None)
        super().__init_subclass__(**kwargs)


ha_config_entries.ConfigFlow = _ConfigFlow


class _OptionsFlow:  # pragma: no cover
    pass


ha_config_entries.OptionsFlow = _OptionsFlow


class _Platform:
    CONVERSATION = "conversation"


ha_const.Platform = _Platform
