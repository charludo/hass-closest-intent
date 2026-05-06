"""
Closest-intent fuzzy-fallback conversation agent for Home Assistant.
"""

from __future__ import annotations

import json
import logging

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_BASE_AGENT,
    CONF_DENYLIST,
    CONF_EXPANSION_CAP,
    CONF_INCLUDE_BUILTINS,
    CONF_SLOT_EXTRACTION,
    CONF_THRESHOLD,
    DEFAULT_BASE_AGENT,
    DEFAULT_EXPANSION_CAP,
    DEFAULT_INCLUDE_BUILTINS,
    DEFAULT_SLOT_EXTRACTION,
    DEFAULT_THRESHOLD,
    DOMAIN,
    KEY_AGENT_INSTANCES,
    KEY_CONVERSATION_EXPANSION_RULES,
    KEY_CONVERSATION_INTENTS,
    KEY_CONVERSATION_LISTS,
    SERVICE_DUMP_CANDIDATES,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_THRESHOLD, default=DEFAULT_THRESHOLD): vol.All(
                    vol.Coerce(int), vol.Range(min=0, max=100)
                ),
                vol.Optional(CONF_EXPANSION_CAP, default=DEFAULT_EXPANSION_CAP): vol.All(
                    vol.Coerce(int), vol.Range(min=0)
                ),
                # Exclude specific intent names from matching. Default = exclude none.
                vol.Optional(CONF_DENYLIST, default=None): vol.Any(None, [cv.string]),
                # Also fuzzy-match HA's built-in intent patterns (HassTurnOn etc.)
                # loaded from `home_assistant_intents`.
                vol.Optional(CONF_INCLUDE_BUILTINS, default=DEFAULT_INCLUDE_BUILTINS): cv.boolean,
                vol.Optional(CONF_SLOT_EXTRACTION, default=DEFAULT_SLOT_EXTRACTION): cv.boolean,
                # Conversation entity to forward the canonical sentence to after a fuzzy match.
                # Default is HA's bundled agent.
                vol.Optional(CONF_BASE_AGENT, default=DEFAULT_BASE_AGENT): cv.string,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """
    Capture the user's `conversation` block and bootstrap the agent.

    HA's conversation integration validates only `intents:` strictly, but
    the YAML schema accepts the full Hassil-style `lists:` and
    `expansion_rules:` blocks (``extra=ALLOW_EXTRA``). We stash all three
    so the conversation entity can layer user-defined lists/rules on top
    of the language pack's defaults.
    """
    conv = config.get("conversation") or {}
    hass.data.setdefault(DOMAIN, {})[KEY_CONVERSATION_INTENTS] = dict(conv.get("intents") or {})
    hass.data[DOMAIN][KEY_CONVERSATION_LISTS] = dict(conv.get("lists") or {})
    hass.data[DOMAIN][KEY_CONVERSATION_EXPANSION_RULES] = dict(conv.get("expansion_rules") or {})

    _async_register_services(hass)

    if DOMAIN not in config:
        return True

    hass.async_create_task(
        hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": SOURCE_IMPORT},
            data=dict(config[DOMAIN]),
        )
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the conversation entity from a config entry."""
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok


def _async_register_services(hass: HomeAssistant) -> None:
    """
    Register the developer-facing dump_candidates service.

    Called once during ``async_setup``.
    Service is a no-op until at least one config entry has been loaded.
    """
    if hass.services.has_service(DOMAIN, SERVICE_DUMP_CANDIDATES):
        return

    async def _dump(call: ServiceCall) -> None:
        agents = hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {})
        if not agents:
            _LOGGER.warning("closest_intent.dump_candidates: no agent instances registered yet")
            return

        for entry_id, agent in agents.items():
            state = agent.dump_state()
            # Pretty-print at DEBUG so users can paste a single block when
            # filing issues. INFO line is a one-liner pointer.
            _LOGGER.info(
                "closest_intent.dump_candidates[%s]: %d candidate(s) across %d language(s); "
                "see DEBUG for details",
                entry_id,
                sum(
                    lang_state["user_candidate_count"] + lang_state["builtin_candidate_count"]
                    for lang_state in state["languages"].values()
                ),
                len(state["languages"]),
            )
            try:
                pretty = json.dumps(state, indent=2, ensure_ascii=False)
            except Exception:  # pragma: no cover
                pretty = repr(state)
            _LOGGER.debug(
                "closest_intent.dump_candidates[%s] full state:\n%s",
                entry_id,
                pretty,
            )

    hass.services.async_register(DOMAIN, SERVICE_DUMP_CANDIDATES, _dump)
