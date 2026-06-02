"""
Fuzzy intent matcher for HomeAssistant. Garbled STT output in, actual intent out.
"""

from __future__ import annotations

import json
import logging

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_BUILTIN_ALLOWLIST,
    CONF_DENYLIST,
    CONF_EXPANSION_CAP,
    CONF_FALLBACK_AGENT,
    CONF_INCLUDE_BUILTINS,
    CONF_SLOT_EXTRACTION,
    CONF_SLOT_THRESHOLD,
    CONF_STARTUP_SELF_CHECK,
    CONF_THRESHOLD,
    DEFAULT_EXPANSION_CAP,
    DEFAULT_FALLBACK_AGENT,
    DEFAULT_INCLUDE_BUILTINS,
    DEFAULT_SLOT_EXTRACTION,
    DEFAULT_STARTUP_SELF_CHECK,
    DEFAULT_THRESHOLD,
    DOMAIN,
    KEY_AGENT_INSTANCES,
    KEY_CONVERSATION_EXPANSION_RULES,
    KEY_CONVERSATION_INTENTS,
    KEY_CONVERSATION_LISTS,
    SERVICE_DUMP_CANDIDATES,
    SERVICE_PARSE,
    VERSION,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_THRESHOLD, default=DEFAULT_THRESHOLD): vol.All(
                    vol.Coerce(int), vol.Range(min=0, max=100)
                ),
                vol.Optional(CONF_SLOT_THRESHOLD): vol.All(
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
                vol.Optional(CONF_BUILTIN_ALLOWLIST, default=None): vol.Any(None, [cv.string]),
                vol.Optional(CONF_SLOT_EXTRACTION, default=DEFAULT_SLOT_EXTRACTION): cv.boolean,
                vol.Optional(
                    CONF_STARTUP_SELF_CHECK, default=DEFAULT_STARTUP_SELF_CHECK
                ): cv.boolean,
                # Conversation entity to forward the canonical sentence to after a fuzzy match.
                # Default is HA's bundled agent.
                vol.Optional(CONF_FALLBACK_AGENT, default=DEFAULT_FALLBACK_AGENT): cv.string,
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


def _format_dump_summary(states: dict[str, dict]) -> str:
    lines: list[str] = [f"closest_intent version {VERSION}", ""]
    for entry_id, state in states.items():
        lines.append(f"Agent {entry_id}")
        lines.append(
            f"  threshold={state['threshold']} "
            f"slot_threshold={state.get('slot_threshold')} "
            f"(effective={state.get('effective_slot_threshold')}) "
            f"expansion_cap={state['expansion_cap']} "
            f"include_builtins={state['include_builtins']} "
            f"slot_extraction={state['slot_extraction']}"
        )
        if state.get("fallback_agent_id"):
            lines.append(f"  fallback_agent={state['fallback_agent_id']}")
        if state.get("denylist"):
            lines.append(f"  denylist={', '.join(state['denylist'])}")
        for lang, ls in state["languages"].items():
            lines.append(
                f"  [{lang}] {ls['user_candidate_count']} user / "
                f"{ls['builtin_candidate_count']} builtin candidates"
            )
            for label, key in (("user", "user_intents"), ("builtin", "builtin_intents")):
                intents = ls.get(key) or {}
                if not intents:
                    continue
                lines.append(f"    {label} intents:")
                for intent, texts in sorted(intents.items()):
                    lines.append(f"      - {intent} ({len(texts)})")
        lines.append("")
    return "\n".join(lines).rstrip()


def _async_register_services(hass: HomeAssistant) -> None:
    """
    Register the developer-facing dump_candidates service.

    Called once during ``async_setup``.
    Service is a no-op until at least one config entry has been loaded.
    """
    if hass.services.has_service(DOMAIN, SERVICE_DUMP_CANDIDATES):
        return

    dump_schema = vol.Schema(
        {
            vol.Optional("include_builtins", default=False): cv.boolean,
            vol.Optional("intent_filter"): cv.string,
            vol.Optional("include_exposure", default=False): cv.boolean,
        }
    )

    async def _dump(call: ServiceCall) -> ServiceResponse:
        data = dump_schema(dict(call.data))
        include_builtins = data["include_builtins"]
        intent_filter = data.get("intent_filter")
        include_exposure = data["include_exposure"]

        agents = hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {})
        if not agents:
            _LOGGER.warning("dump_candidates: no agent instances registered yet")
            return {
                "version": VERSION,
                "agents": {},
                "warning": "no agent instances registered yet",
            }

        states: dict[str, dict] = {}
        for entry_id, agent in agents.items():
            builtin_overrides: dict | None = None
            if include_builtins:
                builtin_overrides = {}
                for lang, (resolver, _, _) in agent._pools.items():
                    try:
                        builtin_overrides[lang] = await agent._async_get_builtin_override(
                            lang, resolver
                        )
                    except Exception:  # pragma: no cover
                        _LOGGER.exception(
                            "dump_candidates[%s]: builtin override build failed",
                            lang,
                        )
            state = agent.dump_state(
                builtin_overrides=builtin_overrides,
                intent_filter=intent_filter,
                include_exposure=include_exposure,
            )
            states[entry_id] = state
            # Pretty-print at DEBUG so users can paste a single block when
            # filing issues. INFO line is a one-liner pointer.
            _LOGGER.info(
                "dump_candidates[%s]: %d candidate(s) across %d language(s) (full state at DEBUG)",
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
            _LOGGER.debug("dump_candidates[%s] full state:\n%s", entry_id, pretty)

        return {
            "version": VERSION,
            "summary": _format_dump_summary(states),
            "agents": states,
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_DUMP_CANDIDATES,
        _dump,
        supports_response=SupportsResponse.OPTIONAL,
    )

    parse_schema = vol.Schema(
        {
            vol.Required("sentence"): cv.string,
            vol.Optional("language"): cv.string,
            vol.Optional("entry_id"): cv.string,
            vol.Optional("include_builtins", default=False): cv.boolean,
            vol.Optional("run_official", default=False): cv.boolean,
            vol.Optional("debug_top_candidates", default=False): cv.boolean,
        }
    )

    async def _parse(call: ServiceCall) -> ServiceResponse:
        data = parse_schema(dict(call.data))
        agents = hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {})
        if not agents:
            return {"error": "no agent instances registered yet"}

        entry_id = data.get("entry_id")
        if entry_id is not None:
            agent = agents.get(entry_id)
            if agent is None:
                return {
                    "error": f"unknown entry_id {entry_id!r}",
                    "available_entry_ids": sorted(agents.keys()),
                }
        else:
            entry_id, agent = next(iter(agents.items()))

        language = data.get("language") or hass.config.language or "en"
        result = await agent.parse_sentence(
            language,
            data["sentence"],
            run_official=data["run_official"],
            include_builtins=data["include_builtins"],
            debug_top_candidates=data["debug_top_candidates"],
        )
        result["entry_id"] = entry_id
        _LOGGER.info(
            "parse[%s][%s] %r -> matched=%s intent=%s canonical=%r [%.1fms]",
            entry_id,
            language,
            data["sentence"],
            result.get("matched"),
            result.get("intent"),
            result.get("canonical"),
            (result.get("pools") or {}).get("match_ms", 0.0),
        )
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_PARSE,
        _parse,
        supports_response=SupportsResponse.ONLY,
    )
