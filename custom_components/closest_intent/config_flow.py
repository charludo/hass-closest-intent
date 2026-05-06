"""
Config + options flow for closest_intent.

Offers entry points both for YAML and UI config.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

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
    KEY_CONVERSATION_INTENTS,
)


def _discovered_intent_names(hass) -> list[str]:
    """
    Best-effort list of intent names known at flow time.

    Pulled from the YAML stash populated by ``async_setup``.
    Empty when the user is configuring through the UI without any
    ``conversation.intents`` block.
    """
    stash = hass.data.get(DOMAIN, {}) or {}
    intents = stash.get(KEY_CONVERSATION_INTENTS, {}) or {}
    return sorted(intents.keys())


def _build_schema(
    hass,
    defaults: dict[str, Any],
) -> vol.Schema:
    """Schema shared by user + options flows."""
    discovered = _discovered_intent_names(hass)

    if discovered:
        denylist_selector = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=discovered,
                multiple=True,
                custom_value=True,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    else:
        # No intents discovered yet (UI-only setup). Fall back to a free-form
        # text-list selector so users can still type names.
        denylist_selector = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[],
                multiple=True,
                custom_value=True,
                mode=selector.SelectSelectorMode.LIST,
            )
        )

    base_agent_default = defaults.get(CONF_BASE_AGENT, DEFAULT_BASE_AGENT)
    denylist_default = defaults.get(CONF_DENYLIST) or []

    return vol.Schema(
        {
            vol.Required(
                CONF_THRESHOLD,
                default=defaults.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, max=100, step=1, mode=selector.NumberSelectorMode.SLIDER
                )
            ),
            vol.Required(
                CONF_EXPANSION_CAP,
                default=defaults.get(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, max=512, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_DENYLIST,
                default=denylist_default,
            ): denylist_selector,
            vol.Required(
                CONF_SLOT_EXTRACTION,
                default=defaults.get(CONF_SLOT_EXTRACTION, DEFAULT_SLOT_EXTRACTION),
            ): selector.BooleanSelector(),
            vol.Required(
                CONF_INCLUDE_BUILTINS,
                default=defaults.get(CONF_INCLUDE_BUILTINS, DEFAULT_INCLUDE_BUILTINS),
            ): selector.BooleanSelector(),
            vol.Required(
                CONF_BASE_AGENT,
                default=base_agent_default,
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="conversation")),
        }
    )


def _normalise(user_input: dict[str, Any]) -> dict[str, Any]:
    """Coerce selector outputs into the types the entity expects."""
    out = dict(user_input)
    if CONF_THRESHOLD in out:
        out[CONF_THRESHOLD] = int(out[CONF_THRESHOLD])
    if CONF_EXPANSION_CAP in out:
        out[CONF_EXPANSION_CAP] = int(out[CONF_EXPANSION_CAP])
    deny = out.get(CONF_DENYLIST)
    if not deny:
        out[CONF_DENYLIST] = None
    return out


class ClosestIntentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for closest_intent."""

    VERSION = 1

    async def async_step_import(self, import_data: dict[str, Any]) -> Any:
        """
        Handle a YAML import.

        YAML remains the source of truth when present: subsequent
        imports overwrite the existing entry's ``data`` so edits to
        ``configuration.yaml`` propagate on restart.
        """
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured(updates=import_data)
        return self.async_create_entry(title="Closest Intent", data=import_data)

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> Any:
        """UI setup. Single step. Same fields as options flow."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            return self.async_create_entry(
                title="Closest Intent",
                data=_normalise(user_input),
            )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema(self.hass, defaults={}),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> ClosestIntentOptionsFlow:
        return ClosestIntentOptionsFlow(config_entry)


class ClosestIntentOptionsFlow(config_entries.OptionsFlow):
    """
    Live-tweak any field after initial setup.

    Note for YAML users: options here override YAML on a per-key basis.
    To return to YAML-only behaviour, clear the override in the UI.
    The entity falls back to ``entry.data`` (i.e. the latest YAML import).
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        # Don't assign to self.config_entry. Newer HA versions provide
        # it as a read-only property and writing raises a deprecation
        # warning. Stash a reference for our own use instead.
        self._entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> Any:
        if user_input is not None:
            return self.async_create_entry(title="", data=_normalise(user_input))

        # Defaults: prefer existing options, fall back to YAML data.
        defaults: dict[str, Any] = {}
        defaults.update(self._entry.data or {})
        defaults.update(self._entry.options or {})

        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(self.hass, defaults=defaults),
        )
