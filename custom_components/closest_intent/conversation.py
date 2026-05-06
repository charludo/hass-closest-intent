"""
Closest-intent conversation entity.
"""

from __future__ import annotations

import asyncio
import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Importable both as part of the package and as a standalone module for tests.
try:
    from .const import (
        CONF_BASE_AGENT,
        CONF_EXPANSION_CAP,
        CONF_SLOT_EXTRACTION,
        CONF_THRESHOLD,
        DEFAULT_BASE_AGENT,
        DEFAULT_EXPANSION_CAP,
        DEFAULT_SLOT_EXTRACTION,
        DEFAULT_THRESHOLD,
        DOMAIN,
        KEY_AGENT_INSTANCES,
        KEY_CONVERSATION_INTENTS,
    )
    from .matching import (
        Candidate,
        Resolver,
        build_canonical,
        expand_pattern,
        extract_slots,
        find_best,
        score,
    )
except ImportError:  # pragma: no cover
    from const import (  # type: ignore
        CONF_BASE_AGENT,
        CONF_EXPANSION_CAP,
        CONF_SLOT_EXTRACTION,
        CONF_THRESHOLD,
        DEFAULT_BASE_AGENT,
        DEFAULT_EXPANSION_CAP,
        DEFAULT_SLOT_EXTRACTION,
        DEFAULT_THRESHOLD,
        DOMAIN,
        KEY_AGENT_INSTANCES,
        KEY_CONVERSATION_INTENTS,
    )
    from matching import (  # type: ignore
        Candidate,
        Resolver,
        build_canonical,
        expand_pattern,
        extract_slots,
        find_best,
        score,
    )

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    data = entry.data
    options = entry.options or {}

    def opt(key, default):
        return options.get(key, data.get(key, default))

    agent = ClosestIntentAgent(
        hass,
        threshold=opt(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        expansion_cap=opt(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
        slot_extraction=opt(CONF_SLOT_EXTRACTION, DEFAULT_SLOT_EXTRACTION),
        base_agent_id=opt(CONF_BASE_AGENT, DEFAULT_BASE_AGENT),
        entry_id=entry.entry_id,
    )
    hass.data.setdefault(DOMAIN, {}).setdefault(KEY_AGENT_INSTANCES, {})[entry.entry_id] = agent
    async_add_entities([agent])


class ClosestIntentAgent(conversation.ConversationEntity):
    _attr_has_entity_name = True
    _attr_name = "Closest Intent"
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        threshold: int,
        expansion_cap: int,
        slot_extraction: bool,
        base_agent_id: str,
        entry_id: str,
    ) -> None:
        self.hass = hass
        self._threshold = threshold
        self._expansion_cap = expansion_cap
        self._slot_extraction = slot_extraction
        self._base_agent_id = base_agent_id
        self._entry_id = entry_id

        # Per-language pools: built lazily on first request for that
        # language. A user with multiple Assist pipelines in different
        # languages gets a fresh pool for each one.
        self._pools: dict[str, tuple[Resolver, list[Candidate]]] = {}
        self._pool_locks: dict[str, asyncio.Lock] = {}

        self._attr_unique_id = "closest_intent_agent"

    @property
    def supported_languages(self) -> list[str] | str:
        return conversation.MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # Pre-warm with HA's configured default language so first user
        # utterance doesn't pay the build cost.
        default_lang = self.hass.config.language or "en"
        await self._async_get_pool(default_lang)

    async def async_will_remove_from_hass(self) -> None:
        self.hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {}).pop(self._entry_id, None)
        await super().async_will_remove_from_hass()

    async def _async_get_pool(self, language: str) -> tuple[Resolver, list[Candidate]]:
        cached = self._pools.get(language)
        if cached is not None:
            return cached
        lock = self._pool_locks.setdefault(language, asyncio.Lock())
        async with lock:
            cached = self._pools.get(language)
            if cached is not None:
                return cached
            pool = await self.hass.async_add_executor_job(self._build_pool, language)
            self._pools[language] = pool
            return pool

    def _build_pool(self, language: str) -> tuple[Resolver, list[Candidate]]:
        resolver = Resolver()
        intents = self._gather_intents()

        candidates: list[Candidate] = []
        for intent_name, patterns in intents.items():
            for idx, pat in enumerate(patterns):
                for text, slot_names in expand_pattern(pat, self._expansion_cap, resolver=resolver):
                    candidates.append(
                        Candidate(
                            intent=intent_name,
                            pattern_idx=idx,
                            text=text,
                            slot_names=slot_names,
                        )
                    )

        _LOGGER.info(
            "closest_intent[%s]: built %d candidates across %d intents",
            language,
            len(candidates),
            len(intents),
        )
        return (resolver, candidates)

    def _gather_intents(self) -> dict[str, list[str]]:
        gathered: dict[str, list[str]] = {}
        conv_intents = self.hass.data.get(DOMAIN, {}).get(KEY_CONVERSATION_INTENTS, {})
        for name, patterns in conv_intents.items():
            if isinstance(patterns, str):
                gathered[name] = [patterns]
            else:
                gathered[name] = list(patterns)
        return gathered

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        language = user_input.language or self.hass.config.language or "en"
        try:
            resolver, candidates = await self._async_get_pool(language)
        except Exception:  # pragma: no cover
            _LOGGER.exception("closest_intent: failed to build pool for language %s", language)
            resolver, candidates = Resolver(), []

        try:
            canonical = self._best_canonical(user_input, resolver, candidates)
        except Exception:  # pragma: no cover
            _LOGGER.exception("closest_intent: unexpected error matching %r", user_input.text)
            canonical = None

        forwarded_text = canonical if canonical is not None else user_input.text

        try:
            return await conversation.async_converse(
                hass=self.hass,
                text=forwarded_text,
                conversation_id=user_input.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=self._base_agent_id,
            )
        except Exception:
            _LOGGER.exception("closest_intent: forwarding to %s failed", self._base_agent_id)
            return _no_match(user_input)

    def _best_canonical(
        self,
        user_input: conversation.ConversationInput,
        resolver: Resolver,
        candidates: list[Candidate],
    ) -> str | None:
        match = find_best(user_input.text, candidates, self._threshold)
        if match is None:
            _LOGGER.debug(
                "closest_intent: no match for %r above %d, passthrough",
                user_input.text,
                self._threshold,
            )
            return None

        candidate, score_value = match

        if candidate.has_slots:
            if not self._slot_extraction:
                return None
            captured = extract_slots(user_input.text, candidate)
            if captured is None:
                fallback = self._best_extractable_sibling(
                    user_input.text, candidate.intent, candidates
                )
                if fallback is None:
                    _LOGGER.debug(
                        "closest_intent: matched %s (score=%d) but no expansion extracted, "
                        "passthrough",
                        candidate.intent,
                        score_value,
                    )
                    return None
                candidate, captured, score_value = fallback
        else:
            captured = []

        canonical = build_canonical(candidate, captured, resolver=resolver)
        _LOGGER.info(
            "closest_intent: %r -> %s (score=%d, captured=%s) -> forwarding %r to %s",
            user_input.text,
            candidate.intent,
            score_value,
            captured,
            canonical,
            self._base_agent_id,
        )
        return canonical

    def _best_extractable_sibling(
        self,
        user_text: str,
        intent_name: str,
        candidates: list[Candidate],
    ) -> tuple[Candidate, list[str], int] | None:
        """
        Among same-intent expansions, return the highest-scoring one
        whose slots actually extract.
        """
        scored: list[tuple[int, Candidate]] = []
        for c in candidates:
            if c.intent != intent_name or not c.has_slots:
                continue
            scored.append((score(user_text, c.text), c))
        scored.sort(key=lambda x: -x[0])
        for s, c in scored:
            if s < self._threshold:
                break
            captured = extract_slots(user_text, c)
            if captured is not None:
                return (c, captured, s)
        return None

    def dump_state(self) -> dict:
        """Return a plain-data snapshot of pools for the diagnostic service."""
        out: dict = {
            "entry_id": self._entry_id,
            "threshold": self._threshold,
            "expansion_cap": self._expansion_cap,
            "slot_extraction": self._slot_extraction,
            "base_agent_id": self._base_agent_id,
            "languages": {},
        }
        for lang, (resolver, candidates) in self._pools.items():
            by_intent: dict[str, list[str]] = {}
            for c in candidates:
                by_intent.setdefault(c.intent, []).append(c.text)
            out["languages"][lang] = {
                "candidate_count": len(candidates),
                "intents": by_intent,
                "expansion_rules": {k: v for k, v in resolver.expansion_rules.items()},
                "slot_values": {k: v for k, v in resolver.slot_values.items()},
            }
        return out


def _no_match(
    user_input: conversation.ConversationInput,
) -> conversation.ConversationResult:
    response = intent_helper.IntentResponse(language=user_input.language)
    response.async_set_error(
        intent_helper.IntentResponseErrorCode.NO_INTENT_MATCH,
        "No matching intent.",
    )
    return conversation.ConversationResult(
        response=response,
        conversation_id=user_input.conversation_id,
    )
