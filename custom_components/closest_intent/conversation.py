"""
Closest-intent conversation entity.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later

# Importable both as part of the package and as a standalone module for tests.
try:
    from .const import (
        CONF_BUILTIN_ALLOWLIST,
        CONF_DENYLIST,
        CONF_EXPANSION_CAP,
        CONF_FALLBACK_AGENT,
        CONF_INCLUDE_BUILTINS,
        CONF_SLOT_EXTRACTION,
        CONF_SLOT_THRESHOLD,
        CONF_THRESHOLD,
        DEFAULT_EXPANSION_CAP,
        DEFAULT_FALLBACK_AGENT,
        DEFAULT_INCLUDE_BUILTINS,
        DEFAULT_SLOT_EXTRACTION,
        DEFAULT_THRESHOLD,
        DOMAIN,
        KEY_AGENT_INSTANCES,
        KEY_CONVERSATION_EXPANSION_RULES,
        KEY_CONVERSATION_INTENTS,
        KEY_CONVERSATION_LISTS,
        PER_INTENT_CANDIDATE_CAP,
        SLOT_WILDCARD,
        VERSION,
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
        CONF_BUILTIN_ALLOWLIST,
        CONF_DENYLIST,
        CONF_EXPANSION_CAP,
        CONF_FALLBACK_AGENT,
        CONF_INCLUDE_BUILTINS,
        CONF_SLOT_EXTRACTION,
        CONF_SLOT_THRESHOLD,
        CONF_THRESHOLD,
        DEFAULT_EXPANSION_CAP,
        DEFAULT_FALLBACK_AGENT,
        DEFAULT_INCLUDE_BUILTINS,
        DEFAULT_SLOT_EXTRACTION,
        DEFAULT_THRESHOLD,
        DOMAIN,
        KEY_AGENT_INSTANCES,
        KEY_CONVERSATION_EXPANSION_RULES,
        KEY_CONVERSATION_INTENTS,
        KEY_CONVERSATION_LISTS,
        PER_INTENT_CANDIDATE_CAP,
        SLOT_WILDCARD,
        VERSION,
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

_REGISTRY_REBUILD_DEBOUNCE_S = 2.0

_HASSIL_AGENT_ID = "conversation.home_assistant"


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
        slot_threshold=opt(CONF_SLOT_THRESHOLD, None),
        expansion_cap=opt(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
        denylist=opt(CONF_DENYLIST, None),
        include_builtins=opt(CONF_INCLUDE_BUILTINS, DEFAULT_INCLUDE_BUILTINS),
        builtin_allowlist=opt(CONF_BUILTIN_ALLOWLIST, None),
        slot_extraction=opt(CONF_SLOT_EXTRACTION, DEFAULT_SLOT_EXTRACTION),
        fallback_agent_id=opt(CONF_FALLBACK_AGENT, DEFAULT_FALLBACK_AGENT),
        entry_id=entry.entry_id,
    )
    hass.data.setdefault(DOMAIN, {}).setdefault(KEY_AGENT_INSTANCES, {})[entry.entry_id] = agent

    # Pick up live option changes without an HA restart.
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    async_add_entities([agent])


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    agent: ClosestIntentAgent | None = (
        hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {}).get(entry.entry_id)
    )
    if agent is None:
        return
    options = entry.options or {}
    data = entry.data

    def opt(key, default):
        return options.get(key, data.get(key, default))

    agent.apply_options(
        threshold=opt(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        slot_threshold=opt(CONF_SLOT_THRESHOLD, None),
        expansion_cap=opt(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
        denylist=opt(CONF_DENYLIST, None),
        include_builtins=opt(CONF_INCLUDE_BUILTINS, DEFAULT_INCLUDE_BUILTINS),
        builtin_allowlist=opt(CONF_BUILTIN_ALLOWLIST, None),
        slot_extraction=opt(CONF_SLOT_EXTRACTION, DEFAULT_SLOT_EXTRACTION),
        fallback_agent_id=opt(CONF_FALLBACK_AGENT, DEFAULT_FALLBACK_AGENT),
    )


class ClosestIntentAgent(conversation.ConversationEntity):
    _attr_has_entity_name = True
    _attr_name = "Closest Intent"
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        threshold: int,
        slot_threshold: int | None,
        expansion_cap: int,
        denylist: list[str] | None,
        include_builtins: bool,
        builtin_allowlist: list[str] | None,
        slot_extraction: bool,
        fallback_agent_id: str,
        entry_id: str,
    ) -> None:
        self.hass = hass
        self._threshold = threshold
        self._slot_threshold = slot_threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
        self._builtin_allowlist = set(builtin_allowlist) if builtin_allowlist else None
        self._slot_extraction = slot_extraction
        self._fallback_agent_id = fallback_agent_id
        self._entry_id = entry_id

        # Per-language pools: built lazily on first request for that
        # language. A user with multiple Assist pipelines in different
        # languages gets a fresh pool for each one.
        # Tuple is (resolver, user_candidates, builtin_candidates).
        # Builtins are kept separate so we can fall back to them only when
        # the user pool produces no match.
        self._pools: dict[str, tuple[Resolver, list[Candidate], list[Candidate]]] = {}
        self._pool_locks: dict[str, asyncio.Lock] = {}
        self._builtin_intents_cache: dict[str, dict[str, list[str]]] = {}
        self._builtin_overrides: dict[str, list[Candidate]] = {}
        self._builtin_override_locks: dict[str, asyncio.Lock] = {}
        self._rebuild_handle = None  # async_call_later cancel handle
        self._unsub_listeners: list = []

        self._attr_unique_id = "closest_intent_agent"

    @property
    def supported_languages(self) -> list[str] | str:
        return conversation.MATCH_ALL

    @property
    def _effective_slot_threshold(self) -> int:
        return self._slot_threshold if self._slot_threshold is not None else self._threshold

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # Pre-warm with HA's configured default language so first user
        # utterance doesn't pay the build cost.
        default_lang = self.hass.config.language or "en"
        await self._async_get_pool(default_lang)

        bus = self.hass.bus
        for event_name in (
            "area_registry_updated",
            "entity_registry_updated",
            "floor_registry_updated",
        ):
            self._unsub_listeners.append(bus.async_listen(event_name, self._on_registry_event))

    async def async_will_remove_from_hass(self) -> None:
        for unsub in self._unsub_listeners:
            unsub()
        self._unsub_listeners.clear()
        if self._rebuild_handle is not None:
            self._rebuild_handle()
            self._rebuild_handle = None
        self.hass.data.get(DOMAIN, {}).get(KEY_AGENT_INSTANCES, {}).pop(self._entry_id, None)
        await super().async_will_remove_from_hass()

    def apply_options(
        self,
        *,
        threshold: int,
        slot_threshold: int | None,
        expansion_cap: int,
        denylist: list[str] | None,
        include_builtins: bool,
        builtin_allowlist: list[str] | None,
        slot_extraction: bool,
        fallback_agent_id: str,
    ) -> None:
        self._threshold = threshold
        self._slot_threshold = slot_threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
        self._builtin_allowlist = set(builtin_allowlist) if builtin_allowlist else None
        self._slot_extraction = slot_extraction
        self._fallback_agent_id = fallback_agent_id
        # Anything affecting candidate composition invalidates the pools.
        self._pools.clear()
        self._builtin_overrides.clear()
        self._builtin_intents_cache.clear()

    @callback
    def _on_registry_event(self, _event) -> None:
        if self._rebuild_handle is not None:
            self._rebuild_handle()  # cancels the pending call
        self._rebuild_handle = async_call_later(
            self.hass, _REGISTRY_REBUILD_DEBOUNCE_S, self._do_debounced_rebuild
        )

    async def _do_debounced_rebuild(self, _now) -> None:
        self._rebuild_handle = None
        languages = list(self._pools.keys())
        self._pools.clear()
        self._builtin_overrides.clear()
        self._builtin_intents_cache.clear()
        for lang in languages:
            try:
                await self._async_get_pool(lang)
            except Exception:  # pragma: no cover
                _LOGGER.exception("[%s] pool rebuild failed", lang)

    async def _async_get_builtin_override(
        self, language: str, resolver: Resolver
    ) -> list[Candidate]:
        """
        Lazily build (and cache) builtin candidates for parse-time override.

        Uses the per-language ``_builtin_intents_cache`` populated by
        ``_async_get_pool``, so we don't re-pull HA's intents_dict here.
        """
        cached = self._builtin_overrides.get(language)
        if cached is not None:
            return cached
        lock = self._builtin_override_locks.setdefault(language, asyncio.Lock())
        async with lock:
            cached = self._builtin_overrides.get(language)
            if cached is not None:
                return cached
            builtin_intents = self._builtin_intents_cache.get(language, {})
            builtins = await self.hass.async_add_executor_job(
                self._expand_intents, builtin_intents, resolver
            )
            self._builtin_overrides[language] = builtins
            return builtins

    async def _async_get_pool(
        self, language: str
    ) -> tuple[Resolver, list[Candidate], list[Candidate]]:
        cached = self._pools.get(language)
        if cached is not None:
            return cached
        lock = self._pool_locks.setdefault(language, asyncio.Lock())
        async with lock:
            cached = self._pools.get(language)
            if cached is not None:
                return cached
            (
                ha_slot_lists,
                ha_expansion_rules,
                user_intents,
                builtin_intents,
            ) = await self._async_collect_ha_intents_data(language)
            for k, v in (await self._async_collect_default_agent_dynamic_slot_lists()).items():
                ha_slot_lists.setdefault(k, v)
            pool = await self.hass.async_add_executor_job(
                self._build_pool,
                language,
                ha_slot_lists,
                ha_expansion_rules,
                user_intents,
                builtin_intents,
            )
            self._pools[language] = pool
            self._builtin_intents_cache[language] = builtin_intents
            return pool

    def _find_default_agent(self):
        get_agent = getattr(conversation, "async_get_agent", None)
        if get_agent is None:  # pragma: no cover - test stub
            return None
        return get_agent(self.hass, "conversation.home_assistant")

    async def _async_collect_default_agent_dynamic_slot_lists(self) -> dict[str, list[str]]:
        """
        Pull dynamically built ``name``/``area``/``floor`` slot lists from HA's default agent.

        Those lists are declared ``wildcard: true`` in ``intents_dict``.
        The real values are computed at recognition time normally.
        Calling ``_make_slot_lists`` ourselves forces this.
        """
        out: dict[str, list[str]] = {}
        agent = self._find_default_agent()
        if agent is None:
            return out

        try:
            slot_lists = await agent._make_slot_lists()
        except Exception:
            _LOGGER.exception("default agent _make_slot_lists() raised")
            return out

        for name, slot_list in (slot_lists or {}).items():
            values = _extract_text_slot_values(slot_list)
            if values:
                out[name] = sorted(set(values))

        if out:
            _LOGGER.info(
                "pulled %d dynamic slot list(s) from default agent: %s",
                len(out),
                {k: len(v) for k, v in out.items()},
            )
        return out

    async def _async_collect_ha_intents_data(
        self, language: str
    ) -> tuple[
        dict[str, list[str]],
        dict[str, str],
        dict[str, list[str]],
        dict[str, list[str]],
    ]:
        """
        Pull slot lists, expansion rules, and intents from HA's default agent, and
        the ``conversation:`` YAML config block stash.

        User vs builtin intents are split by name membership in the language pack.
        """
        slot_lists: dict[str, list[str]] = {}
        rules: dict[str, str] = {}
        user_intents: dict[str, list[str]] = {}
        builtin_intents: dict[str, list[str]] = {}
        denylist = self._denylist or set()

        # (1) Stash from `conversation:` YAML block.
        stash = self.hass.data.get(DOMAIN, {})
        for name, raw_def in (stash.get(KEY_CONVERSATION_LISTS) or {}).items():
            values = _parse_raw_list_values(raw_def)
            if values:
                slot_lists[name] = values
        for name, body in (stash.get(KEY_CONVERSATION_EXPANSION_RULES) or {}).items():
            if isinstance(body, str) and body:
                rules[name] = body
        for name, patterns in (stash.get(KEY_CONVERSATION_INTENTS) or {}).items():
            if name in denylist:
                continue
            if isinstance(patterns, str):
                user_intents[name] = [patterns]
            elif patterns:
                user_intents[name] = list(patterns)

        # (2) HA default agent intents_dict.
        agent = self._find_default_agent()
        if agent is None:
            _LOGGER.warning(
                "[%s] HA default conversation agent not available, only stash-defined "
                "vocabulary will be used",
                language,
            )
            return slot_lists, rules, user_intents, builtin_intents

        # Force-load the language pack so _lang_intents[language] exists.
        await agent.async_get_or_load_intents(language)

        lang_intents = agent._lang_intents.get(language)
        if lang_intents is None:
            _LOGGER.debug(
                "[%s] default agent has no intents for this language (available: %s)",
                language,
                list(agent._lang_intents.keys()),
            )
            return slot_lists, rules, user_intents, builtin_intents

        intents_dict = lang_intents.intents_dict or {}
        for name, body in (intents_dict.get("expansion_rules") or {}).items():
            if isinstance(body, str) and body:
                rules[name] = body
        for name, raw_def in (intents_dict.get("lists") or {}).items():
            values = _parse_raw_list_values(raw_def)
            if values:
                slot_lists[name] = values

        builtin_names = await self.hass.async_add_executor_job(self._builtin_intent_names, language)
        for name, payload in (intents_dict.get("intents") or {}).items():
            if name in denylist:
                continue
            sentences: list[str] = []
            for block in payload.get("data") or []:
                sentences.extend(block.get("sentences") or [])
            if not sentences:
                continue
            if name in builtin_names:
                builtin_intents[name] = sentences
            else:
                user_intents[name] = sentences

        _LOGGER.info(
            "[%s] vocabulary: %d slot list(s), %d expansion rule(s), %d user / "
            "%d builtin intent(s)",
            language,
            len(slot_lists),
            len(rules),
            len(user_intents),
            len(builtin_intents),
        )
        return slot_lists, rules, user_intents, builtin_intents

    def _builtin_intent_names(self, language: str) -> set[str]:
        """Names of intents defined by the language pack. Used to classify as user/builtin."""
        try:
            from home_assistant_intents import get_intents  # type: ignore
        except ImportError:
            return set()
        try:
            raw = get_intents(language) or {}
        except Exception:  # pragma: no cover
            return set()
        return set((raw.get("intents") or {}).keys())

    def _build_pool(
        self,
        language: str,
        ha_slot_lists: dict[str, list[str]],
        ha_expansion_rules: dict[str, str],
        user_intents: dict[str, list[str]],
        builtin_intents: dict[str, list[str]],
    ) -> tuple[Resolver, list[Candidate], list[Candidate]]:
        resolver = self._build_resolver(language, ha_slot_lists, ha_expansion_rules)
        resolver.match_threshold = self._threshold
        resolver.slot_resolution_threshold = self._threshold

        user_candidates = self._expand_intents(user_intents, resolver)
        if self._include_builtins:
            selected_builtins = builtin_intents
        elif self._builtin_allowlist:
            selected_builtins = {
                name: patterns
                for name, patterns in builtin_intents.items()
                if name in self._builtin_allowlist
            }
        else:
            selected_builtins = {}
        builtin_candidates: list[Candidate] = self._expand_intents(selected_builtins, resolver)

        _LOGGER.info(
            "[%s] built pool: %d user candidate(s) across %d intent(s), "
            "%d builtin candidate(s) (include_builtins=%s, allowlist=%s)",
            language,
            len(user_candidates),
            len(user_intents),
            len(builtin_candidates),
            self._include_builtins,
            sorted(self._builtin_allowlist) if self._builtin_allowlist else None,
        )
        return (resolver, user_candidates, builtin_candidates)

    def _expand_intents(
        self,
        intents: dict[str, list[str]],
        resolver: Resolver,
    ) -> list[Candidate]:
        """Expand ``intents`` into candidates, round-robin across sentence patterns."""
        candidates: list[Candidate] = []
        for intent_name, patterns in intents.items():
            per_pattern: list[tuple[int, list[tuple[str, str, list[str]]]]] = []
            for idx, pat in enumerate(patterns):
                forms = list(expand_pattern(pat, self._expansion_cap, resolver=resolver))
                if forms:
                    per_pattern.append((idx, forms))
            if not per_pattern:
                continue

            # Dedupe across patterns within this intent.
            seen_texts: set[str] = set()
            kept = 0
            depth = 0
            while kept < PER_INTENT_CANDIDATE_CAP:
                progress = False
                for idx, forms in per_pattern:
                    if depth >= len(forms):
                        continue
                    text, display_text, slot_names = forms[depth]
                    progress = True
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    candidates.append(
                        Candidate(
                            intent=intent_name,
                            pattern_idx=idx,
                            text=text,
                            display_text=display_text,
                            slot_names=slot_names,
                        )
                    )
                    kept += 1
                    if kept >= PER_INTENT_CANDIDATE_CAP:
                        break
                if not progress:
                    break
                depth += 1

            total = sum(len(forms) for _, forms in per_pattern)
            if kept < total:
                _LOGGER.debug(
                    "intent %s hit cap or duplicate ceiling: kept %d of %d expansions (cap=%d)",
                    intent_name,
                    kept,
                    total,
                    PER_INTENT_CANDIDATE_CAP,
                )
        return candidates

    def _build_resolver(
        self,
        language: str,
        ha_slot_lists: dict[str, list[str]],
        ha_expansion_rules: dict[str, str],
    ) -> Resolver:
        """Build a Resolver from HA's authoritative vocabulary."""
        resolver = Resolver()
        for name, values in ha_slot_lists.items():
            if values:
                resolver.slot_values[name] = values
        for name, body in ha_expansion_rules.items():
            if body:
                resolver.expansion_rules[name] = [body]

        _LOGGER.debug(
            "[%s] resolver assembled: %d expansion rules, %d slot lists",
            language,
            len(resolver.expansion_rules),
            len(resolver.slot_values),
        )
        return resolver

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        language = user_input.language or self.hass.config.language or "en"
        try:
            resolver, user_candidates, builtin_candidates = await self._async_get_pool(language)
        except Exception:  # pragma: no cover
            _LOGGER.exception("[%s] failed to build pool", language)
            resolver, user_candidates, builtin_candidates = Resolver(), [], []

        forwarded_text = user_input.text
        t0 = time.perf_counter()
        try:
            detail, _ = self._match_in_pools(
                user_input.text, resolver, user_candidates, builtin_candidates
            )
        except Exception:  # pragma: no cover
            _LOGGER.exception("match: unexpected error for %r", user_input.text)
            detail = None
        match_ms = (time.perf_counter() - t0) * 1000.0

        if detail is None:
            _LOGGER.debug(
                "match: no candidate above threshold %d for %r, passthrough [%.1fms]",
                self._threshold,
                user_input.text,
                match_ms,
            )
        else:
            candidate, captured, score_value, canonical = detail
            forwarded_text = canonical
            _LOGGER.info(
                "match: %r -> %s (score=%d, captured=%s) [%.1fms], forwarding %r to hassil",
                user_input.text,
                candidate.intent,
                score_value,
                captured,
                match_ms,
                canonical,
            )

        hassil_result = None
        try:
            hassil_result = await conversation.async_converse(
                hass=self.hass,
                text=forwarded_text,
                conversation_id=user_input.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=_HASSIL_AGENT_ID,
            )
        except Exception:
            _LOGGER.exception("forward to hassil failed for %r", forwarded_text)

        if hassil_result is not None and not _is_error_result(hassil_result):
            return hassil_result

        if self._fallback_agent_id == _HASSIL_AGENT_ID:
            return hassil_result if hassil_result is not None else _no_match(user_input)

        try:
            return await conversation.async_converse(
                hass=self.hass,
                text=user_input.text,
                conversation_id=user_input.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id=self._fallback_agent_id,
            )
        except Exception:
            _LOGGER.exception("fallback agent %s failed", self._fallback_agent_id)
            return hassil_result if hassil_result is not None else _no_match(user_input)

    def _match(
        self,
        text: str,
        resolver: Resolver,
        candidates: list[Candidate],
    ) -> tuple[Candidate, list[str], int, str] | None:
        """Match ``text`` against ``candidates`` and resolve slots.

        Returns ``(candidate, captured, score, canonical)`` or ``None``. When
        the top-scoring candidate is slot-bearing but its slots fail to
        extract, falls back to the highest-scoring same-intent sibling whose
        slots do extract. With ``slot_extraction=False`` slot-bearing matches
        are skipped (passthrough).
        """
        resolver.match_threshold = self._threshold
        resolver.slot_resolution_threshold = self._effective_slot_threshold
        match = find_best(text, candidates, resolver)
        if match is None:
            return None
        candidate, score_value = match

        if candidate.has_slots:
            if not self._slot_extraction:
                return None
            captured = extract_slots(text, candidate)
            if captured is None:
                sibling = self._best_extractable_sibling(text, candidate, candidates, resolver)
                if sibling is None:
                    return None
                candidate, captured, score_value = sibling
        else:
            captured = []

        canonical = build_canonical(candidate, captured, resolver=resolver)
        return (candidate, captured, score_value, canonical)

    def _match_in_pools(
        self,
        text: str,
        resolver: Resolver,
        user_candidates: list[Candidate],
        builtin_candidates: list[Candidate],
    ) -> tuple[tuple[Candidate, list[str], int, str] | None, str | None]:
        """Match against user + builtin pools as one. Records ``pool_used`` for diagnostics."""
        combined = user_candidates + builtin_candidates
        if not combined:
            return None, None
        detail = self._match(text, resolver, combined)
        if detail is None:
            return None, None
        winner = detail[0]
        in_user_pool = any(c is winner for c in user_candidates)
        return detail, "user" if in_user_pool else "builtin"

    def _best_extractable_sibling(
        self,
        user_text: str,
        skip: Candidate,
        candidates: list[Candidate],
        resolver: Resolver,
    ) -> tuple[Candidate, list[str], int] | None:
        """Highest-scoring same-intent slot-bearing sibling whose slots extract."""
        scored = sorted(
            (
                (score(user_text, c.text, resolver), c)
                for c in candidates
                if c is not skip and c.intent == skip.intent and c.has_slots
            ),
            key=lambda sc: -sc[0],
        )
        for s, c in scored:
            if s < resolver.match_threshold:
                break
            captured = extract_slots(user_text, c)
            if captured is not None:
                return (c, captured, s)
        return None

    async def parse_sentence(
        self,
        language: str,
        sentence: str,
        run_official: bool = False,
        include_builtins: bool = False,
        debug_top_candidates: bool = False,
    ) -> dict:
        """
        Diagnostic: run the closest-intent matcher (and hassil) on ``sentence``.

        ``include_builtins=True`` forces builtin intents into the candidate
        pool for this call even if the integration is configured without them.
        ``debug_top_candidates=True`` includes the top-10 raw-scored candidates
        in the response, regardless of threshold or tie-break ordering.
        """
        try:
            resolver, user_candidates, builtin_candidates = await self._async_get_pool(language)
        except Exception:
            _LOGGER.exception("parse[%s]: pool build failed", language)
            return {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "error": f"failed to build pool for language {language!r}",
            }

        if include_builtins and not builtin_candidates:
            try:
                builtin_candidates = await self._async_get_builtin_override(language, resolver)
            except Exception:
                _LOGGER.exception("parse[%s]: builtin override build failed", language)

        t0 = time.perf_counter()
        detail, pool_used = self._match_in_pools(
            sentence, resolver, user_candidates, builtin_candidates
        )
        match_ms = (time.perf_counter() - t0) * 1000.0

        pools_info = {
            "include_builtins_requested": include_builtins,
            "user_candidate_count": len(user_candidates),
            "builtin_candidate_count": len(builtin_candidates),
            "match_ms": round(match_ms, 2),
        }

        top_scored: list[dict] | None = None
        if debug_top_candidates:
            scored_for_debug = sorted(
                (
                    (c, score(sentence, c.text, resolver))
                    for c in (user_candidates + builtin_candidates)
                ),
                key=lambda cs: -cs[1],
            )
            top_scored = [
                {
                    "intent": c.intent,
                    "text": c.text,
                    "score": s,
                    "slot_names": c.slot_names,
                    "fixed_text_len": len(
                        re.sub(r"\s+", " ", c.text.replace(SLOT_WILDCARD, " ")).strip()
                    ),
                }
                for (c, s) in scored_for_debug[:10]
            ]

        if detail is None:
            # Surface common slot lists so the user can immediately see whether
            # entity exposure is wired up (slot_values['name'] populated).
            common_slot_summary = {
                key: {
                    "size": len(resolver.slot_values.get(key, [])),
                    "sample": resolver.slot_values.get(key, [])[:25],
                }
                for key in ("name", "area", "floor")
            }
            result: dict = {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "matched": False,
                "canonical": None,
                "pools": pools_info,
                "slot_lists_overview": common_slot_summary,
            }
            if top_scored is not None:
                result["top_scored_candidates"] = top_scored
        else:
            candidate, captured, score_value, canonical = detail
            slot_map: dict[str, str] = (
                dict(zip(candidate.slot_names, captured, strict=False))
                if candidate.slot_names
                else {}
            )
            slot_resolution: dict[str, dict] = {}
            for slot_name, raw_value in slot_map.items():
                values = resolver.slot_values.get(slot_name) or []
                resolved = resolver.resolve_slot(raw_value, slot_name)
                slot_resolution[slot_name] = {
                    "captured": raw_value,
                    "resolved": resolved,
                    "changed": resolved != raw_value,
                    "list_size": len(values),
                    "list_values_sample": values[:25],
                    "list_truncated": len(values) > 25,
                    "threshold": resolver.slot_resolution_threshold,
                }
            result = {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "matched": True,
                "intent": candidate.intent,
                "score": score_value,
                "matched_pattern": candidate.text,
                "captured_slots": slot_map,
                "slot_resolution": slot_resolution,
                "canonical": canonical,
                "pool": pool_used,
                "pools": pools_info,
            }
            if top_scored is not None:
                result["top_scored_candidates"] = top_scored

        if run_official:
            text_for_recognize = result["canonical"] or sentence
            try:
                official = await self._official_recognize(language, text_for_recognize)
            except Exception as exc:
                _LOGGER.exception("parse: official recognize raised")
                official = {
                    "available": False,
                    "reason": f"could not connect to hassil ({type(exc).__name__}: {exc})",
                }
            result["official"] = official
        return result

    async def _official_recognize(self, language: str, text: str) -> dict:
        """Route ``text`` through HA's default conversation agent in parse-only mode."""
        try:
            from homeassistant.components.conversation import (  # type: ignore
                ConversationInput,
                async_get_agent,
            )
            from homeassistant.core import Context  # type: ignore
        except ImportError:
            return {"available": False, "reason": "conversation API import failed"}

        agent = async_get_agent(self.hass, None)
        if agent is None:
            return {"available": False, "reason": "default conversation agent not available"}

        debug = getattr(agent, "async_debug_recognize", None)
        if debug is None:
            return {
                "available": False,
                "reason": "default agent has no async_debug_recognize "
                "(Home Assistant version may be too old)",
            }

        try:
            user_input = ConversationInput(
                text=text,
                context=Context(),
                conversation_id=None,
                device_id=None,
                satellite_id=None,
                language=language,
                agent_id="conversation.home_assistant",
            )
        except TypeError:
            try:
                user_input = ConversationInput(  # type: ignore[call-arg]
                    text=text,
                    context=Context(),
                    conversation_id=None,
                    device_id=None,
                    language=language,
                    agent_id="conversation.home_assistant",
                )
            except Exception as exc:
                return {"available": False, "reason": f"ConversationInput build failed: {exc}"}

        try:
            outcome = await debug(user_input)
        except Exception as exc:
            _LOGGER.exception("parse: default agent debug recognize raised")
            return {"available": True, "input": text, "matched": False, "error": str(exc)}

        if outcome is None:
            return {"available": True, "input": text, "matched": False}

        return {"available": True, "input": text, **outcome}

    def _exposure_diagnostic(self) -> dict:
        """Per-state exposure check, so users can see why slot_values['name']
        ends up empty even when entities look exposed in the UI."""
        try:
            states = list(self.hass.states.async_all())
        except Exception:
            return {"error": "hass.states.async_all() raised"}

        api_path = _EXPOSE_API_PATH

        exposed: list[dict] = []
        not_exposed: list[dict] = []
        for state in states:
            entry = {
                "entity_id": state.entity_id,
                "friendly_name": state.attributes.get("friendly_name") or state.name,
            }
            try:
                ok = _is_exposed(self.hass, state.entity_id)
            except Exception as exc:
                entry["error"] = repr(exc)
                not_exposed.append(entry)
                continue
            (exposed if ok else not_exposed).append(entry)

        return {
            "exposure_api_used": api_path or "<none found, defaulting to expose-all>",
            "total_states": len(states),
            "exposed_count": len(exposed),
            "not_exposed_count": len(not_exposed),
            "exposed_sample": exposed[:25],
            "not_exposed_sample": not_exposed[:25],
        }

    def dump_state(
        self,
        builtin_overrides: dict[str, list[Candidate]] | None = None,
        intent_filter: str | None = None,
        include_exposure: bool = False,
    ) -> dict:
        """Return a plain-data snapshot of pools for the diagnostic service."""
        needle = intent_filter.lower() if intent_filter else None

        def _keep(intent_name: str) -> bool:
            return needle is None or needle in intent_name.lower()

        out: dict = {
            "version": VERSION,
            "entry_id": self._entry_id,
            "threshold": self._threshold,
            "slot_threshold": self._slot_threshold,
            "effective_slot_threshold": self._effective_slot_threshold,
            "expansion_cap": self._expansion_cap,
            "include_builtins": self._include_builtins,
            "builtin_allowlist": sorted(self._builtin_allowlist)
            if self._builtin_allowlist
            else None,
            "slot_extraction": self._slot_extraction,
            "fallback_agent_id": self._fallback_agent_id,
            "denylist": sorted(self._denylist) if self._denylist else None,
            "languages": {},
        }
        if intent_filter is not None:
            out["intent_filter"] = intent_filter
        if include_exposure:
            out["exposure"] = self._exposure_diagnostic()
        for lang, (resolver, user_candidates, builtin_candidates) in self._pools.items():
            effective_builtins = builtin_overrides.get(lang) if builtin_overrides else None
            if effective_builtins is None:
                effective_builtins = builtin_candidates

            user_by_intent: dict[str, list[str]] = {}
            for c in user_candidates:
                if _keep(c.intent):
                    user_by_intent.setdefault(c.intent, []).append(c.text)
            builtin_by_intent: dict[str, list[str]] = {}
            for c in effective_builtins:
                if _keep(c.intent):
                    builtin_by_intent.setdefault(c.intent, []).append(c.text)
            out["languages"][lang] = {
                "user_candidate_count": len(user_candidates),
                "builtin_candidate_count": len(effective_builtins),
                "user_intents": user_by_intent,
                "builtin_intents": builtin_by_intent,
                "expansion_rules": {k: v for k, v in resolver.expansion_rules.items()},
                "slot_values": {k: v for k, v in resolver.slot_values.items()},
            }
        return out


def _is_error_result(result: conversation.ConversationResult) -> bool:
    """Did the agent return a recognizable failure response?"""
    response = getattr(result, "response", None)
    return getattr(response, "error_code", None) is not None


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


def _parse_raw_list_values(raw_def) -> list[str]:
    """
    Pull plain string values from the raw YAML form of a Hassil slot list.

    Handles the two enumerable shapes:
      - ``values: [str, ..., {in: ..., out: ...}, ...]``
      - ``range: {from: int, to: int, step: int}``
    ``wildcard: true`` yields nothing (can't enumerate).
    """
    if not isinstance(raw_def, dict):
        return []
    values_field = raw_def.get("values")
    if isinstance(values_field, list):
        out: list[str] = []
        for v in values_field:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                in_val = v.get("in") or v.get("out")
                if isinstance(in_val, str):
                    out.append(in_val)
        if out:
            return sorted(set(out))
        return []
    rng = raw_def.get("range")
    if isinstance(rng, dict):
        from_v = rng.get("from", 0)
        to_v = rng.get("to", 0)
        step = rng.get("step", 1) or 1
        if (
            isinstance(from_v, int)
            and isinstance(to_v, int)
            and isinstance(step, int)
            and step > 0
            and from_v <= to_v
        ):
            return [str(i) for i in range(from_v, to_v + 1, step)]
    return []


def _extract_text_slot_values(slot_list) -> list[str]:
    """
    Pull plain string values out of a hassil ``TextSlotList``.

    Each ``TextSlotValue.text_in`` is a ``Sentence`` whose ``.text`` holds
    the original input string. ``.value_out`` is used as a fallback for
    values that were constructed without preserving the raw text.
    """
    values: list[str] = []
    for v in getattr(slot_list, "values", None) or []:
        text_in = getattr(v, "text_in", None)
        text = getattr(text_in, "text", None) if text_in is not None else None
        if isinstance(text, str) and text:
            values.append(text)
            continue
        value_out = getattr(v, "value_out", None)
        if isinstance(value_out, str) and value_out:
            values.append(value_out)
    return values


_EXPOSE_ASSISTANT = "conversation"
_EXPOSE_API_PATH: str | None
try:
    from homeassistant.components.homeassistant.exposed_entities import (  # type: ignore
        async_should_expose as _async_should_expose,
    )

    _EXPOSE_API_PATH = "homeassistant.components.homeassistant.exposed_entities"
except ImportError:  # pragma: no cover -- only hit by the test stub
    _async_should_expose = None  # type: ignore[assignment]
    _EXPOSE_API_PATH = None


def _is_exposed(hass: HomeAssistant, entity_id: str) -> bool:
    """Check whether ``entity_id`` is voice-exposed to the conversation assistant."""
    if _async_should_expose is None:
        return True
    return bool(_async_should_expose(hass, _EXPOSE_ASSISTANT, entity_id))
