"""
Closest-intent conversation entity.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import intent as intent_helper
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later

# Importable both as part of the package and as a standalone module for tests.
try:
    from .const import (
        CONF_DENYLIST,
        CONF_EXPANSION_CAP,
        CONF_FALLBACK_AGENT,
        CONF_INCLUDE_BUILTINS,
        CONF_SLOT_EXTRACTION,
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
        CONF_DENYLIST,
        CONF_EXPANSION_CAP,
        CONF_FALLBACK_AGENT,
        CONF_INCLUDE_BUILTINS,
        CONF_SLOT_EXTRACTION,
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
        expansion_cap=opt(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
        denylist=opt(CONF_DENYLIST, None),
        include_builtins=opt(CONF_INCLUDE_BUILTINS, DEFAULT_INCLUDE_BUILTINS),
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
        expansion_cap=opt(CONF_EXPANSION_CAP, DEFAULT_EXPANSION_CAP),
        denylist=opt(CONF_DENYLIST, None),
        include_builtins=opt(CONF_INCLUDE_BUILTINS, DEFAULT_INCLUDE_BUILTINS),
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
        expansion_cap: int,
        denylist: list[str] | None,
        include_builtins: bool,
        slot_extraction: bool,
        fallback_agent_id: str,
        entry_id: str,
    ) -> None:
        self.hass = hass
        self._threshold = threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
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
        self._builtin_overrides: dict[str, list[Candidate]] = {}
        self._builtin_override_locks: dict[str, asyncio.Lock] = {}
        self._rebuild_handle = None  # async_call_later cancel handle
        self._unsub_listeners: list = []

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
        expansion_cap: int,
        denylist: list[str] | None,
        include_builtins: bool,
        slot_extraction: bool,
        fallback_agent_id: str,
    ) -> None:
        self._threshold = threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
        self._slot_extraction = slot_extraction
        self._fallback_agent_id = fallback_agent_id
        # Anything affecting candidate composition invalidates the pools.
        self._pools.clear()
        self._builtin_overrides.clear()

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
        for lang in languages:
            try:
                await self._async_get_pool(lang)
            except Exception:  # pragma: no cover
                _LOGGER.exception("closest_intent: rebuild for %s failed", lang)

    async def _async_get_builtin_override(
        self, language: str, resolver: Resolver, user_intent_names: set[str]
    ) -> list[Candidate]:
        """Lazily build (and cache) builtin candidates for parse-time override."""
        cached = self._builtin_overrides.get(language)
        if cached is not None:
            return cached
        lock = self._builtin_override_locks.setdefault(language, asyncio.Lock())
        async with lock:
            cached = self._builtin_overrides.get(language)
            if cached is not None:
                return cached
            builtins = await self.hass.async_add_executor_job(
                self._build_builtin_candidates, language, resolver, user_intent_names
            )
            self._builtin_overrides[language] = builtins
            return builtins

    def _build_builtin_candidates(
        self, language: str, resolver: Resolver, exclude: set[str]
    ) -> list[Candidate]:
        builtin_intents = self._gather_builtin_intents(language, exclude=exclude)
        return self._expand_intents(builtin_intents, resolver)

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
            pool = await self.hass.async_add_executor_job(self._build_pool, language)
            self._pools[language] = pool
            return pool

    def _build_pool(self, language: str) -> tuple[Resolver, list[Candidate], list[Candidate]]:
        custom_docs = self._load_custom_sentences(language)
        resolver = self._build_resolver(language, custom_docs)

        user_intents = self._gather_user_intents(custom_docs)
        user_candidates = self._expand_intents(user_intents, resolver)

        builtin_candidates: list[Candidate] = []
        if self._include_builtins:
            builtin_intents = self._gather_builtin_intents(language, exclude=set(user_intents))
            builtin_candidates = self._expand_intents(builtin_intents, resolver)

        _LOGGER.info(
            "closest_intent[%s]: built %d user candidate(s) across %d intent(s); "
            "%d builtin candidate(s) (builtins=%s)",
            language,
            len(user_candidates),
            len(user_intents),
            len(builtin_candidates),
            self._include_builtins,
        )
        return (resolver, user_candidates, builtin_candidates)

    def _expand_intents(self, intents: dict[str, list[str]], resolver: Resolver) -> list[Candidate]:
        candidates: list[Candidate] = []
        for intent_name, patterns in intents.items():
            kept = 0
            for idx, pat in enumerate(patterns):
                if kept >= PER_INTENT_CANDIDATE_CAP:
                    break
                for text, slot_names in expand_pattern(pat, self._expansion_cap, resolver=resolver):
                    candidates.append(
                        Candidate(
                            intent=intent_name,
                            pattern_idx=idx,
                            text=text,
                            slot_names=slot_names,
                        )
                    )
                    kept += 1
                    if kept >= PER_INTENT_CANDIDATE_CAP:
                        _LOGGER.debug(
                            "closest_intent: %s hit per-intent cap (%d), truncating",
                            intent_name,
                            PER_INTENT_CANDIDATE_CAP,
                        )
                        break
        return candidates

    def _load_custom_sentences(self, language: str) -> list[dict]:
        """
        Walk ``<configDir>/custom_sentences/<language>/*.yaml`` and load each.

        Files written by ``hass.voice.custom_sentences`` (or hand-placed by the user)
        live there and contain Hassil-format ``intents`` / ``lists`` / ``expansion_rules``.
        We merge their contents into both the resolver and the candidate pool so closest_intent
        sees the same vocabulary HA's default agent does.
        """
        import os

        try:
            import yaml  # type: ignore
        except ImportError:
            _LOGGER.warning("PyYAML not importable; skipping custom_sentences")
            return []

        base = self.hass.config.path("custom_sentences", language)
        if not os.path.isdir(base):
            return []

        docs: list[dict] = []
        for fname in sorted(os.listdir(base)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(base, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    doc = yaml.safe_load(f)
            except Exception:
                _LOGGER.warning("closest_intent: failed to load %s", path, exc_info=True)
                continue
            if isinstance(doc, dict):
                docs.append(doc)
        return docs

    def _build_resolver(self, language: str, custom_docs: list[dict]) -> Resolver:
        """
        Pre-compute expansion-rule expansions + slot-list values.

        Two sources, merged into one ``Resolver``:

        - **Hassil's static data** (``home_assistant_intents`` package):
          Per-language expansion rules and static slot lists.
          Sampled via ``hassil.sample`` to flatten alternations and
          nested rules into surface forms.
        - **HA's runtime registries**:
          Areas, floors, and exposed entity friendly names.
          Populated by HA's default agent at recognition time,
          but we read the same registries directly so we can use the same vocabulary.
        """
        resolver = Resolver()

        try:
            from hassil.intents import (  # type: ignore
                Intents,
                RangeSlotList,
                TextSlotList,
            )
            from hassil.sample import sample_expression  # type: ignore
            from home_assistant_intents import get_intents  # type: ignore

            hassil_available = True
        except ImportError:
            _LOGGER.debug("hassil/home_assistant_intents not importable; skipping language pack")
            hassil_available = False
            get_intents = None  # type: ignore
            Intents = None  # type: ignore
            RangeSlotList = None  # type: ignore
            TextSlotList = None  # type: ignore
            sample_expression = None  # type: ignore

        raw = None
        if hassil_available:
            try:
                raw = get_intents(language)  # type: ignore[misc]
            except Exception:  # pragma: no cover
                raw = None

        stash = self.hass.data.get(DOMAIN, {})
        user_lists = dict(stash.get(KEY_CONVERSATION_LISTS) or {})
        user_rules = dict(stash.get(KEY_CONVERSATION_EXPANSION_RULES) or {})
        for doc in custom_docs:
            for k, v in (doc.get("lists") or {}).items():
                user_lists[k] = v
            for k, v in (doc.get("expansion_rules") or {}).items():
                user_rules[k] = v
        if user_lists or user_rules:
            raw = dict(raw or {})
            raw.setdefault("lists", {})
            raw["lists"].update(user_lists)
            raw.setdefault("expansion_rules", {})
            raw["expansion_rules"].update(user_rules)
            raw.setdefault("intents", {})
            raw.setdefault("language", language)

        if hassil_available and raw:
            try:
                intents = Intents.from_dict(raw)  # type: ignore[union-attr]
            except Exception:  # pragma: no cover
                _LOGGER.exception("closest_intent: failed to parse intents for %s", language)
                intents = None

            if intents is not None:
                # Expansion rules -> list of surface forms.
                for name, rule in (intents.expansion_rules or {}).items():
                    try:
                        forms = list(_dedupe(sample_expression(rule.expression, intents)))
                    except Exception:
                        continue
                    # Cap rule expansion to keep alternation explosions bounded.
                    resolver.expansion_rules[name] = forms[: max(self._expansion_cap, 32)]

                # Slot lists -> list of acceptable values.
                for name, lst in (intents.slot_lists or {}).items():
                    values = _slot_list_values(
                        lst, intents, sample_expression, TextSlotList, RangeSlotList
                    )
                    if values:
                        resolver.slot_values[name] = values

        try:
            from homeassistant.helpers import (
                area_registry as ar,
            )
            from homeassistant.helpers import (
                entity_registry as er,
            )
            from homeassistant.helpers import (
                floor_registry as fr,
            )
        except ImportError:
            return resolver

        try:
            areas = ar.async_get(self.hass)
            area_names: list[str] = []
            for area in areas.async_list_areas():
                area_names.append(area.name)
                if area.aliases:
                    area_names.extend(area.aliases)
            if area_names:
                resolver.slot_values["area"] = sorted(set(area_names))
        except Exception:
            _LOGGER.debug("closest_intent: failed to read area registry", exc_info=True)

        try:
            floors = fr.async_get(self.hass)
            floor_names: list[str] = [f.name for f in floors.async_list_floors()]
            if floor_names:
                resolver.slot_values["floor"] = sorted(set(floor_names))
        except Exception:
            _LOGGER.debug("closest_intent: failed to read floor registry", exc_info=True)

        try:
            ent_reg = er.async_get(self.hass)
            names: list[str] = []
            for entity in ent_reg.entities.values():
                if not _is_exposed(self.hass, entity.entity_id):
                    continue
                state = self.hass.states.get(entity.entity_id)
                if state is not None:
                    fname = state.attributes.get("friendly_name") or state.name
                    if fname:
                        names.append(fname)
                if entity.aliases:
                    names.extend(entity.aliases)
            if names:
                resolver.slot_values["name"] = sorted(set(names))
        except Exception:
            _LOGGER.debug("closest_intent: failed to read entity registry", exc_info=True)

        _LOGGER.debug(
            "closest_intent[%s]: resolver has %d expansion rules, %d slot lists",
            language,
            len(resolver.expansion_rules),
            len(resolver.slot_values),
        )
        return resolver

    def _gather_user_intents(self, custom_docs: list[dict]) -> dict[str, list[str]]:
        gathered: dict[str, list[str]] = {}

        conv_intents = self.hass.data.get(DOMAIN, {}).get(KEY_CONVERSATION_INTENTS, {})
        for name, patterns in conv_intents.items():
            if isinstance(patterns, str):
                gathered[name] = [patterns]
            else:
                gathered[name] = list(patterns)

        for doc in custom_docs:
            for name, payload in (doc.get("intents") or {}).items():
                sentences: list[str] = []
                for block in payload.get("data") or []:
                    sentences.extend(block.get("sentences") or [])
                if sentences:
                    gathered.setdefault(name, []).extend(sentences)

        return self._apply_denylist(gathered)

    def _gather_builtin_intents(self, language: str, *, exclude: set[str]) -> dict[str, list[str]]:
        gathered: dict[str, list[str]] = {}
        try:
            from home_assistant_intents import get_intents  # type: ignore

            builtin = get_intents(language) or {}
        except Exception:  # pragma: no cover
            _LOGGER.warning(
                "closest_intent: include_builtins=true but home_assistant_intents "
                "is unavailable; skipping built-in patterns"
            )
            return gathered

        for name, payload in (builtin.get("intents") or {}).items():
            if name in exclude:
                continue
            sentences: list[str] = []
            for block in payload.get("data") or []:
                sentences.extend(block.get("sentences") or [])
            if sentences:
                gathered[name] = sentences

        return self._apply_denylist(gathered)

    def _apply_denylist(self, intents: dict[str, list[str]]) -> dict[str, list[str]]:
        if self._denylist is None:
            return intents
        return {k: v for k, v in intents.items() if k not in self._denylist}

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        language = user_input.language or self.hass.config.language or "en"
        try:
            resolver, user_candidates, builtin_candidates = await self._async_get_pool(language)
        except Exception:  # pragma: no cover
            _LOGGER.exception("closest_intent: failed to build pool for language %s", language)
            resolver, user_candidates, builtin_candidates = Resolver(), [], []

        forwarded_text = user_input.text
        try:
            detail, _ = self._match_in_pools(
                user_input.text, resolver, user_candidates, builtin_candidates
            )
        except Exception:  # pragma: no cover
            _LOGGER.exception("closest_intent: unexpected error matching %r", user_input.text)
            detail = None

        if detail is None:
            _LOGGER.debug(
                "closest_intent: no match for %r above %d, passthrough",
                user_input.text,
                self._threshold,
            )
        else:
            candidate, captured, score_value, canonical = detail
            forwarded_text = canonical
            _LOGGER.info(
                "closest_intent: %r -> %s (score=%d, captured=%s) -> forwarding %r to hassil",
                user_input.text,
                candidate.intent,
                score_value,
                captured,
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
            _LOGGER.exception("closest_intent: hassil forwarding failed for %r", forwarded_text)

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
            _LOGGER.exception("closest_intent: fallback agent %s failed", self._fallback_agent_id)
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
        match = find_best(text, candidates, self._threshold)
        if match is None:
            return None
        candidate, score_value = match

        if candidate.has_slots:
            if not self._slot_extraction:
                return None
            captured = extract_slots(text, candidate)
            if captured is None:
                sibling = self._best_extractable_sibling(text, candidate, candidates)
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
        """Try the user pool, fall back to builtins. Returns ``(detail, pool_used)``."""
        detail = self._match(text, resolver, user_candidates)
        if detail is not None:
            return detail, "user"
        if builtin_candidates:
            detail = self._match(text, resolver, builtin_candidates)
            if detail is not None:
                return detail, "builtin"
        return None, None

    def _best_extractable_sibling(
        self,
        user_text: str,
        skip: Candidate,
        candidates: list[Candidate],
    ) -> tuple[Candidate, list[str], int] | None:
        """Highest-scoring same-intent slot-bearing sibling whose slots extract."""
        scored = sorted(
            (
                (score(user_text, c.text), c)
                for c in candidates
                if c is not skip and c.intent == skip.intent and c.has_slots
            ),
            key=lambda sc: -sc[0],
        )
        for s, c in scored:
            if s < self._threshold:
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
    ) -> dict:
        """
        Diagnostic: run the closest-intent matcher (and hassil) on ``sentence``.

        ``include_builtins=True`` forces builtin intents into the candidate
        pool for this call even if the integration is configured without them.
        """
        try:
            resolver, user_candidates, builtin_candidates = await self._async_get_pool(language)
        except Exception:
            _LOGGER.exception("closest_intent.parse: pool build failed for %s", language)
            return {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "error": f"failed to build pool for language {language!r}",
            }

        if include_builtins and not builtin_candidates:
            try:
                builtin_candidates = await self._async_get_builtin_override(
                    language, resolver, {c.intent for c in user_candidates}
                )
            except Exception:
                _LOGGER.exception(
                    "closest_intent.parse: builtin override build failed for %s", language
                )

        detail, pool_used = self._match_in_pools(
            sentence, resolver, user_candidates, builtin_candidates
        )

        if detail is None:
            result: dict = {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "matched": False,
                "canonical": None,
            }
        else:
            candidate, captured, score_value, canonical = detail
            slot_map: dict[str, str] = (
                dict(zip(candidate.slot_names, captured, strict=False))
                if candidate.slot_names
                else {}
            )
            result = {
                "version": VERSION,
                "language": language,
                "input": sentence,
                "matched": True,
                "intent": candidate.intent,
                "score": score_value,
                "matched_pattern": candidate.text,
                "captured_slots": slot_map,
                "canonical": canonical,
                "pool": pool_used,
            }

        if run_official:
            text_for_recognize = result["canonical"] or sentence
            try:
                official = await self._official_recognize(language, text_for_recognize)
            except Exception as exc:
                _LOGGER.exception("closest_intent.parse: official recognize blew up")
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
            _LOGGER.exception("closest_intent.parse: default agent debug recognize raised")
            return {"available": True, "input": text, "matched": False, "error": str(exc)}

        if outcome is None:
            return {"available": True, "input": text, "matched": False}

        return {"available": True, "input": text, **outcome}

    def dump_state(self) -> dict:
        """Return a plain-data snapshot of pools for the diagnostic service."""
        out: dict = {
            "version": VERSION,
            "entry_id": self._entry_id,
            "threshold": self._threshold,
            "expansion_cap": self._expansion_cap,
            "include_builtins": self._include_builtins,
            "slot_extraction": self._slot_extraction,
            "fallback_agent_id": self._fallback_agent_id,
            "denylist": sorted(self._denylist) if self._denylist else None,
            "languages": {},
        }
        for lang, (resolver, user_candidates, builtin_candidates) in self._pools.items():
            user_by_intent: dict[str, list[str]] = {}
            for c in user_candidates:
                user_by_intent.setdefault(c.intent, []).append(c.text)
            builtin_by_intent: dict[str, list[str]] = {}
            for c in builtin_candidates:
                builtin_by_intent.setdefault(c.intent, []).append(c.text)
            out["languages"][lang] = {
                "user_candidate_count": len(user_candidates),
                "builtin_candidate_count": len(builtin_candidates),
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


def _dedupe(items: Iterable[str]) -> list[str]:
    """Stable de-duplicate."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        norm = s.strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _slot_list_values(
    lst,
    intents,
    sample_expression,
    TextSlotList,
    RangeSlotList,
) -> list[str]:
    """
    Best-effort enumeration of values from a Hassil SlotList.

    - Wildcards return ``[]`` (can't enumerate).
    - Text lists flatten each value's input pattern via ``sample_expression``.
    - Range lists enumerate digit forms, word forms (e.g. ``"zwölf"``) are out of
      scope here, as Hassil resolves them downstream when the canonical sentence is forwarded.
    """
    values: list[str] = []
    try:
        if isinstance(lst, TextSlotList):
            for v in getattr(lst, "values", []) or []:
                expr = getattr(getattr(v, "text_in", None), "expression", None)
                if expr is None:
                    continue
                try:
                    values.extend(sample_expression(expr, intents))
                except Exception:
                    continue
        elif isinstance(lst, RangeSlotList):
            from_value = getattr(lst, "from_value", 0)
            to_value = getattr(lst, "to_value", 0)
            step = getattr(lst, "step", 1) or 1
            if from_value <= to_value:
                values.extend(str(i) for i in range(from_value, to_value + 1, step))
    except Exception:  # pragma: no cover
        pass
    return _dedupe(values)


def _is_exposed(hass: HomeAssistant, entity_id: str) -> bool:
    """
    Best-effort check that ``entity_id`` is voice-exposed.

    Assume expose is true by default.
    Not a security concern is unexposed, as we never call any actions or the like,
    just forward are cleaned/matched sentence to Hassil.
    If Hassil fucks up on this, not our fault :)
    """
    try:
        from homeassistant.components.conversation import const as conv_const  # type: ignore

        DOMAIN = getattr(conv_const, "DOMAIN", "conversation")
    except Exception:
        DOMAIN = "conversation"

    try:
        from homeassistant.helpers.entity import async_should_expose  # type: ignore

        return async_should_expose(hass, DOMAIN, entity_id)
    except Exception:
        pass
    try:
        from homeassistant.helpers import exposed_entities  # type: ignore

        return exposed_entities.async_should_expose(hass, DOMAIN, entity_id)
    except Exception:
        return True
