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
        PER_INTENT_CANDIDATE_CAP,
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
        PER_INTENT_CANDIDATE_CAP,
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
        base_agent_id=opt(CONF_BASE_AGENT, DEFAULT_BASE_AGENT),
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
        base_agent_id=opt(CONF_BASE_AGENT, DEFAULT_BASE_AGENT),
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
        base_agent_id: str,
        entry_id: str,
    ) -> None:
        self.hass = hass
        self._threshold = threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
        self._slot_extraction = slot_extraction
        self._base_agent_id = base_agent_id
        self._entry_id = entry_id

        # Per-language pools: built lazily on first request for that
        # language. A user with multiple Assist pipelines in different
        # languages gets a fresh pool for each one.
        # Tuple is (resolver, user_candidates, builtin_candidates).
        # Builtins are kept separate so we can fall back to them only when
        # the user pool produces no match.
        self._pools: dict[str, tuple[Resolver, list[Candidate], list[Candidate]]] = {}
        self._pool_locks: dict[str, asyncio.Lock] = {}
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
        base_agent_id: str,
    ) -> None:
        self._threshold = threshold
        self._expansion_cap = expansion_cap
        self._denylist = set(denylist) if denylist else None
        self._include_builtins = include_builtins
        self._slot_extraction = slot_extraction
        self._base_agent_id = base_agent_id
        # Anything affecting candidate composition invalidates the pools.
        self._pools.clear()

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
        for lang in languages:
            try:
                await self._async_get_pool(lang)
            except Exception:  # pragma: no cover
                _LOGGER.exception("closest_intent: rebuild for %s failed", lang)

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
            _LOGGER.warn("PyYAML not importable; skipping custom_sentences")
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

        try:
            canonical = self._best_canonical(user_input, resolver, user_candidates)
            if canonical is None and builtin_candidates:
                # User pool produced nothing usable; fall back to builtins so
                # vanilla "turn on the light" still resolves when no custom
                # intent covers it.
                canonical = self._best_canonical(user_input, resolver, builtin_candidates)
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
            "include_builtins": self._include_builtins,
            "slot_extraction": self._slot_extraction,
            "base_agent_id": self._base_agent_id,
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
