"""
End-to-end matching tests against an example intent corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "custom_components" / "closest_intent"),
)

import pytest  # noqa: E402
from const import SLOT_WILDCARD  # type: ignore  # noqa: E402
from matching import (  # type: ignore  # noqa: E402
    Candidate,
    Resolver,
    build_canonical,
    expand_pattern,
    extract_slots,
    find_best,
)

THRESHOLD = 70
EXPANSION_CAP = 32
_RESOLVER = Resolver(match_threshold=THRESHOLD)
_RESOLVER_ANY = Resolver(match_threshold=0)


CORPUS: dict[str, list[str]] = {
    "Botty_Start": [
        "Starte die Reinigung",
        "Beginne Reinigung",
        "Reinigung starten",
        "Botty los",
        "Botty saugen",
        "Sauge",
    ],
    "Botty_Ende": [
        "Beende die Reinigung",
        "Stoppe Reinigung",
        "Reinigung beenden",
        "Botty zurück",
        "Botty nach Hause",
        "Botty stop",
    ],
    "Botty_Wohnzimmer": [
        "Reinige im Wohnzimmer",
        "Sauge das Wohnzimmer",
        "Botty ins Wohnzimmer",
        "Wohnzimmer reinigen",
    ],
    "Botty_Buero": [
        "Reinige Büro",
        "Sauge im Arbeitszimmer",
        "Büro reinigen",
        "Arbeitszimmer saugen",
    ],
    "Botty_Kueche": [
        "Reinige in der Küche",
        "Sauge die Küche",
        "Küche reinigen",
    ],
    "Botty_Sofa": [
        "Reinige vor dem Sofa",
        "Sauge unter dem Fernseher",
        "Sofa reinigen",
    ],
    "PumpeAn": [
        "Aktiviere die Pumpe",
        "Schalte die Wasserpumpe an",
        "Pumpe an",
        "Wasserpumpe ein",
    ],
    "PumpeAus": [
        "Deaktiviere die Pumpe",
        "Schalte die Wasserpumpe aus",
        "Pumpe aus",
        "Wasserpumpe ab",
    ],
    "MusikAn": [
        "Spiele Musik",
        "Spiel die Musik",
        "Starte Musik",
        "Musik an",
        "Musik abspielen",
    ],
    "MusikFortsetzen": [
        "Musik fortsetzen",
        "Mache Musik fort",
        "Setze Musik fort",
        "Weiter abspielen",
        "Weiterspielen",
    ],
    "MusikPause": [
        "Pausiere die Musik",
        "Stoppe Musik",
        "Musik pausieren",
        "Musik anhalten",
        "Pause",
    ],
    "MusikNaechster": [
        "Nächster Titel",
        "Nächstes Lied",
        "Skip",
        "Weiter",
    ],
    "MusikShuffleAn": [
        "Shuffle an",
        "Mischen ein",
        "Zufallswiedergabe aktivieren",
    ],
    "MusikShuffleAus": [
        "Shuffle aus",
        "Mischen ab",
        "Zufallswiedergabe deaktivieren",
    ],
    "PlayerNeustart": [
        "Player neu starten",
        "Spieler neustarten",
        "Sonos resetten",
        "Restart Player",
    ],
    "ZufaelligesAlbum": [
        "Spiele ein zufälliges Album",
        "Zufälliges Album",
        "Random Album",
    ],
    "ZufaelligerKuenstler": [
        "Spiele einen zufälligen Künstler",
        "Zufälliger Artist",
        "Random Artist",
    ],
    "NeueMusik": [
        "Spiele die neue Musik",
        "Spiel die neuesten Tracks",
        "Spiele die Playlist Recently Added",
        "Recently Added",
    ],
    "KuerzlichGespielt": [
        "Spiele die zuletzt gehörten Titel",
        "Spiel die zuletzt gespielten Lieder",
        "Recently Played",
        "Spiel die selben Songs nochmal",
    ],
    "Tagesschau": [
        "Spiele die Tagesschau",
        "Spiel Tagesschau in 100 Sekunden",
        "Starte die Tagesschau",
        "Tagesschau",
    ],
    "WDR_Aktuell": [
        "Spiele WDR Aktuell",
        "WDR Nachrichten",
    ],
    "Nachrichten": [
        "Spiele die Nachrichten",
        "Starte Nachrichten",
        "Nachrichten",
        "Tägliche Zusammenfassung",
    ],
    "UhrZeit": [
        "Wie spät ist es",
        "Wie viel Uhr ist es",
        "Uhrzeit",
    ],
    "Datum": [
        "Welches Datum haben wir",
        "Was ist heute für ein Datum",
        "Datum",
    ],
    "Wochentag": [
        "Welcher Tag ist heute",
        "Welcher Wochentag ist heute",
        "Was ist heute für ein Tag",
        "Tag",
        "Wochentag",
    ],
    "TV_Hell": [
        "Mache den Fernseher heller",
        "Setze das Bild hell",
        "Fernseher Tagmodus",
    ],
    "TV_Dunkel": [
        "Mache den Fernseher dunkel",
        "Stelle das Bild dunkler",
        "Fernseher Nachtmodus",
    ],
    "WetterHeute": [
        "Wie ist das Wetter heute",
        "Wie ist das Wetter draußen",
        "Wie warm ist es draußen",
    ],
    "WetterMorgen": [
        "Wie wird das Wetter morgen",
        "Wie wird das Wetter morgen früh",
        "Wie warm wird es morgen",
    ],
    "WetterWoche": [
        "Wie wird das Wetter diese Woche",
        "Wie wird das Wetter in den nächsten Tagen",
        "Wettervorhersage",
    ],
    "WindAktuell": [
        "Wie windig ist es heute",
        "Wie stark weht der Wind",
    ],
    "WindHeuteNacht": [
        "Wie windig wird es heute Nacht",
        "Wie windig wird es nachts",
    ],
    "TemperaturMaxHeute": [
        "Wie warm wird es heute",
        "Was ist die Höchsttemperatur heute",
    ],
    "RegenHeute": [
        "Regnet es heute",
        "Wird es heute regnen",
        "Gibt es heute Regen",
    ],
}


SLOT_CORPUS: list[tuple[str, str, str, list[str]]] = [
    # (intent, pattern, user_text, expected_slots)
    (
        "WetterStunde",
        "Wie ist das Wetter um {timer_hours:hours} Uhr",
        "wie ist das wetter um zwölf uhr",
        ["zwölf"],
    ),
    (
        "WetterStunde",
        "Wie wird das Wetter um {timer_hours:hours} Uhr",
        "wie wird das wetter um 14 uhr",
        ["14"],
    ),
    ("RegenStunde", "Regnet es um {timer_hours:hours} Uhr", "regnet es um 18 uhr", ["18"]),
    ("Test_Area", "Test zwei im {area}", "test zwei im wohnzimmer", ["wohnzimmer"]),
    ("Test_Name", "Test drei mit {name}", "test drei mit charlotte", ["charlotte"]),
    (
        "Einkauf_Add",
        "(setze|pack|tu|schreib) {item} auf (die|meine) Einkaufsliste",
        "schreib brot auf die einkaufsliste",
        ["brot"],
    ),
    ("Einkauf_Add", "{item} auf die Einkaufsliste", "salami auf die einkaufsliste", ["salami"]),
    (
        "Einkauf_Add",
        "Füge {item} zur Einkaufsliste hinzu",
        "füge milch zur einkaufsliste hinzu",
        ["milch"],
    ),
    (
        "ToDo_Add",
        "(setze|pack|tu|schreib) {item} auf (die|meine) (ToDo|To-Do|To Do)-Liste",
        "schreib termin auf die todo-liste",
        ["termin"],
    ),
    (
        "MusikPlaylist",
        "(Spiele|Spiel|Starte) [die ]Playlist {playlist}",
        "spiele playlist sea shanties",
        ["sea shanties"],
    ),
]


def _build_no_slot_candidates() -> list[Candidate]:
    """
    Treat each utterance pattern in CORPUS as if it came from a user-defined intent
    and expand into candidates.
    """
    out: list[Candidate] = []
    for intent_name, phrases in CORPUS.items():
        for idx, phrase in enumerate(phrases):
            for text, display_text, slot_names in expand_pattern(phrase, EXPANSION_CAP):
                out.append(
                    Candidate(
                        intent=intent_name,
                        pattern_idx=idx,
                        text=text,
                        display_text=display_text,
                        slot_names=slot_names,
                    )
                )
    return out


_CANDIDATES = _build_no_slot_candidates()


NOSLOT_PARAMS = [
    pytest.param(intent_name, phrase, id=f"{intent_name}::{phrase}")
    for intent_name, phrases in CORPUS.items()
    for phrase in phrases
]


@pytest.mark.parametrize("intent_name,phrase", NOSLOT_PARAMS)
def test_corpus_clean_phrase_matches(intent_name: str, phrase: str) -> None:
    """Exact corpus phrase must match its own intent above threshold."""
    match = find_best(phrase, _CANDIDATES, _RESOLVER)
    assert match is not None, f"no match for {phrase!r}"
    assert match[0].intent == intent_name, (
        f"{phrase!r} matched {match[0].intent!r} instead of {intent_name!r}"
    )


# A handful of representative typos / abbreviations.
# Check that the fuzzy matcher actually delivers value.
TYPO_CASES = [
    ("Botty_Start", "starte rinigung"),  # one-char typo
    ("PumpeAn", "pumpr an"),  # one-char typo
    ("MusikShuffleAn", "shuffl an"),  # truncation
    ("MusikPause", "pausir die musik"),  # typo + alternation
    ("UhrZeit", "wie sät ist es"),  # one-char drop
    ("WetterHeute", "wie warm ist es draussn"),  # one-char typo
    ("Tagesschau", "spiel tagesshau"),  # one-char typo
    # yeah these are not really typos, just irrelevant, but w/e
    ("WDR_Aktuell", "WDR-Aktuell"),  # internal punctuation in user input
    ("WDR_Aktuell", "WDR.Aktuell"),
    ("UhrZeit", "Wie spät ist es?"),  # trailing sentence punctuation
]


@pytest.mark.parametrize("intent_name,phrase", TYPO_CASES)
def test_corpus_typo_matches(intent_name: str, phrase: str) -> None:
    match = find_best(phrase, _CANDIDATES, _RESOLVER)
    assert match is not None, f"no match for typo'd {phrase!r}"
    assert match[0].intent == intent_name


@pytest.mark.parametrize(
    "intent_name,pattern,user_text,expected",
    [pytest.param(*row, id=f"{row[0]}::{row[2]}") for row in SLOT_CORPUS],
)
def test_slot_corpus_extracts(
    intent_name: str,
    pattern: str,
    user_text: str,
    expected: list[str],
) -> None:
    """Slot patterns: best expansion is picked, slot text aligns."""
    expansions = expand_pattern(pattern, EXPANSION_CAP)
    candidates = [
        Candidate(
            intent=intent_name,
            pattern_idx=0,
            text=text,
            display_text=display_text,
            slot_names=slots,
        )
        for text, display_text, slots in expansions
    ]
    match = find_best(user_text, candidates, _RESOLVER)
    assert match is not None, f"no match for {user_text!r}"

    # Walk siblings: the highest-scoring expansion may not be the one
    # whose fixed parts align. Production code does the same fallback.
    # We additionally reject empty captures: when two expansions tie on
    # partial_ratio (alternations like "(setze|schreib) ..."), the one
    # whose prefix isn't actually in the user text "extracts" with an
    # empty slot, which isn't useful. Prefer extractions that actually
    # captured something.
    best = None
    for c, _s in sorted(
        ((c, find_best(user_text, [c], _RESOLVER_ANY)[1]) for c in candidates),
        key=lambda kv: -kv[1],
    ):
        captured = extract_slots(user_text, c)
        if captured is None:
            continue
        if any(s.strip() for s in captured):
            best = (c, captured)
            break
    assert best is not None, f"no extractable expansion for {user_text!r}"
    candidate, captured = best
    assert captured == expected, (
        f"expected {expected!r}, got {captured!r} (matched expansion {candidate.text!r})"
    )


def test_resolver_canonicalises_typo_d_area() -> None:
    """
    The end-to-end shape:
        pattern -> expansion -> score -> extract -> resolve slot value -> canonical sentence
    """
    pattern = "Test zwei im {area}"
    candidates = [
        Candidate(
            intent="Test_Area",
            pattern_idx=0,
            text=text,
            display_text=display_text,
            slot_names=slots,
        )
        for text, display_text, slots in expand_pattern(pattern, EXPANSION_CAP)
    ]
    resolver = Resolver(slot_values={"area": ["Wohnzimmer", "Büro", "Küche"]})
    user = "test zwei im wohnzma"
    match = find_best(user, candidates, _RESOLVER)
    assert match is not None
    captured = extract_slots(user, match[0])
    assert captured is not None
    canonical = build_canonical(match[0], captured, resolver=resolver)
    assert canonical == "Test zwei im Wohnzimmer"


# Below here: actual misfires I encountered.
# Regression tests, if you will.


def _full_einkauf_todo_pool() -> list[Candidate]:
    patterns = {
        "Einkauf_Add": [
            "(setze|pack|tu|schreib) {item} auf (die|meine) Einkaufsliste",
            "{item} auf die Einkaufsliste",
            "Einkaufsliste {item}",
            "Füge {item} zur Einkaufsliste hinzu",
        ],
        "ToDo_Add": [
            "(setze|pack|tu|schreib) {item} auf (die|meine) (ToDo|To-Do|To Do)-Liste",
            "{item} auf die ToDo-Liste",
            "ToDo-Liste {item}",
        ],
    }
    out: list[Candidate] = []
    for intent_name, pats in patterns.items():
        for idx, pat in enumerate(pats):
            for text, display_text, slots in expand_pattern(pat, EXPANSION_CAP):
                out.append(
                    Candidate(
                        intent=intent_name,
                        pattern_idx=idx,
                        text=text,
                        display_text=display_text,
                        slot_names=slots,
                    )
                )
    return out


def _agent_match(user: str, candidates: list[Candidate]) -> tuple[str, list[str]] | None:
    """
    Mimic the agent's full match -> extract -> fallback flow without
    booting the conversation entity. Returns (intent, captured) or None.
    """
    match = find_best(user, candidates, _RESOLVER)
    if match is None:
        return None
    candidate, _ = match
    if not candidate.has_slots:
        return (candidate.intent, [])
    captured = extract_slots(user, candidate)
    if captured is None:
        # walk same-intent siblings in score order until one extracts
        scored = sorted(
            (
                (c, find_best(user, [c], _RESOLVER_ANY)[1])
                for c in candidates  # type: ignore
                if c.intent == candidate.intent and c.has_slots
            ),
            key=lambda kv: -kv[1],
        )
        for c, s in scored:
            if s < THRESHOLD:
                break
            captured = extract_slots(user, c)
            if captured is not None:
                candidate = c
                break
        else:
            return None
    return (candidate.intent, captured)


def test_regression_tudu_liste_does_not_capture_short_alternation() -> None:
    """
    Bug: 'Setze Arzt anrufen auf meine Tudu-Liste' resolved to ToDo
    with item='tu', because the short ``tu`` alternation in
    ``(setze|pack|tu|schreib)`` aligned its 2-char prefix to the ``tu``
    *inside* "Tudu-Liste" and out-scored the structurally-correct
    ``setze`` expansion via ``partial_ratio``. The anchor penalty makes
    a leading ``tu`` that isn't actually at the start of user text
    cost more than the score it gains.
    """
    user = "setze arzt anrufen auf meine tudu-liste"
    result = _agent_match(user, _full_einkauf_todo_pool())
    assert result is not None
    intent, captured = result
    assert intent == "ToDo_Add", f"matched wrong intent: {intent}"
    assert captured == ["arzt anrufen"], f"bad capture: {captured!r}"


def test_regression_setze_dosenmais_einkaufsliste_variants() -> None:
    """
    Bug: 'Setze Dosenmais auf meine Einkaufsliste' (and STT variants
    'meiner einkaufsliste', 'meiner einkauflöste') previously got
    NO_INTENT_MATCH because ``Einkaufsliste {item}`` scored 100 via
    substring partial_ratio, captured an empty slot, and produced a
    canonical that the base agent rejected. The anchor penalty
    reduces ``Einkaufsliste {item}`` (whose leading anchor sits 4
    tokens deep in the user text) below threshold, letting the
    structurally-correct ``setze {item} auf meine einkaufsliste``
    expansion win.
    """
    pool = _full_einkauf_todo_pool()
    for user in [
        "setze dosenmais auf meine einkaufsliste",
        "setze dosenmais auf meiner einkaufsliste",
        "setze dosenmais auf meiner einkauflöste",
    ]:
        result = _agent_match(user, pool)
        assert result is not None, f"no match for {user!r}"
        intent, captured = result
        assert intent == "Einkauf_Add", f"{user!r}: matched wrong intent {intent}"
        assert captured == ["dosenmais"], f"{user!r}: bad capture {captured!r}"


def test_regression_setze_milch_picks_setze_anchored_expansion() -> None:
    """
    Bug: 'Setze Milch auf die Einkaufsliste' matched ``{item} auf die
    einkaufsliste`` capturing item='setze milch' instead of the
    structurally correct ``setze {item} auf die einkaufsliste``
    expansion (item='milch').
    """
    pool = _full_einkauf_todo_pool()
    user = "setze milch auf die einkaufsliste"
    result = _agent_match(user, pool)
    assert result is not None
    intent, captured = result
    assert intent == "Einkauf_Add", f"matched wrong intent: {intent}"
    assert captured == ["milch"], f"bad capture: {captured!r}"

    # Also assert the chosen candidate is the setze-anchored one,
    # not the bare slot-leading one.
    match = find_best(user, pool, _RESOLVER)
    assert match is not None
    candidate, _ = match
    leading = candidate.text.split(SLOT_WILDCARD)[0].strip()
    assert "setze" in leading, (
        f"expected leading anchor with 'setze', got pattern {candidate.text!r}"
    )


def test_regression_stt_split_in_trailing_fixed_part_does_not_leak_into_slot() -> None:
    """Make sure start and end are both refined for word boundaries"""
    user = "setz e milch a uf die iasnfksugliste"
    result = _agent_match(user, _full_einkauf_todo_pool())
    assert result is not None
    intent, captured = result
    assert intent == "Einkauf_Add", f"matched wrong intent: {intent}"
    assert captured == ["milch"], f"bad capture: {captured!r}"


def test_regression_stt_corrupted_setze_picks_long_anchored_intent() -> None:
    """
    Ensure that garbled optional prefixes aren't pulled into the slot value
    when a similar intent without the prefix exists.
    """
    pool = _full_einkauf_todo_pool()
    cases = [
        "setz milch auf die einkaufsliste",
        "stze milch auf die einkaufsliste",
        "set ze milch auf die einkaufsliste",
        "setz e milch auf die einkaufsliste",
        "setz e milch auf die einkaufslis ste",
        "setze milch auf die einkaufslis ste",
        "setze milch auf die einkaufslsite",
    ]
    for user in cases:
        result = _agent_match(user, pool)
        assert result is not None, f"no match for {user!r}"
        intent, captured = result
        assert intent == "Einkauf_Add", f"{user!r}: matched wrong intent {intent}"
        assert captured == ["milch"], f"{user!r}: bad capture {captured!r}"


def test_regression_fuege_hinzu_does_not_match_einkaufsliste_only() -> None:
    """
    Bug: 'Füge Dosenmais zur Einkaufsliste hinzu' added 'hinzu' to
    the shopping list because ``Einkaufsliste {item}`` matched at 100
    via substring partial_ratio and captured the trailing ``hinzu``
    as the slot. The anchor penalty rejects that candidate (its
    leading anchor sits 3 tokens deep) and the proper
    ``Füge {item} zur Einkaufsliste hinzu`` pattern wins, capturing
    'dosenmais'.
    """
    user = "füge dosenmais zur einkaufsliste hinzu"
    result = _agent_match(user, _full_einkauf_todo_pool())
    assert result is not None
    intent, captured = result
    assert intent == "Einkauf_Add"
    assert captured == ["dosenmais"], f"bad capture: {captured!r}"


def _pool_from_patterns(patterns: dict[str, list[str]]) -> list[Candidate]:
    out: list[Candidate] = []
    for intent_name, pats in patterns.items():
        for idx, pat in enumerate(pats):
            for text, display_text, slots in expand_pattern(pat, EXPANSION_CAP):
                out.append(
                    Candidate(
                        intent=intent_name,
                        pattern_idx=idx,
                        text=text,
                        display_text=display_text,
                        slot_names=slots,
                    )
                )
    return out


_MUSIC_POOL = _pool_from_patterns(
    {
        # Slot-bearing playlist intent competes with non-slot music intents
        # that share the "Spiele" / "Spiel die" prefix.
        "MusikPlaylist": [
            "(Spiele|Spiel|Starte) [die ]Playlist {playlist}",
            "Playlist {playlist}",
        ],
        "MusikAn": [
            "(Spiele|Spiel|Starte) [Musik|die Musik]",
            "Musik (an|abspielen|starten)",
        ],
        "ZufaelligesAlbum": [
            "(Spiele|Spiel) [ein ]zufälliges Album",
            "Zufälliges Album",
        ],
        "NeueMusik": [
            "(Spiele|Spiel) [die ]neue[sten|n] (Musik|Tracks|Titel|Lieder)",
            "(Spiele|Spiel) [die ]Playlist (neue Musik|Neue Tracks|Recently Added)",
        ],
    }
)


def test_regression_musikplaylist_picks_anchored_over_bare_slot() -> None:
    """
    Bug: 'Spiele die Playlist Sea Shanties' could match the bare
    ``Playlist {playlist}`` pattern, absorbing 'Spiele die' into the
    leading boundary slot.
    """
    user = "spiele die playlist sea shanties"
    result = _agent_match(user, _MUSIC_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "MusikPlaylist"
    assert captured == ["sea shanties"], f"bad capture: {captured!r}"


def test_regression_musikplaylist_multiword_slot() -> None:
    """
    Multi-word playlist names ('Bridgerton Pop') must end up in the
    slot, not in fixed text. The longer-anchored expansion is also the
    only one whose canonical reconstructs to a sentence the official
    parser will accept.
    """
    user = "spiel die playlist bridgerton pop"
    result = _agent_match(user, _MUSIC_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "MusikPlaylist"
    assert captured == ["bridgerton pop"], f"bad capture: {captured!r}"


def test_regression_musik_an_does_not_get_swallowed_by_playlist_slot() -> None:
    """'Spiele Musik' should pick MusikAn, not MusikPlaylist with playlist=['musik']."""
    user = "spiele musik"
    result = _agent_match(user, _MUSIC_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "MusikAn", f"matched wrong intent: {intent}"


_TODO_POOL = _pool_from_patterns(
    {
        "ToDo_Add": [
            "(setze|pack|tu|schreib) {item} auf (die|meine) (ToDo|To-Do|To Do)-Liste",
            "{item} auf die ToDo-Liste",
            "ToDo-Liste {item}",
        ],
    }
)


def test_regression_todo_multiword_item_with_setze_anchor() -> None:
    """must capture the full multi-word item, not split it across the slot and a fixed suffix"""
    user = "schreib zahnarzt termin auf meine to-do-liste"
    result = _agent_match(user, _TODO_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "ToDo_Add"
    assert captured == ["zahnarzt termin"], f"bad capture: {captured!r}"


_EINKAUF_MULTIWORD_POOL = _full_einkauf_todo_pool()


def test_regression_einkauf_multiword_item_with_setze_anchor() -> None:
    """must capture the full multi-word item, not split it across the slot and a fixed suffix"""
    user = "pack frische tomaten auf meine einkaufsliste"
    result = _agent_match(user, _EINKAUF_MULTIWORD_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "Einkauf_Add"
    assert captured == ["frische tomaten"], f"bad capture: {captured!r}"


_WETTER_POOL = _pool_from_patterns(
    {
        # Slot-bearing weather hour intent competes with the bare
        # WetterHeute/WetterMorgen no-slot patterns that share the
        # 'Wie wird das Wetter' prefix.
        "WetterStunde": [
            "Wie [wird|ist] das Wetter um {timer_hours:hours} Uhr",
            "Wie warm wird es um {timer_hours:hours} Uhr",
        ],
        "WetterHeute": [
            "Wie ist das Wetter (heute|jetzt|gerade|draußen|aktuell)",
            "Wie warm ist es (draußen|gerade|jetzt)",
        ],
        "WetterMorgen": [
            "Wie wird das Wetter morgen [früh|nachmittag|abend]",
            "Wie warm wird es morgen",
        ],
    }
)


def test_regression_wetter_stunde_keeps_hour_in_slot() -> None:
    """'Wie wird das Wetter um 14 Uhr' must match WetterStunde with capture ['14']."""
    user = "wie wird das wetter um 14 uhr"
    result = _agent_match(user, _WETTER_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "WetterStunde"
    assert captured == ["14"], f"bad capture: {captured!r}"


def test_regression_wetter_morgen_does_not_steal_stunde_pattern() -> None:
    """
    'Wie wird das Wetter morgen' must match WetterMorgen, not WetterStunde with capture ['morgen']
    """
    user = "wie wird das wetter morgen"
    result = _agent_match(user, _WETTER_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "WetterMorgen", f"matched wrong intent: {intent}"


_MUSIK_SONG_POOL = _pool_from_patterns(
    {
        "Musik_Song": [
            "Spiele Lied {mass_track}",
            "Spiele Song {mass_track}",
            "Spiel Lied {mass_track}",
            "Spiel Song {mass_track}",
        ],
        "Musik_SongVonKuenstler": [
            "Spiele Lied {mass_track} von {mass_artist}",
            "Spiele Song {mass_track} von {mass_artist}",
            "Spiel Lied {mass_track} von {mass_artist}",
            "Spiel Song {mass_track} von {mass_artist}",
        ],
    }
)


def test_regression_song_pattern_not_shadowed_by_song_von_kuenstler() -> None:
    """
    'Spiele Lied X' (1 slot) must win over 'Spiele Lied X von Y' (2 slots)
    when the user input contains no 'von' connective.
    """
    user = "spiele lied bohemian rhapsody"
    result = _agent_match(user, _MUSIK_SONG_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "Musik_Song", f"matched wrong intent: {intent}"
    assert captured == ["bohemian rhapsody"], f"bad capture: {captured!r}"


def test_regression_song_pattern_with_von_routes_to_kuenstler() -> None:
    """The 2-slot variant still wins when ' von ' IS present in the input."""
    user = "spiele lied bohemian rhapsody von queen"
    result = _agent_match(user, _MUSIK_SONG_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "Musik_SongVonKuenstler", f"matched wrong intent: {intent}"
    assert captured == ["bohemian rhapsody", "queen"], f"bad capture: {captured!r}"


_SONNENAUFGANG_POOL = _pool_from_patterns(
    {
        "Sonnenaufgang_Heute": [
            "Wann geht die Sonne auf",
            "Wann ist Sonnenaufgang",
            "Wann ist der Sonnenaufgang",
        ],
        "Sonnenaufgang_Morgen": [
            "Wann geht die Sonne morgen auf",
            "Wann ist morgen Sonnenaufgang",
            "Wann ist morgen der Sonnenaufgang",
        ],
    }
)


def test_regression_sonnenaufgang_heute_not_shadowed_by_morgen() -> None:
    """
    Both Sonnenaufgang_Morgen patterns are supersets of their Heute siblings
    (one extra word `morgen`). On 'Wann geht die Sonne auf', the source
    Heute candidate scores 100 while Morgen scores ~86 -- within band 15
    and Morgen's longer fixed text would dominate without an alignment check.
    """
    user = "wann geht die sonne auf"
    result = _agent_match(user, _SONNENAUFGANG_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Sonnenaufgang_Heute", f"matched wrong intent: {intent}"


def test_regression_sonnenaufgang_morgen_keeps_explicit_morgen() -> None:
    """The morgen variant still wins when 'morgen' is in the user input."""
    user = "wann geht die sonne morgen auf"
    result = _agent_match(user, _SONNENAUFGANG_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Sonnenaufgang_Morgen", f"matched wrong intent: {intent}"


_TIMER_POOL = _pool_from_patterns(
    {
        "Timer_Stellen_Stunden": [
            "Stelle einen Timer für {timer_hours} Stunden",
            "Stelle Timer für {timer_hours} Stunden",
        ],
        "Timer_Stellen_Minuten": [
            "Stelle einen Timer für {timer_minutes} Minuten",
        ],
        "Timer_Stellen_Sekunden": [
            "Stelle einen Timer für {timer_seconds} Sekunden",
            "Stelle Timer für {timer_seconds} Sekunden",
        ],
        "Timer_Stellen_Kombiniert": [
            "Stelle einen Timer für {timer_hours} Stunden {timer_minutes} Minuten",
            "Stelle Timer für {timer_hours} Stunden {timer_minutes} Minuten",
        ],
    }
)


def test_regression_timer_kombiniert_multi_slot_self_routes() -> None:
    """
    Multi-slot candidates (`{h} Stunden {m} Minuten`) must score above
    threshold on their own perfect input.
    """
    user = "stelle einen timer für 2 stunden 30 minuten"
    result = _agent_match(user, _TIMER_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "Timer_Stellen_Kombiniert", f"matched wrong intent: {intent}"
    assert captured == ["2", "30"], f"bad capture: {captured!r}"


def test_regression_timer_sekunden_not_shadowed_by_stunden() -> None:
    """
    `{s} Sekunden` and `{h} Stunden` share an identical prefix and the
    trailing tokens fuzzy-overlap (`unden` lives inside `sekunden`).
    """
    user = "stelle einen timer für 30 sekunden"
    result = _agent_match(user, _TIMER_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Timer_Stellen_Sekunden", f"matched wrong intent: {intent}"


def test_regression_timer_stunden_routes_correctly() -> None:
    """The Stunden variant wins when the user actually said `Stunden`."""
    user = "stelle einen timer für 2 stunden"
    result = _agent_match(user, _TIMER_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Timer_Stellen_Stunden", f"matched wrong intent: {intent}"


_MUSIK_MULTI_POOL = _pool_from_patterns(
    {
        "Musik_Genre": ["Spiele {mass_genre} Musik"],
        "Musik_Kuenstler": ["Spiele Artist {mass_artist}"],
        "Musik_Album": ["Spiele Album {mass_album}"],
        "MusikAn": ["Spiele"],
    }
)


def test_regression_musik_genre_not_shadowed_by_sibling_intents() -> None:
    """
    On `"Spiele jazz Musik"`, the source `Musik_Genre` must win over
    `Musik_Kuenstler` and `Musik_Album` siblings whose internal anchors
    (`Artist`, `Album`) are absent from user input.
    """
    user = "spiele jazz musik"
    result = _agent_match(user, _MUSIK_MULTI_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "Musik_Genre", f"matched wrong intent: {intent}"
    assert captured == ["jazz"], f"bad capture: {captured!r}"


def test_regression_musik_kuenstler_routes_correctly() -> None:
    """The Artist variant wins when the user says `Artist`."""
    user = "spiele artist queen"
    result = _agent_match(user, _MUSIK_MULTI_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Musik_Kuenstler", f"matched wrong intent: {intent}"


_TOKEN_PREFIX_POOL = _pool_from_patterns(
    {
        "MusikAn": ["Spiel Musik", "Starte"],
        "Musik_Genre_Long": ["Spiel Musikrichtung {mass_genre}"],
        "Musik_Kuenstler": ["Starte Artist {mass_artist}"],
    }
)


def test_regression_0slot_phrase_not_shadowed_by_token_prefix_match() -> None:
    """
    A bare 0-slot ``"Spiel Musik"`` must not be shadowed by a 1-slot
    ``"Spiel Musikrichtung {genre}"``.
    """
    user = "spiel musik"
    result = _agent_match(user, _TOKEN_PREFIX_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "MusikAn", f"matched wrong intent: {intent}"


def test_regression_0slot_verb_not_shadowed_by_slotted_continuation() -> None:
    """``"Starte"`` must not be shadowed by ``"Starte Artist {artist}"``."""
    user = "starte"
    result = _agent_match(user, _TOKEN_PREFIX_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "MusikAn", f"matched wrong intent: {intent}"


_LOOSE_ALIGNMENT_POOL = _pool_from_patterns(
    {
        "ShuffleAn": ["Zufallswiedergabe an"],
        "ShuffleAus": ["Zufallswiedergabe aus"],
    }
)


def test_regression_loose_alignment_rescues_user_word_split() -> None:
    """
    User-side word splits (STT producing ``"zufalls wiedergabe"`` for
    ``"zufallswiedergabe"``) must resolve correctly.
    """
    user = "zufalls wiedergabe an"
    result = _agent_match(user, _LOOSE_ALIGNMENT_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "ShuffleAn", f"matched wrong intent: {intent}"


_AUS_POOL = _pool_from_patterns(
    {
        "Stille_Alle": ["Aus"],
        "DeviceTurnOff": ["{geraet} aus", "das {geraet} aus"],
    }
)


def test_regression_bare_aus_routes_to_stille_alle_not_empty_slot() -> None:
    """
    On bare ``"aus"``, the 0-slot ``Stille_Alle`` must win over
    ``DeviceTurnOff "{geraet} aus"``.
    """
    user = "aus"
    result = _agent_match(user, _AUS_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "Stille_Alle", f"matched wrong intent: {intent}"


def test_regression_geraet_aus_still_works_when_geraet_is_supplied() -> None:
    """The 1-slot ``{geraet} aus`` still wins when user actually names a device."""
    user = "pumpe aus"
    result = _agent_match(user, _AUS_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "DeviceTurnOff", f"matched wrong intent: {intent}"
    assert captured == ["pumpe"], f"bad capture: {captured!r}"


_IST_DAS_POOL = _pool_from_patterns(
    {
        "DeviceTurnOff": ["das {geraet} aus", "die {geraet} aus"],
        "DeviceStatus": ["Ist das {geraet} aus", "Ist die {geraet} aus"],
    }
)


def test_regression_ist_das_does_not_shadow_das() -> None:
    """
    A candidate ``"Ist das {geraet} aus"`` must not shadow the source
    ``"das {geraet} aus"`` on the latter's own canonical input.
    """
    user = "das pumpe aus"
    result = _agent_match(user, _IST_DAS_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "DeviceTurnOff", f"matched wrong intent: {intent}"


def test_regression_ist_das_still_wins_when_user_says_ist() -> None:
    """When the user actually said ``"Ist"``, DeviceStatus is the right match."""
    user = "ist das pumpe aus"
    result = _agent_match(user, _IST_DAS_POOL)
    assert result is not None
    intent, _ = result
    assert intent == "DeviceStatus", f"matched wrong intent: {intent}"


_STRAHLER_POOL = _pool_from_patterns(
    {
        "DeviceTurnOn": ["{geraet} an", "das {geraet} an", "die {geraet} an", "den {geraet} an"],
        "DeviceStatus": ["Ist {geraet} an", "Ist das {geraet} an", "Ist die {geraet} an"],
    }
)


def test_regression_strahler_an_does_not_route_to_ist_geraet_an() -> None:
    user = "strahler an"
    result = _agent_match(user, _STRAHLER_POOL)
    assert result is not None
    intent, captured = result
    assert intent == "DeviceTurnOn", f"matched wrong intent: {intent}"
    assert captured == ["strahler"], f"bad capture: {captured!r}"
