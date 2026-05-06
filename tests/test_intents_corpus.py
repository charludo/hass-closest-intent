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
            for text, slot_names in expand_pattern(phrase, EXPANSION_CAP):
                out.append(
                    Candidate(
                        intent=intent_name,
                        pattern_idx=idx,
                        text=text,
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
    match = find_best(phrase, _CANDIDATES, threshold=THRESHOLD)
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
]


@pytest.mark.parametrize("intent_name,phrase", TYPO_CASES)
def test_corpus_typo_matches(intent_name: str, phrase: str) -> None:
    match = find_best(phrase, _CANDIDATES, threshold=THRESHOLD)
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
            slot_names=slots,
        )
        for text, slots in expansions
    ]
    match = find_best(user_text, candidates, threshold=THRESHOLD)
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
        ((c, find_best(user_text, [c], 0)[1]) for c in candidates),  # type: ignore
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
            slot_names=slots,
        )
        for text, slots in expand_pattern(pattern, EXPANSION_CAP)
    ]
    resolver = Resolver(slot_values={"area": ["Wohnzimmer", "Büro", "Küche"]})
    user = "test zwei im wohnzma"
    match = find_best(user, candidates, threshold=THRESHOLD)
    assert match is not None
    captured = extract_slots(user, match[0])
    assert captured is not None
    canonical = build_canonical(match[0], captured, resolver=resolver)
    assert canonical == "test zwei im wohnzimmer"


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
            for text, slots in expand_pattern(pat, EXPANSION_CAP):
                out.append(
                    Candidate(
                        intent=intent_name,
                        pattern_idx=idx,
                        text=text,
                        slot_names=slots,
                    )
                )
    return out


def _agent_match(user: str, candidates: list[Candidate]) -> tuple[str, list[str]] | None:
    """
    Mimic the agent's full match -> extract -> fallback flow without
    booting the conversation entity. Returns (intent, captured) or None.
    """
    match = find_best(user, candidates, threshold=THRESHOLD)
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
                (c, find_best(user, [c], 0)[1])
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
