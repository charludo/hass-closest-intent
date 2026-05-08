"""
Unit tests for the pure-Python matching logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "custom_components" / "closest_intent")
)

from const import SLOT_WILDCARD  # type: ignore  # noqa: E402
from matching import (  # type: ignore  # noqa: E402
    Candidate,
    Resolver,
    build_canonical,
    expand_pattern,
    extract_slots,
    find_best,
    score,
)


def test_expand_no_syntax() -> None:
    out = expand_pattern("Wie spät ist es", cap=16)
    assert out == [("wie spät ist es", "Wie spät ist es", [])]


def test_expand_alternatives() -> None:
    texts = [t for (t, _, _) in expand_pattern("(Hallo|Guten Tag)", cap=16)]
    assert "hallo" in texts
    assert "guten tag" in texts


def test_expand_optional() -> None:
    texts = [t for (t, _, _) in expand_pattern("Pumpe [an]", cap=16)]
    assert "pumpe" in texts
    assert "pumpe an" in texts


def test_expand_optional_with_alternation_inside() -> None:
    """``[a|b]`` is semantically equivalent to ``(|a|b)``"""
    texts = {t for (t, _, _) in expand_pattern("Spiele [Musik|die Musik]", cap=16)}
    assert "spiele" in texts
    assert "spiele musik" in texts
    assert "spiele die musik" in texts


def test_expand_optional_with_alternation_cap_zero() -> None:
    """With expansion disabled, ``[a|b]`` collapses to its first branch."""
    out = expand_pattern("Spiele [Musik|die Musik]", cap=0)
    assert len(out) == 1
    assert out[0][0] == "spiele musik"


def test_expand_combined() -> None:
    texts = {t for (t, _, _) in expand_pattern("(Schalte|Mache) [die ]Pumpe an", cap=16)}
    assert "schalte pumpe an" in texts
    assert "mache die pumpe an" in texts


def test_expand_cap_zero_disables_expansion() -> None:
    out = expand_pattern("(a|b) [c] d", cap=0)
    assert len(out) == 1
    assert out[0][0] == "a c d"


def test_expand_records_slots_in_order() -> None:
    out = expand_pattern("Wetter um {stunde} Uhr am {tag}", cap=16)
    for _, _, slots in out:
        assert slots == ["stunde", "tag"]
    assert all(SLOT_WILDCARD in t for (t, _, _) in out)


def test_expand_records_slots_with_list_reference() -> None:
    # `{name:list}` syntax is also supported; only the name is captured.
    out = expand_pattern("Wetter um {stunde:time} Uhr", cap=16)
    for _, _, slots in out:
        assert slots == ["stunde"]


def test_expand_preserves_case_in_display_text() -> None:
    out = expand_pattern("(Spiele|Starte) WDR (Aktuell|aktuell)", cap=16)
    displays = {d for (_, d, _) in out}
    texts = {t for (t, _, _) in out}
    assert "Spiele WDR Aktuell" in displays
    assert "spiele wdr aktuell" in texts


def test_score_handles_typos() -> None:
    assert score("pumpr an", "pumpe an") >= 70


def test_score_handles_intra_word_truncation() -> None:
    assert score("shuffl an", "shuffle an") >= 70


def test_score_unrelated_with_shared_short_token() -> None:
    pumpe = score("schaffeln aus", "pumpe aus")
    shuffle = score("schaffeln aus", "shuffle aus")
    assert shuffle > pumpe
    assert pumpe < 70


def test_score_handles_extra_words() -> None:
    assert score("schalte mal die pumpe an", "schalte die pumpe an") >= 80


def test_score_ignores_slot_wildcard() -> None:
    cand = f"wie ist das wetter um {SLOT_WILDCARD} uhr"
    assert score("wie ist das wetter um zwölf uhr", cand) >= 80


def test_score_slot_pattern_with_multi_token_slot() -> None:
    cand = f"test zwei im {SLOT_WILDCARD}"
    assert score("test zwei im wohn zimmern", cand) >= 80


def test_score_slot_pattern_with_typo_in_fixed_parts() -> None:
    cand = f"test zwei im {SLOT_WILDCARD}"
    assert score("tst zwei im wohnzimmer", cand) >= 70


def test_score_slot_pattern_with_typo_in_slot_value() -> None:
    cand = f"test drei mit {SLOT_WILDCARD}"
    assert score("test drei mit chrlotte", cand) >= 80


def test_score_slot_pattern_rejects_unrelated() -> None:
    cand = f"test zwei im {SLOT_WILDCARD}"
    assert score("ich mag musik", cand) < 70


def test_find_best_picks_highest() -> None:
    cands = [
        Candidate(intent="A", pattern_idx=0, text="schalte das licht an"),
        Candidate(intent="B", pattern_idx=0, text="pumpe an"),
    ]
    res = find_best("pumpr an", cands, threshold=60)
    assert res is not None
    assert res[0].intent == "B"


def test_find_best_below_threshold() -> None:
    cands = [Candidate(intent="A", pattern_idx=0, text="hallo welt")]
    assert find_best("purple banana", cands, threshold=70) is None


def test_extract_slots_returns_empty_for_no_slots() -> None:
    cand = Candidate(intent="X", pattern_idx=0, text="pumpe an")
    assert extract_slots("pumpe an", cand) == []


def test_extract_slots_returns_raw_text() -> None:
    # No coercion: whatever lies between the surrounding fixed tokens
    # is captured verbatim. HA's Hassil resolves number words / digits
    # downstream when the canonical sentence is forwarded.
    cand = Candidate(
        intent="WetterStunde",
        pattern_idx=0,
        text=f"wie ist das wetter um {SLOT_WILDCARD} uhr",
        slot_names=["stunde"],
    )
    assert extract_slots("wie ist das wetter um zwölf uhr", cand) == ["zwölf"]
    assert extract_slots("wie ist das wetter um 14 uhr", cand) == ["14"]


def test_extract_slots_tolerates_typo_in_fixed_part() -> None:
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"test zwei im {SLOT_WILDCARD}",
        slot_names=["area"],
    )
    out = extract_slots("test zwei in büro", cand)
    assert out is not None and "büro" in out[0]


def test_extract_slots_tolerates_merged_tokens() -> None:
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"test zwei im {SLOT_WILDCARD}",
        slot_names=["area"],
    )
    out = extract_slots("test zwein büro", cand)
    assert out is not None and "büro" in out[0]


def test_extract_slots_multi_token_slot_value() -> None:
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"test zwei im {SLOT_WILDCARD}",
        slot_names=["area"],
    )
    assert extract_slots("test zwei im wohn zimmern", cand) == ["wohn zimmern"]


def test_extract_slots_preserves_user_casing() -> None:
    cand = Candidate(
        intent="Einkauf_Add",
        pattern_idx=0,
        text=f"add {SLOT_WILDCARD} to the shopping list",
        slot_names=["item"],
    )
    assert extract_slots("add Milk to the shopping list", cand) == ["Milk"]


def test_extract_slots_recovers_from_stt_split_token() -> None:
    cand = Candidate(
        intent="Einkauf_Add",
        pattern_idx=0,
        text=f"einkaufsliste {SLOT_WILDCARD}",
        slot_names=["item"],
    )
    assert extract_slots("einkaufslis ste veganes hack", cand) == ["veganes hack"]
    assert extract_slots("einkaufsli ste salami", cand) == ["salami"]
    assert extract_slots("einkaufslüsste veganes hack", cand) == ["veganes hack"]


def test_extract_slots_word_boundary_alignment() -> None:
    """
    Regression: slot boundaries land on whitespace, not mid-word.

    There's no good reason for ``"Einkaufsliste salami"`` to be split
    into ``"Einkaufslist"`` (prefix) and ``"e salami"`` (slot)
    """
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"einkaufsliste {SLOT_WILDCARD}",
        slot_names=["item"],
    )
    assert extract_slots("einkaufsliste salami", cand) == ["salami"]
    assert extract_slots("einkaufsliste vollmilch", cand) == ["vollmilch"]
    cand2 = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"{SLOT_WILDCARD} auf die einkaufsliste",
        slot_names=["item"],
    )
    assert extract_slots("salami auf die einkaufsliste", cand2) == ["salami"]
    assert extract_slots("vollmilch auf die einkaufsliste", cand2) == ["vollmilch"]


def test_extract_slots_two_slots() -> None:
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"wetter am {SLOT_WILDCARD} um {SLOT_WILDCARD} uhr",
        slot_names=["tag", "stunde"],
    )
    assert extract_slots("wetter am freitag um 12 uhr", cand) == ["freitag", "12"]


def test_build_canonical_passthrough_no_slots() -> None:
    cand = Candidate(intent="X", pattern_idx=0, text="pumpe an")
    assert build_canonical(cand, []) == "pumpe an"


def test_build_canonical_substitutes_slot() -> None:
    cand = Candidate(
        intent="WetterStunde",
        pattern_idx=0,
        text=f"wie ist das wetter um {SLOT_WILDCARD} uhr",
        slot_names=["stunde"],
    )
    assert build_canonical(cand, ["zwölf"]) == "wie ist das wetter um zwölf uhr"
    assert build_canonical(cand, ["14"]) == "wie ist das wetter um 14 uhr"


def test_build_canonical_handles_multiple_slots() -> None:
    cand = Candidate(
        intent="X",
        pattern_idx=0,
        text=f"a {SLOT_WILDCARD} b {SLOT_WILDCARD} c",
        slot_names=["x", "y"],
    )
    assert build_canonical(cand, ["foo", "bar"]) == "a foo b bar c"


def test_resolver_inlines_simple_rule() -> None:
    r = Resolver(expansion_rules={"gruss": ["hallo", "moin"]})
    assert r.inline_rules("<gruss> closest_intent") == "(hallo|moin) closest_intent"


def test_resolver_inlines_recursively() -> None:
    r = Resolver(
        expansion_rules={
            "outer": ["<inner> da", "anders"],
            "inner": ["hier", "dort"],
        }
    )
    out = r.inline_rules("<outer>")
    assert out == "((hier|dort) da|anders)"


def test_resolver_inlines_unknown_rule_unchanged() -> None:
    r = Resolver()
    assert r.inline_rules("<unknown> rest") == "<unknown> rest"


def test_expand_pattern_uses_resolver_rules() -> None:
    r = Resolver(expansion_rules={"gruss": ["hallo", "moin"]})
    out = expand_pattern("<gruss> closest_intent", cap=16, resolver=r)
    texts = [t for (t, _, _) in out]
    assert "hallo closest_intent" in texts
    assert "moin closest_intent" in texts


def test_resolver_resolves_exact_match() -> None:
    r = Resolver(slot_values={"area": ["Wohnzimmer", "Küche", "Büro"]})
    assert r.resolve_slot("Wohnzimmer", "area") == "Wohnzimmer"


def test_resolver_resolves_typo_to_closest() -> None:
    r = Resolver(slot_values={"area": ["Wohnzimmer", "Küche", "Büro"]})
    assert r.resolve_slot("wohnzma", "area") == "Wohnzimmer"


def test_resolver_returns_raw_when_no_close_match() -> None:
    r = Resolver(slot_values={"area": ["Wohnzimmer", "Küche", "Büro"]})
    assert r.resolve_slot("garage", "area") == "garage"


def test_resolver_returns_raw_for_unknown_list() -> None:
    r = Resolver()
    assert r.resolve_slot("anything", "no_such_list") == "anything"


def test_build_canonical_uses_resolver_for_slot_value() -> None:
    r = Resolver(slot_values={"area": ["Wohnzimmer", "Küche"]})
    cand = Candidate(
        intent="LightOn",
        pattern_idx=0,
        text=f"schalte das licht im {SLOT_WILDCARD} an",
        slot_names=["area"],
    )
    canonical = build_canonical(cand, ["wohnzma"], resolver=r)
    # Resolver returns the slot value with its registered casing,
    # so hassil sees `Wohnzimmer` rather than the user's lowercased speech.
    assert canonical == "schalte das licht im Wohnzimmer an"


def test_build_canonical_keeps_raw_when_resolver_finds_nothing() -> None:
    r = Resolver(slot_values={"area": ["Wohnzimmer"]})
    cand = Candidate(
        intent="LightOn",
        pattern_idx=0,
        text=f"licht im {SLOT_WILDCARD}",
        slot_names=["area"],
    )
    canonical = build_canonical(cand, ["garage"], resolver=r)
    assert canonical == "licht im garage"
