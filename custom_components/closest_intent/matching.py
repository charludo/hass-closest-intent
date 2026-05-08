"""
Hassil-pattern expansion + RapidFuzz scoring + slot extraction.

Optionally augmented by a :class:`Resolver` that holds Hassil expansion
rules (``<rule>`` references) and slot-list values (``{list}`` look-ups).
When passed in, patterns get richer pre-expansion (so user patterns that
reference HA built-in rules like ``<set>`` actually score correctly)
and captured slot text gets fuzz-resolved against the slot list
(e.g. ``"livg ruom"`` becomes ``"Living Room"`` before being substituted
 into the canonical sentence).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

try:
    from .const import SLOT_WILDCARD
except ImportError:  # pragma: no cover
    from const import SLOT_WILDCARD  # type: ignore

_LOGGER = logging.getLogger(__name__)

_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[a-zA-Z_][a-zA-Z0-9_]*)?\}")
_RULE_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>")
_ALT_RE = re.compile(r"\(([^()]+)\)")
_OPT_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Resolver:
    """Pre-computed pools for ``<rule>`` and ``{list}`` references."""

    expansion_rules: dict[str, list[str]] = field(default_factory=dict)
    slot_values: dict[str, list[str]] = field(default_factory=dict)

    def inline_rules(self, pattern: str) -> str:
        """Replace ``<rule>`` references in ``pattern`` with ``(form1|form2|...)``.

        Recursive!! Undefined rules ignored.
        """
        seen_in_chain: set[str] = set()
        return self._inline_rules_inner(pattern, seen_in_chain, depth=0)

    def _inline_rules_inner(self, pattern: str, seen: set[str], depth: int) -> str:
        if depth > 10:
            return pattern  # cycle guard

        def sub(m: re.Match[str]) -> str:
            rule = m.group(1)
            if rule in seen or rule not in self.expansion_rules:
                return m.group(0)
            forms = self.expansion_rules[rule]
            if not forms:
                return m.group(0)
            inner = "(" + "|".join(forms) + ")"
            return self._inline_rules_inner(inner, seen | {rule}, depth + 1)

        return _RULE_RE.sub(sub, pattern)

    def resolve_slot(self, captured: str, list_name: str | None, threshold: int = 70) -> str:
        """Fuzz-match ``captured`` against the ``list_name`` values.

        Returns the closest known value if it scores above ``threshold``.
        Otherwise, returns ``captured`` unchanged so the canonical sentence
        carries through the user's original speech (and Hassil downstream
        either resolves it via its own rules or politely fails).
        """
        if not captured or not list_name:
            return captured
        values = self.slot_values.get(list_name)
        if not values:
            return captured

        captured_norm = captured.strip().lower()
        for v in values:
            if v.lower() == captured_norm:
                return v

        best: str | None = None
        best_score = 0
        for v in values:
            s = int(fuzz.token_sort_ratio(captured_norm, v.lower()))
            if s > best_score:
                best, best_score = v, s
        if best is not None and best_score >= threshold:
            return best
        return captured


@dataclass
class Candidate:
    """One expanded sentence pattern, ready for scoring + slot extraction."""

    intent: str
    """Intent name (e.g. ``WetterStunde``)."""

    pattern_idx: int
    """Index into the intent's original pattern list (for debugging)."""

    text: str
    """Flattened text used for scoring. ``SLOT_WILDCARD`` stands in for slots.

    Lowercased and whitespace-collapsed.
    """

    display_text: str = ""
    """
    Same flattened pattern as ``text`` but with the intent author's
    original casing preserved (still whitespace-collapsed).

    Used by ``build_canonical`` so the sentence forwarded to hassil keeps
    case-sensitive tokens intact. Defaults to ``text`` when a ``Candidate``
    is built without an explicit display form.
    """

    slot_names: list[str] = field(default_factory=list)
    """
    Per Hassil's ``{LIST:CAPTURE}`` syntax, this is the *list* name in
    each slot position. Used to look up resolver values.
    HA's downstream capture-name (CAPTURE in the pattern) is its own concern.
    """

    @property
    def has_slots(self) -> bool:
        return bool(self.slot_names)


def expand_pattern(
    pattern: str,
    cap: int,
    resolver: Resolver | None = None,
) -> list[tuple[str, str, list[str]]]:
    """
    Expand a Hassil-style pattern into ``[(text, display_text, slot_lists), ...]``.

    Handles ``[optional]``, ``(a|b|c)``, ``{slot}``/``{slot:capture}`` and,
    if a ``resolver`` is supplied, ``<rule>`` references (inlined into
    alternatives before ordinary expansion runs).
    """
    if resolver is not None:
        pattern = resolver.inline_rules(pattern)

    slot_lists: list[str] = []

    def _slot_sub(m: re.Match[str]) -> str:
        slot_lists.append(m.group(1))
        return f" {SLOT_WILDCARD} "

    pat = _SLOT_RE.sub(_slot_sub, pattern)

    if cap == 0:
        text = _ALT_RE.sub(lambda m: m.group(1).split("|")[0], pat)
        text = _OPT_RE.sub(lambda m: m.group(1).split("|")[0], text)
        return [(_normalise(text), _normalise_keepcase(text), list(slot_lists))]

    variants: list[str] = [pat]
    while True:
        new_variants: list[str] = []
        changed = False
        for v in variants:
            m_alt = _ALT_RE.search(v)
            m_opt = _OPT_RE.search(v)
            if m_alt and m_opt:
                chosen = m_alt if m_alt.start() < m_opt.start() else m_opt
            else:
                chosen = m_alt or m_opt
            if chosen is None:
                new_variants.append(v)
                continue
            changed = True
            before, after = v[: chosen.start()], v[chosen.end() :]

            # ``[a|b]`` is semantically equivalent to ``(|a|b)``
            opts = chosen.group(1).split("|")
            if chosen is m_opt:
                opts = ["", *opts]
            for o in opts:
                new_variants.append(before + o + after)
            if len(new_variants) >= cap:
                break
        variants = new_variants[:cap]
        if not changed:
            break

    out = []
    seen: set[str] = set()
    for v in variants:
        text = _normalise(v)
        if text in seen:
            continue
        seen.add(text)
        out.append((text, _normalise_keepcase(v), list(slot_lists)))
        if len(out) >= cap:
            break
    return out


def _normalise(s: str) -> str:
    """Lowercase, strip extra whitespace and trailing punctuation."""
    return _normalise_keepcase(s).lower()


def _normalise_keepcase(s: str) -> str:
    """Strip extra whitespace and trailing punctuation. Preserve case."""
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip("?.!,;:")
    return s


# Slot patterns: how far from the relevant edge of the user text a
# fixed anchor is allowed to land before we start charging penalty.
# One token of leading STT-noise is fine (``"uhm add bread..."``);
# beyond that, the candidate's fixed prefix doesn't actually anchor
# the user phrase and we must downscore.
_ANCHOR_NOISE_TOLERANCE_TOKENS = 1
# Penalty per misaligned token. Two extra tokens before/after a fixed
# anchor enough to push a 100-scoring candidate below the default 70
# threshold.
_ANCHOR_MISALIGN_PENALTY_PER_TOKEN = 25
# When a non-empty leading/trailing anchor doesn't appear in user text
# at all (no alignment passes this score), candidate is structurally
# wrong: charge a flat penalty equivalent to ~2 misaligned tokens.
_ANCHOR_ABSENT_PENALTY = 50
_ANCHOR_ALIGNMENT_MIN_SCORE = 60


def _anchor_offset_tokens(anchor: str, user_norm: str, *, from_end: bool) -> int | None:
    """
    Return token count between ``anchor``'s alignment and the
    relevant edge of ``user_norm``, or ``None`` if no usable alignment.

    ``from_end=False`` measures tokens before the anchor,
    ``from_end=True`` measures tokens after the anchor's end.
    """
    if not anchor:
        return 0
    if not user_norm:
        return None
    align = fuzz.partial_ratio_alignment(anchor, user_norm)
    if align is None or align.score < _ANCHOR_ALIGNMENT_MIN_SCORE:
        return None
    if from_end:
        tail = user_norm[align.dest_end :]
        return len(tail.split())
    head = user_norm[: align.dest_start]
    return len(head.split())


def _anchor_penalty(parts: list[str], user_norm: str) -> int:
    """
    Sum of edge-anchor misalignment penalties for a slot pattern.

    A pattern like ``"shopping list {item}"`` requires "shopping list"
    at (or very near) the start of user input. If it lands several
    tokens deep, the candidate doesn't actually fit the user text shape,
    even though the substring is present and ``partial_ratio`` happily scores 100.

    Same idea at the trailing edge for ``"{item} to the shopping list"``.

    Patterns with a slot at the boundary (empty leading/trailing fixed text)
    are unconstrained at that edge, since the slot can soak up
    arbitrary content there.
    """
    leading = parts[0].strip() if parts else ""
    trailing = parts[-1].strip() if len(parts) > 1 else ""
    penalty = 0

    if leading:
        offset = _anchor_offset_tokens(leading, user_norm, from_end=False)
        if offset is None:
            penalty += _ANCHOR_ABSENT_PENALTY
        else:
            extra = max(0, offset - _ANCHOR_NOISE_TOLERANCE_TOKENS)
            penalty += extra * _ANCHOR_MISALIGN_PENALTY_PER_TOKEN

    if trailing:
        offset = _anchor_offset_tokens(trailing, user_norm, from_end=True)
        if offset is None:
            penalty += _ANCHOR_ABSENT_PENALTY
        else:
            extra = max(0, offset - _ANCHOR_NOISE_TOLERANCE_TOKENS)
            penalty += extra * _ANCHOR_MISALIGN_PENALTY_PER_TOKEN

    return penalty


def score(user_text: str, candidate_text: str) -> int:
    """
    Similarity 0..100 with the slot wildcard ignored.

    Two regimes, picked by whether the candidate contains slot positions:

    - **No slots**: ``token_sort_ratio`` on the whole phrase.
    - **With slots**: ``partial_ratio`` on the candidate's *fixed parts*
      against the full user text, minus an edge-anchor misalignment
      penalty (see ``_anchor_penalty``). Finds the best contiguous window
      of the fixed parts within the user input, but rejects matches
      where the leading/trailing fixed anchor doesn't actually sit at
      the corresponding edge of the user phrase.
    """
    user_norm = _normalise(user_text)
    cand_stripped = re.sub(r"\s+", " ", candidate_text.replace(SLOT_WILDCARD, " ")).strip()

    if SLOT_WILDCARD in candidate_text:
        if not cand_stripped:
            return 0
        base = int(fuzz.partial_ratio(user_norm, cand_stripped))
        parts = candidate_text.split(SLOT_WILDCARD)
        penalty = _anchor_penalty(parts, user_norm)
        return max(0, base - penalty)

    return int(fuzz.token_sort_ratio(user_norm, cand_stripped))


_FIND_BEST_TIEBREAK_BAND = 15


def _fixed_text_length(candidate_text: str) -> int:
    """Length of the candidate's non-slot text, whitespace-collapsed."""
    stripped = candidate_text.replace(SLOT_WILDCARD, " ")
    return len(re.sub(r"\s+", " ", stripped).strip())


def find_best(
    user_text: str, candidates: Iterable[Candidate], threshold: int
) -> tuple[Candidate, int] | None:
    """
    Find the best candidate above ``threshold``.

    First, find highest-scoring candidate.
    Then, among the top performing ones, find the one with the shortest slot-text.
    Tie-break rejects siblings whose slot at a leading/trailing
    boundary absorbs material a more-anchored sibling would treat as
    a fixed prefix/suffix, e.g. ``put {item} on the shopping list`` over the bare
    ``{item} on the shopping list`` when the user actually said 'put'.
    ``partial_ratio`` would happily scores the bare one at 100
    because its fixed tail is a substring.
    """
    scored: list[tuple[Candidate, int]] = []
    for c in candidates:
        s = score(user_text, c.text)
        if s < threshold:
            continue
        scored.append((c, s))
    if not scored:
        return None

    scored.sort(key=lambda cs: -cs[1])
    top_score = scored[0][1]
    band_floor = top_score - _FIND_BEST_TIEBREAK_BAND
    contenders = [cs for cs in scored if cs[1] >= band_floor]
    if len(contenders) == 1:
        return contenders[0]

    if all(SLOT_WILDCARD in c.text for (c, _) in contenders):
        contenders.sort(key=lambda cs: (-_fixed_text_length(cs[0].text), -cs[1]))
    return contenders[0]


_FIXED_PART_ALIGNMENT_THRESHOLD = 60

_MAX_BOUNDARY_LOOKAHEAD = 8


def _word_boundary_ends(sub: str, s: int, max_words: int = _MAX_BOUNDARY_LOOKAHEAD) -> list[int]:
    """
    End positions in ``sub[s:]`` that fall on word boundaries (each space, plus end-of-string).
    Capped at ``max_words`` so search stays cheap regardless of input length.
    """
    pos = s
    out: list[int] = []
    while pos < len(sub) and len(out) < max_words:
        next_space = sub.find(" ", pos)
        if next_space == -1:
            out.append(len(sub))
            break
        out.append(next_space)
        pos = next_space + 1
    return out


def _align_fixed_part(fixed: str, user: str, start: int) -> tuple[int, int] | None:
    """
    Find where ``fixed`` approximately occurs in ``user[start:]``.

    Two-stage alignment:
      1. ``partial_ratio_alignment`` finds a starting point with merged-token tolerance
      2. We then enumerate word-boundary end positions in the input,
         and pick the one with the highest ``fuzz.ratio`` against ``fixed``.

    Slot captures end up on token boundaries unless the input is genuinely mid-word.
    """
    sub = user[start:]
    if not fixed:
        return (start, start)
    if not sub:
        return None
    alignment = fuzz.partial_ratio_alignment(fixed, sub)
    if alignment is None or alignment.score < _FIXED_PART_ALIGNMENT_THRESHOLD:
        return None
    s = alignment.dest_start
    best_end = alignment.dest_end
    best_score = fuzz.ratio(fixed, sub[s:best_end])
    for cand_end in _word_boundary_ends(sub, s):
        score = fuzz.ratio(fixed, sub[s:cand_end])
        if score > best_score:
            best_end, best_score = cand_end, score
    return (start + s, start + best_end)


def extract_slots(user_text: str, candidate: Candidate) -> list[str] | None:
    """
    Pull slot values out of ``user_text`` aligned to ``candidate``.

    Character-level fuzzy alignment of each fixed part. Slot value is
    whatever lies between adjacent fixed parts (or between a fixed part
    and the end of the user text). Imperfect captures (extra leading
    chars from a misaligned boundary) get cleaned up downstream by
    ``Resolver.resolve_slot`` fuzz-matching against the slot's known
    values.

    Returns captured segments in left-to-right slot order, or ``None`` if
    alignment fails.
    """
    if not candidate.has_slots:
        return []

    parts = candidate.text.split(SLOT_WILDCARD)
    if len(parts) - 1 != len(candidate.slot_names):
        return None

    # Try to align on case-preserving display string.
    # In the rare case where lower/upper case have different amount of utf8 chars
    # (e.g. Turkish ``İ``) fall back to the lowercased capture for that slot.
    user_display = _normalise_keepcase(user_text)
    user = user_display.lower()
    indices_aligned = len(user) == len(user_display)
    cursor = 0
    captured: list[str] = []

    for i, prefix in enumerate(parts[:-1]):
        prefix_norm = " ".join(prefix.split())
        span = _align_fixed_part(prefix_norm, user, cursor)
        if span is None:
            return None
        end_pos = span[1]

        next_norm = " ".join(parts[i + 1].split())
        if next_norm:
            next_span = _align_fixed_part(next_norm, user, end_pos)
            slot_end = next_span[0] if next_span else len(user)
        else:
            slot_end = len(user)

        source = user_display if indices_aligned else user
        captured.append(source[end_pos:slot_end].strip())
        cursor = slot_end

    return captured


def build_canonical(
    candidate: Candidate,
    captured: list[str],
    resolver: Resolver | None = None,
    slot_resolution_threshold: int = 70,
) -> str:
    """
    Reconstruct a clean, case-preserving sentence from ``candidate`` with slot values.

    If ``resolver`` is supplied, each captured slot value is fuzz-matched
    against the slot's known values (``resolver.slot_values[list_name]``)
    and replaced with the closest known value when one scores above ``slot_resolution_threshold``.
    Otherwise (or when nothing scores high enough) the user's raw spoken text is preserved.
    """
    template = candidate.display_text or candidate.text
    if SLOT_WILDCARD not in template:
        return template
    parts = template.split(SLOT_WILDCARD)
    out: list[str] = [parts[0]]
    for i, raw in enumerate(captured):
        list_name = candidate.slot_names[i] if i < len(candidate.slot_names) else None
        if resolver is not None:
            value = resolver.resolve_slot(raw, list_name, slot_resolution_threshold)
        else:
            value = raw
        out.append(value)
        out.append(parts[i + 1])
    return _normalise_keepcase("".join(out))
