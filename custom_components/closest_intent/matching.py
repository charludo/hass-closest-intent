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
    match_threshold: int = 70
    slot_resolution_threshold: int = 70
    _unknown_rules_seen: set[str] = field(default_factory=set, repr=False)

    def inline_rules(self, pattern: str) -> str:
        """Replace ``<rule>`` references in ``pattern`` with ``(form1|form2|...)``.

        Recursive!! Unknown rules fall back to wildcard slots with the same name.
        """
        seen_in_chain: set[str] = set()
        return self._inline_rules_inner(pattern, seen_in_chain, depth=0)

    def _inline_rules_inner(self, pattern: str, seen: set[str], depth: int) -> str:
        if depth > 10:
            return pattern  # cycle guard

        def sub(m: re.Match[str]) -> str:
            rule = m.group(1)
            if rule in seen:
                # In a recursion chain -- leave as-is to avoid infinite loop.
                return m.group(0)
            forms = self.expansion_rules.get(rule)
            if not forms:
                # Unknown rules fall back to wildcard slot instead of dropping altogether.
                if rule not in self._unknown_rules_seen:
                    self._unknown_rules_seen.add(rule)
                    _LOGGER.debug(
                        "unknown expansion rule <%s>, substituting wildcard slot {%s}",
                        rule,
                        rule,
                    )
                return "{" + rule + "}"
            inner = "(" + "|".join(forms) + ")"
            return self._inline_rules_inner(inner, seen | {rule}, depth + 1)

        return _RULE_RE.sub(sub, pattern)

    def resolve_slot(self, captured: str, list_name: str | None) -> str:
        """Fuzz-match ``captured`` against the ``list_name`` values.

        Returns the closest known value if it scores above ``self.slot_resolution_threshold``.
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

        threshold = self.slot_resolution_threshold
        best: str | None = None
        best_score = 0
        for v in values:
            s = int(fuzz.token_sort_ratio(captured_norm, v.lower(), score_cutoff=threshold))
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


_INNER_SLOT_RE = re.compile(r"\x00slot:([a-zA-Z_][a-zA-Z0-9_]*)\x00")


def _inner_slot_marker(slot_name: str) -> str:
    return f"\x00slot:{slot_name}\x00"


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
    Also handles nested slot expansion only to the relevant variants.
    """
    if resolver is not None:
        pattern = resolver.inline_rules(pattern)

    def _slot_sub(m: re.Match[str]) -> str:
        return f" {_inner_slot_marker(m.group(1))} "

    pat = _SLOT_RE.sub(_slot_sub, pattern)

    def _finalise(v: str) -> tuple[str, str, list[str]]:
        """Pull per-variant slot names, then rewrite markers to SLOT_WILDCARD."""
        variant_slot_names = _INNER_SLOT_RE.findall(v)
        v_canonical = _INNER_SLOT_RE.sub(SLOT_WILDCARD, v)
        return _normalise(v_canonical), _normalise_keepcase(v_canonical), variant_slot_names

    if cap == 0:
        text = _ALT_RE.sub(lambda m: m.group(1).split("|")[0], pat)
        text = _OPT_RE.sub(lambda m: m.group(1).split("|")[0], text)
        return [_finalise(text)]

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
        text, display_text, variant_slot_names = _finalise(v)
        if text in seen:
            continue
        seen.add(text)
        out.append((text, display_text, variant_slot_names))
        if len(out) >= cap:
            break
    return out


_MATCH_PUNCT_RE = re.compile(r"[^\w\s\x00]", re.UNICODE)


def _normalise(s: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return _normalise_for_capture(s).lower()


def _normalise_for_capture(s: str) -> str:
    """Replace punctuation with spaces, collapse whitespace."""
    s = _MATCH_PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalise_keepcase(s: str) -> str:
    """Used for the candidate's ``display_text``, passed on to hassil."""
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

    Mid-word alignments are rejected.
    """
    if not anchor:
        return 0
    if not user_norm:
        return None
    align = fuzz.partial_ratio_alignment(
        anchor, user_norm, score_cutoff=_ANCHOR_ALIGNMENT_MIN_SCORE
    )
    if align is None or align.score < _ANCHOR_ALIGNMENT_MIN_SCORE:
        return None
    if from_end:
        if align.dest_end < len(user_norm) and user_norm[align.dest_end].isalnum():
            return None
        tail = user_norm[align.dest_end :]
        return len(tail.split())
    if align.dest_start > 0 and user_norm[align.dest_start - 1].isalnum():
        return None
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


_SLOT_TOKEN_BUDGET = 3
_SLOT_PROPORTION_PENALTY_PER_TOKEN = 10
_SLOT_PROPORTION_PENALTY_CAP = 40


def _slot_proportion_penalty(cand_stripped: str, user_norm: str, slot_count: int) -> int:
    """Penalty for slotted candidates whose fixed text is only a small part of the user input."""
    if not cand_stripped or slot_count == 0:
        return 0
    fixed_tokens = len(cand_stripped.split())
    user_tokens = len(user_norm.split())
    expected_tokens = fixed_tokens + slot_count * _SLOT_TOKEN_BUDGET
    excess = user_tokens - expected_tokens
    if excess <= 0:
        return 0
    return min(_SLOT_PROPORTION_PENALTY_CAP, excess * _SLOT_PROPORTION_PENALTY_PER_TOKEN)


def score(user_text: str, candidate_text: str, resolver: Resolver | None = None) -> int:
    """
    Similarity 0..100 with the slot wildcard ignored.

    Two regimes, picked by whether the candidate contains slot positions:

    - **No slots**: ``token_sort_ratio`` on the whole phrase.
    - **With slots**: ``partial_ratio`` on the candidate's *fixed parts*
      against the full user text, minus an edge-anchor misalignment penalty
    """
    user_norm = _normalise(user_text)
    cand_stripped = re.sub(r"\s+", " ", candidate_text.replace(SLOT_WILDCARD, " ")).strip()
    threshold = resolver.match_threshold if resolver is not None else 70

    if SLOT_WILDCARD in candidate_text:
        if not cand_stripped:
            return 0
        base = int(fuzz.partial_ratio(user_norm, cand_stripped, score_cutoff=threshold))
        parts = candidate_text.split(SLOT_WILDCARD)
        penalty = _anchor_penalty(parts, user_norm)
        penalty += _slot_proportion_penalty(cand_stripped, user_norm, len(parts) - 1)
        return max(0, base - penalty)

    ts = int(fuzz.token_sort_ratio(user_norm, cand_stripped))
    r = int(fuzz.ratio(user_norm, cand_stripped))
    return max(ts, r)


_FIND_BEST_TIEBREAK_BAND = 15
_FIND_BEST_SLOT_COUNT_TIEBREAK_BAND = 5


def _fixed_text_length(candidate_text: str) -> int:
    """Length of the candidate's non-slot text, whitespace-collapsed."""
    stripped = candidate_text.replace(SLOT_WILDCARD, " ")
    return len(re.sub(r"\s+", " ", stripped).strip())


def _slot_count(candidate_text: str) -> int:
    return candidate_text.count(SLOT_WILDCARD)


def find_best(
    user_text: str, candidates: Iterable[Candidate], resolver: Resolver
) -> tuple[Candidate, int] | None:
    """
    Find the best candidate above ``threshold``.

    - First, find highest-scoring candidate.
    - Then, among the top performing ones (within ``_FIND_BEST_TIEBREAK_BAND`` of the top score),
      prefer fewer slots, then longer fixed text, then higher score.
      Tie-break rejects siblings whose slot at a leading or trailing boundary absorbs material
      a more-anchored sibling would treat as a fixed prefix/suffix, e.g.
      ``put {item} on the shopping list`` over the bare ``{item} on the shopping list``
      when the user actually said 'put'.
    """
    threshold = resolver.match_threshold
    scored: list[tuple[Candidate, int]] = []
    for c in candidates:
        s = score(user_text, c.text, resolver)
        if s < threshold:
            continue
        scored.append((c, s))
    if not scored:
        return None

    scored.sort(key=lambda cs: -cs[1])
    top_score = scored[0][1]

    tight_floor = top_score - _FIND_BEST_SLOT_COUNT_TIEBREAK_BAND
    tight = [cs for cs in scored if cs[1] >= tight_floor]
    if len(tight) > 1 and len({_slot_count(c.text) for (c, _) in tight}) > 1:
        tight.sort(
            key=lambda cs: (
                _slot_count(cs[0].text),
                -_fixed_text_length(cs[0].text),
                -cs[1],
            )
        )
        return tight[0]

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


def _word_boundary_starts(sub: str, e: int, max_words: int = _MAX_BOUNDARY_LOOKAHEAD) -> list[int]:
    """Start positions in ``sub[:e]`` on word boundaries."""
    pos = e
    out: list[int] = []
    while pos > 0 and len(out) < max_words:
        prev_space = sub.rfind(" ", 0, pos)
        if prev_space == -1:
            out.append(0)
            break
        out.append(prev_space + 1)
        pos = prev_space
    return out


def _is_word_boundary_start(sub: str, pos: int) -> bool:
    return pos == 0 or not sub[pos - 1].isalnum()


def _is_word_boundary_end(sub: str, pos: int) -> bool:
    return pos == len(sub) or not sub[pos].isalnum()


_MID_WORD_ALIGNMENT_PENALTY = 25


def _aligned_score(fixed: str, sub: str, s: int, e: int) -> int:
    score = int(fuzz.ratio(fixed, sub[s:e]))
    if not (_is_word_boundary_start(sub, s) and _is_word_boundary_end(sub, e)):
        score -= _MID_WORD_ALIGNMENT_PENALTY
    return score


def _align_fixed_part(fixed: str, user: str, start: int) -> tuple[int, int] | None:
    """
    Find where ``fixed`` approximately occurs in ``user[start:]``.

    Two-stage alignment:
      1. ``partial_ratio_alignment`` finds a starting point with merged-token tolerance
      2. We then enumerate word-boundary start/end positions, picking the (start, end) pair
         with the highest ``fuzz.ratio`` against ``fixed``.

    Slot captures end up on token boundaries unless the input is genuinely mid-word.
    Mid-word alignments are penalized so that a clean word-boundary match wins over a
    perfect-but-incidental substring match.
    """
    sub = user[start:]
    if not fixed:
        return (start, start)
    if not sub:
        return None
    alignment = fuzz.partial_ratio_alignment(fixed, sub)
    if alignment is None or alignment.score < _FIXED_PART_ALIGNMENT_THRESHOLD:
        return None
    best_start = alignment.dest_start
    best_end = alignment.dest_end
    best_score = _aligned_score(fixed, sub, best_start, best_end)
    for cand_end in _word_boundary_ends(sub, best_start):
        score = _aligned_score(fixed, sub, best_start, cand_end)
        if score > best_score:
            best_end, best_score = cand_end, score
    for cand_start in _word_boundary_starts(sub, alignment.dest_start):
        score = _aligned_score(fixed, sub, cand_start, best_end)
        if score > best_score:
            best_start, best_score = cand_start, score
    return (start + best_start, start + best_end)


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
    user_display = _normalise_for_capture(user_text)
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
) -> str:
    """
    Reconstruct a clean, case-preserving sentence from ``candidate`` with slot values.

    If ``resolver`` is supplied, each captured slot value is fuzz-matched
    against the slot's known values (``resolver.slot_values[list_name]``)
    and replaced with the closest known value when one scores above
    ``resolver.slot_resolution_threshold``. Otherwise (or when nothing
    scores high enough) the user's raw spoken text is preserved.
    """
    template = candidate.display_text or candidate.text
    if SLOT_WILDCARD not in template:
        return template
    parts = template.split(SLOT_WILDCARD)
    out: list[str] = [parts[0]]
    for i, raw in enumerate(captured):
        list_name = candidate.slot_names[i] if i < len(candidate.slot_names) else None
        value = resolver.resolve_slot(raw, list_name) if resolver is not None else raw
        out.append(value)
        out.append(parts[i + 1])
    return _normalise_keepcase("".join(out))
