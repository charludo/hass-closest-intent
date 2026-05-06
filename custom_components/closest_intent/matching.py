"""
Hassil-pattern expansion + RapidFuzz scoring + slot extraction.
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
_ALT_RE = re.compile(r"\(([^()]+)\)")
_OPT_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Candidate:
    """One expanded sentence pattern, ready for scoring + slot extraction."""

    intent: str
    """Intent name (e.g. ``WetterStunde``)."""

    pattern_idx: int
    """Index into the intent's original pattern list (for debugging)."""

    text: str
    """Flattened text used for scoring. ``SLOT_WILDCARD`` stands in for slots."""

    slot_names: list[str] = field(default_factory=list)
    """
    Per Hassil's ``{LIST:CAPTURE}`` syntax, this is the *list* name in
    each slot position.
    """

    @property
    def has_slots(self) -> bool:
        return bool(self.slot_names)


def expand_pattern(pattern: str, cap: int) -> list[tuple[str, list[str]]]:
    """
    Expand a Hassil-style pattern into ``[(text, slot_lists), ...]``.

    Handles ``[optional]``, ``(a|b|c)``, ``{slot}``/``{slot:capture}``.
    """
    slot_lists: list[str] = []

    def _slot_sub(m: re.Match[str]) -> str:
        slot_lists.append(m.group(1))
        return f" {SLOT_WILDCARD} "

    pat = _SLOT_RE.sub(_slot_sub, pattern)

    if cap == 0:
        text = _ALT_RE.sub(lambda m: m.group(1).split("|")[0], pat)
        text = _OPT_RE.sub(lambda m: m.group(1), text)
        return [(_normalise(text), list(slot_lists))]

    variants: list[str] = [pat]
    while True:
        new_variants: list[str] = []
        changed = False
        for v in variants:
            m_alt = _ALT_RE.search(v)
            m_opt = _OPT_RE.search(v)
            chosen = None
            if m_alt and m_opt:
                chosen = m_alt if m_alt.start() < m_opt.start() else m_opt
            else:
                chosen = m_alt or m_opt
            if chosen is None:
                new_variants.append(v)
                continue
            changed = True
            before, after = v[: chosen.start()], v[chosen.end() :]
            opts = chosen.group(1).split("|") if chosen is m_alt else ["", chosen.group(1)]
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
        out.append((text, list(slot_lists)))
        if len(out) >= cap:
            break
    return out


def _normalise(s: str) -> str:
    """Lowercase, strip extra whitespace and trailing punctuation."""
    s = re.sub(r"\s+", " ", s).strip().lower()
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
      penalty (see ``_anchor_penalty``).
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


def find_best(
    user_text: str, candidates: Iterable[Candidate], threshold: int
) -> tuple[Candidate, int] | None:
    """Find the highest-scoring candidate above ``threshold``."""
    best: tuple[Candidate, int] | None = None
    for c in candidates:
        s = score(user_text, c.text)
        if s < threshold:
            continue
        if best is None or s > best[1]:
            best = (c, s)
    return best


_FIXED_PART_ALIGNMENT_THRESHOLD = 60

_STT_NOISE_TOKENS = frozenset(
    {
        "s",
        "t",
        "e",
        "r",
        "se",
        "te",
        "ne",
        "ge",
        "be",
        "ste",
        "ehm",
        "uhm",
        "äh",
        "ähm",
        "uh",
        "hmm",
    }
)


def _is_noise_token(t: str) -> bool:
    return (len(t) == 1 and t.isalpha()) or t.lower() in _STT_NOISE_TOKENS


def _strip_stt_noise(s: str) -> str:
    tokens = s.split()
    while tokens and _is_noise_token(tokens[0]):
        tokens.pop(0)
    while tokens and _is_noise_token(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


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
    and the end of the user text).

    Returns captured segments in left-to-right slot order, or ``None`` if
    alignment fails.
    """
    if not candidate.has_slots:
        return []

    parts = candidate.text.split(SLOT_WILDCARD)
    if len(parts) - 1 != len(candidate.slot_names):
        return None

    user = _normalise(user_text)
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

        captured.append(_strip_stt_noise(user[end_pos:slot_end].strip()))
        cursor = slot_end

    return captured


def build_canonical(candidate: Candidate, captured: list[str]) -> str:
    """Reconstruct a clean sentence from ``candidate`` with ``captured`` slot values."""
    if SLOT_WILDCARD not in candidate.text:
        return candidate.text
    parts = candidate.text.split(SLOT_WILDCARD)
    out: list[str] = [parts[0]]
    for i, raw in enumerate(captured):
        out.append(raw)
        out.append(parts[i + 1])
    return _normalise("".join(out))
