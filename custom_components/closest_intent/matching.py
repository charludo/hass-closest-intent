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


_MIN_ALIGNMENT_SCORE = 60
_DEST_COVERAGE_MIN_RATIO = 0.6


def score(
    user_text: str,
    candidate_text: str,
    resolver: Resolver | None = None,
    slot_names: list[str] | None = None,
) -> int:
    """
    Similarity 0..100 with the slot wildcard ignored.

    Two regimes, picked by whether the candidate contains slot positions.
    """
    user_norm = _normalise(user_text)
    cand_stripped = re.sub(r"\s+", " ", candidate_text.replace(SLOT_WILDCARD, " ")).strip()

    if SLOT_WILDCARD in candidate_text:
        if not cand_stripped:
            return 0
        return _slotted_score(
            candidate_text.split(SLOT_WILDCARD),
            user_norm,
            slot_names=slot_names,
            resolver=resolver,
        )

    ts = int(fuzz.token_sort_ratio(user_norm, cand_stripped))
    r = int(fuzz.ratio(user_norm, cand_stripped))
    return max(ts, r)


def _slot_credit(slot_text: str, slot_name: str | None, resolver: Resolver | None) -> int:
    """How well ``slot_text`` matches the slot's known list values."""
    if not slot_text:
        return 0
    if resolver is None or not slot_name:
        return 100
    values = resolver.slot_values.get(slot_name)
    if not values:
        return 100
    best = 0
    for v in values:
        r = int(fuzz.ratio(slot_text, _normalise(v)))
        if r > best:
            best = r
            if best >= 100:
                break
    return best


def _slotted_score(
    parts: list[str],
    user_norm: str,
    slot_names: list[str] | None = None,
    resolver: Resolver | None = None,
) -> int:
    user_len = len(user_norm)
    if user_len == 0:
        return 0

    last_part = len(parts) - 1
    cursor = 0
    aligned: list[tuple[int, int, int, int]] = []  # (part_index, dest_start, dest_end, score)
    for i, part in enumerate(parts):
        seg = part.strip()
        if not seg:
            continue
        sub = user_norm[cursor:]
        if not sub:
            continue
        a = fuzz.partial_ratio_alignment(seg, sub)
        if a is None or a.score < _MIN_ALIGNMENT_SCORE:
            continue
        ds = cursor + a.dest_start
        de = cursor + a.dest_end
        # reject alignments where most of the needle isn't present in user text
        if (de - ds) < _DEST_COVERAGE_MIN_RATIO * len(seg):
            continue
        need_wb_start = i < last_part
        need_wb_end = i > 0
        if " " not in user_norm[ds:de]:
            if need_wb_start and not _is_word_boundary_start(user_norm, ds):
                continue
            if need_wb_end and not _is_word_boundary_end(user_norm, de):
                continue
        aligned.append((i, ds, de, int(a.score)))
        cursor = de

    aligned_idx = {a[0] for a in aligned}
    missing_segments = [
        p.strip() for i, p in enumerate(parts) if p.strip() and i not in aligned_idx
    ]
    if not aligned and missing_segments:
        return 0

    numerator = sum(sc for _, _, _, sc in aligned)
    n_segments = len(aligned) + len(missing_segments)

    seen: set[tuple[int, int]] = set()
    n_slots = len(parts) - 1
    for k in range(n_slots):
        left = 0
        for a in reversed(aligned):
            if a[0] <= k:
                left = a[2]
                break
        right = user_len
        for a in aligned:
            if a[0] >= k + 1:
                right = a[1]
                break
        if (left, right) in seen:
            continue
        seen.add((left, right))
        n_segments += 1
        if right > left:
            slot_text = user_norm[left:right].strip()
            slot_name = slot_names[k] if slot_names and k < len(slot_names) else None
            numerator += _slot_credit(slot_text, slot_name, resolver)

    if n_segments <= 0:
        return 0
    return min(100, numerator // n_segments)


_FIND_BEST_TIEBREAK_BAND = 15
_TOKEN_PRESENCE_MIN_RATIO = 60


def _absent_fixed_token_count(cand_text: str, user_tokens: list[str]) -> int:
    """Number of candidate fixed tokens with no above-threshold match in user."""
    stripped = cand_text.replace(SLOT_WILDCARD, " ")
    absent = 0
    for tok in stripped.split():
        best = 0
        for ut in user_tokens:
            r = int(fuzz.ratio(tok, ut))
            if r > best:
                best = r
                if best >= 100:
                    break
        if best < _TOKEN_PRESENCE_MIN_RATIO:
            absent += 1
    return absent


def find_best(
    user_text: str, candidates: Iterable[Candidate], resolver: Resolver
) -> tuple[Candidate, int] | None:
    """Highest-scoring candidate above ``resolver.match_threshold``, or ``None``."""
    threshold = resolver.match_threshold
    scored: list[tuple[Candidate, int]] = []
    for c in candidates:
        s = score(user_text, c.text, resolver, slot_names=c.slot_names)
        if s >= threshold:
            scored.append((c, s))
    if not scored:
        return None

    scored.sort(key=lambda cs: -cs[1])
    band_floor = scored[0][1] - _FIND_BEST_TIEBREAK_BAND
    contenders = [cs for cs in scored if cs[1] >= band_floor]
    if len(contenders) == 1:
        return contenders[0]

    user_norm = _normalise(user_text)
    user_tokens = user_norm.split()

    def _key(cs: tuple[Candidate, int]) -> tuple[int, int, int, int]:
        absent = _absent_fixed_token_count(cs[0].text, user_tokens)
        fixed = re.sub(r"\s+", " ", cs[0].text.replace(SLOT_WILDCARD, " ")).strip()
        ts = int(fuzz.token_sort_ratio(user_norm, fixed))
        return (absent, -ts, -cs[1], -len(fixed))

    contenders.sort(key=_key)
    return contenders[0]


def _is_word_boundary_start(sub: str, pos: int) -> bool:
    return pos == 0 or not sub[pos - 1].isalnum()


def _is_word_boundary_end(sub: str, pos: int) -> bool:
    return pos == len(sub) or not sub[pos].isalnum()


_MAX_BOUNDARY_LOOKAHEAD = 8
_MID_WORD_ALIGNMENT_PENALTY = 25


def _align_fixed_part(fixed: str, user: str, start: int) -> tuple[int, int] | None:
    """Find where ``fixed`` approximately occurs in ``user[start:]``."""
    if not fixed:
        return (start, start)
    sub = user[start:]
    if not sub:
        return None
    align = fuzz.partial_ratio_alignment(fixed, sub)
    if align is None or align.score < _MIN_ALIGNMENT_SCORE:
        return None

    def _score_window(s: int, e: int) -> int:
        sc = int(fuzz.ratio(fixed, sub[s:e]))
        if not (_is_word_boundary_start(sub, s) and _is_word_boundary_end(sub, e)):
            sc -= _MID_WORD_ALIGNMENT_PENALTY
        return sc

    def _boundaries(pivot: int, forward: bool) -> Iterable[int]:
        """Up to ``_MAX_BOUNDARY_LOOKAHEAD`` word boundaries in one direction."""
        out: list[int] = []
        pos = pivot
        if forward:
            while pos < len(sub) and len(out) < _MAX_BOUNDARY_LOOKAHEAD:
                nxt = sub.find(" ", pos)
                if nxt == -1:
                    out.append(len(sub))
                    break
                out.append(nxt)
                pos = nxt + 1
        else:
            while pos > 0 and len(out) < _MAX_BOUNDARY_LOOKAHEAD:
                prev = sub.rfind(" ", 0, pos)
                if prev == -1:
                    out.append(0)
                    break
                out.append(prev + 1)
                pos = prev
        return out

    best_start, best_end = align.dest_start, align.dest_end
    best = _score_window(best_start, best_end)
    for e in _boundaries(best_start, forward=True):
        sc = _score_window(best_start, e)
        if sc > best:
            best_end, best = e, sc
    for s in _boundaries(align.dest_start, forward=False):
        sc = _score_window(s, best_end)
        if sc > best:
            best_start, best = s, sc
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
