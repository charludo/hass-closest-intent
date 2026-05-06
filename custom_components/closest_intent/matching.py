"""
Hassil-pattern expansion + RapidFuzz scoring.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

_LOGGER = logging.getLogger(__name__)

_ALT_RE = re.compile(r"\(([^()]+)\)")
_OPT_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Candidate:
    """One expanded sentence pattern, ready for scoring."""

    intent: str
    """Intent name (e.g. ``WetterStunde``)."""

    pattern_idx: int
    """Index into the intent's original pattern list (for debugging)."""

    text: str
    """Flattened text used for scoring."""


def expand_pattern(pattern: str, cap: int) -> list[str]:
    """
    Expand a Hassil-style pattern into a list of surface forms.

    Handles ``[optional]`` and ``(a|b|c)`` alternations. ``cap`` bounds
    the combinatorial blow-up; ``cap=0`` keeps just the first alternative
    of every group.
    """
    if cap == 0:
        text = _ALT_RE.sub(lambda m: m.group(1).split("|")[0], pattern)
        text = _OPT_RE.sub(lambda m: m.group(1), text)
        return [_normalise(text)]

    variants: list[str] = [pattern]
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

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        text = _normalise(v)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= cap:
            break
    return out


def _normalise(s: str) -> str:
    """Lowercase, strip extra whitespace and trailing punctuation."""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.rstrip("?.!,;:")
    return s


def score(user_text: str, candidate_text: str) -> int:
    """Similarity 0..100 between user input and a fixed candidate phrase."""
    return int(fuzz.token_sort_ratio(_normalise(user_text), candidate_text))


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
