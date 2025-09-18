"""Utility helpers for query expansion and fuzzy matching."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

QUERY_NAME_DELIMITER = "||"
DEFAULT_FUZZY_THRESHOLD = 72
MAX_VARIANTS = 5

WORD_SUBSTITUTIONS: Dict[str, Iterable[str]] = {
    "ph": ("f",),
    "f": ("ph",),
    "ck": ("k", "c"),
    "oo": ("u", "o"),
    "ou": ("u", "o"),
    "ie": ("ei",),
    "ei": ("ie",),
    "y": ("i",),
    "i": ("y",),
    "c": ("k", "q", "s"),
    "k": ("c", "q"),
    "q": ("k", "c"),
    "v": ("w", "b"),
    "w": ("v",),
    "b": ("v",),
    "m": ("n",),
    "n": ("m",),
    "ll": ("l",),
    "rr": ("r",),
    "ss": ("s",),
    "tt": ("t",),
    "pp": ("p",),
    "ch": ("sh",),
    "sh": ("ch",),
    "gh": ("g",),
    "g": ("gh", "j"),
    "j": ("g",),
}


def _normalize_text(text: Optional[str]) -> str:
    """Return a simplified representation of ``text`` suitable for matching."""

    if not text:
        return ""

    normalized = unidecode(text)
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("&", " ")
    normalized = normalized.replace("'", " ")
    normalized = normalized.lower()
    return " ".join(normalized.split())


def _generate_word_variants(word: str) -> List[str]:
    """Generate single-word typo variants for ``word``."""

    variants: set[str] = set()
    if not word:
        return []

    # Remove duplicated letters (Soufflenheim -> Souflenheim)
    for idx in range(len(word) - 1):
        if word[idx] == word[idx + 1]:
            variants.add(word[:idx] + word[idx + 1 :])

    # Swap neighbouring letters (Luminarc -> Luiminarc)
    for idx in range(len(word) - 1):
        if word[idx] != word[idx + 1]:
            variants.add(word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :])

    # Apply substitution rules
    for pattern in sorted(WORD_SUBSTITUTIONS, key=len, reverse=True):
        start = 0
        while True:
            found = word.find(pattern, start)
            if found == -1:
                break
            for replacement in WORD_SUBSTITUTIONS[pattern]:
                variants.add(word[:found] + replacement + word[found + len(pattern) :])
            start = found + 1

    return [variant for variant in variants if variant and variant != word]


def _expand_search_text_variants(search_text: str) -> List[str]:
    """Return typo variants for ``search_text`` (including the original value)."""

    if search_text is None:
        return []

    base = search_text.strip()
    if not base:
        return []

    normalized_base = _normalize_text(base)
    collected: Dict[str, str] = {}

    def add_variant(value: str) -> None:
        normalized_key = _normalize_text(value)
        if not normalized_key or normalized_key in collected:
            return
        collected[normalized_key] = value.strip()

    add_variant(base)
    add_variant(base.lower())
    add_variant(unidecode(base))

    normalized_tokens = normalized_base.split()
    if normalized_tokens:
        sequence_key = " ".join(normalized_tokens)
        seen_sequences = {sequence_key}
        queue: deque[Tuple[List[str], int]] = deque()
        queue.append((normalized_tokens, 0))
        limit_guard = MAX_VARIANTS * 3

        while queue:
            tokens, depth = queue.popleft()
            for idx, token in enumerate(tokens):
                for candidate in _generate_word_variants(token):
                    new_tokens = list(tokens)
                    new_tokens[idx] = candidate
                    key = " ".join(new_tokens)
                    if key in seen_sequences:
                        continue
                    seen_sequences.add(key)
                    add_variant(" ".join(new_tokens))
                    if len(collected) >= limit_guard:
                        break
                    if depth < 1:  # Allow chaining up to two edits
                        queue.append((new_tokens, depth + 1))
                if len(collected) >= limit_guard:
                    break
            if len(collected) >= limit_guard:
                break

    scored_variants = sorted(
        (
            (
                Levenshtein.distance(normalized_base, key),
                value,
            )
            for key, value in collected.items()
        ),
        key=lambda item: (item[0], item[1]),
    )

    return [value for _, value in scored_variants][:MAX_VARIANTS]


def encode_query_name(display_name: Optional[str], base_search_text: Optional[str]) -> Optional[str]:
    """Combine ``display_name`` and ``base_search_text`` for storage."""

    display = (display_name or "").strip().replace(QUERY_NAME_DELIMITER, " ")
    base = (base_search_text or "").strip().replace(QUERY_NAME_DELIMITER, " ")

    if base:
        combined = display or base
        return f"{combined}{QUERY_NAME_DELIMITER}{base}".strip(QUERY_NAME_DELIMITER)

    if display:
        return display

    return None


def decode_query_name(raw_value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract display and base search text from ``raw_value``."""

    if not raw_value:
        return None, None

    if QUERY_NAME_DELIMITER in raw_value:
        display, base = raw_value.split(QUERY_NAME_DELIMITER, 1)
        display = display.strip() or None
        base = base.strip() or None
        return display, base

    cleaned = raw_value.strip()
    return (cleaned or None, None)


def _build_fuzzy_targets(base_search_text: str) -> Dict[str, str]:
    normalized_base = _normalize_text(base_search_text)
    targets: Dict[str, str] = {}
    if normalized_base:
        targets[normalized_base] = base_search_text.strip()
        for token in normalized_base.split():
            if len(token) >= 4 and token not in targets:
                targets[token] = token
    return targets


def find_best_fuzzy_match(
    base_search_text: Optional[str],
    title: Optional[str],
    brand: Optional[str] = None,
    threshold: int = DEFAULT_FUZZY_THRESHOLD,
) -> Optional[Dict[str, object]]:
    """Return the best fuzzy match result against ``title`` or ``brand``."""

    if not base_search_text:
        return None

    targets = _build_fuzzy_targets(base_search_text)
    if not targets:
        return None

    choices = list(targets.keys())
    best: Optional[Dict[str, object]] = None

    for source, candidate in ("title", title), ("brand", brand):
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            continue

        match = process.extractOne(
            normalized_candidate,
            choices,
            scorer=fuzz.token_set_ratio,
            processor=None,
        )
        if match is None:
            continue

        choice, score, _ = match
        if score < threshold:
            continue

        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "source": source,
                "target": targets[choice],
                "source_text": candidate.strip() if candidate else candidate,
            }

    return best


def format_fuzzy_match(result: Optional[Dict[str, object]]) -> str:
    """Format a fuzzy match ``result`` for display."""

    if not result:
        return "No fuzzy match"

    label = "Title" if result.get("source") == "title" else "Brand"
    score = int(round(float(result.get("score", 0))))
    target = (result.get("target") or "").strip()
    if target:
        return f"{label} {score}% ↔ {target}"
    return f"{label} {score}%"

