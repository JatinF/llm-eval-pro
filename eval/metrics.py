# eval/metrics.py
"""
Metric implementations for llm-eval-pro.

These are intentionally lightweight so the library is easy to run anywhere,
but structured so you can add richer metrics later (BLEU, BERTScore, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping
import difflib
import re


def _normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - strip leading/trailing whitespace
    - lowercase
    - collapse multiple spaces
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return _normalize(text).split()


def exact_match(prediction: str, expected: str) -> float:
    """
    Returns 1.0 if prediction == expected after normalization, else 0.0.
    """
    return 1.0 if _normalize(prediction) == _normalize(expected) else 0.0


def token_overlap(prediction: str, expected: str) -> float:
    """
    Symmetric token overlap:
    2 * |A ∩ B| / (|A| + |B|)
    where A and B are token sets.
    """
    p_tokens = set(_tokenize(prediction))
    e_tokens = set(_tokenize(expected))
    if not p_tokens and not e_tokens:
        return 1.0
    if not p_tokens or not e_tokens:
        return 0.0
    intersection = len(p_tokens & e_tokens)
    return 2.0 * intersection / (len(p_tokens) + len(e_tokens))


def similarity_ratio(prediction: str, expected: str) -> float:
    """
    Character-level similarity using difflib.SequenceMatcher.
    Returns a value in [0.0, 1.0].
    """
    a = _normalize(prediction)
    b = _normalize(expected)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---- Registry & aggregation -------------------------------------------------


MetricFn = Callable[[str, str], float]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    fn: MetricFn
    description: str


METRICS: Dict[str, MetricDefinition] = {
    "exact_match": MetricDefinition(
        name="exact_match",
        fn=exact_match,
        description="1.0 if normalized prediction == expected, else 0.0.",
    ),
    "token_overlap": MetricDefinition(
        name="token_overlap",
        fn=token_overlap,
        description="Symmetric token overlap based on whitespace tokens.",
    ),
    "similarity_ratio": MetricDefinition(
        name="similarity_ratio",
        fn=similarity_ratio,
        description="Character-level similarity using SequenceMatcher.",
    ),
}


def evaluate_one(
    prediction: str,
    expected: str,
    metric_names: Iterable[str],
) -> Dict[str, float]:
    """
    Compute a set of metrics for one prediction/expected pair.

    Returns:
        dict like {"exact_match": 1.0, "token_overlap": 0.8, ...}
    """
    results: Dict[str, float] = {}
    for name in metric_names:
        metric_def = METRICS.get(name)
        if metric_def is None:
            raise KeyError(f"Unknown metric: {name}")
        try:
            results[name] = float(metric_def.fn(prediction, expected))
        except Exception as exc:  # pragma: no cover (defensive)
            # In production you'd want logging here; for now we fail fast.
            raise RuntimeError(f"Metric '{name}' failed for input.") from exc
    return results


def aggregate(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """
    Aggregate metric values across examples using simple arithmetic mean.

    Args:
        rows: iterable of dicts (each from evaluate_one).

    Returns:
        dict of metric_name -> average value.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: (sums[key] / counts[key]) if counts[key] else 0.0
        for key in sums.keys()
    }
