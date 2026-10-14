"""Meaning-based text comparison through the local Ollama embedding model.

Runs wherever Ollama runs (CPU or GPU) and adds no Python dependency. Everything here is
optional: without an embedding model the callers keep their word-based behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

_log = logging.getLogger(__name__)

# Cosine similarity of nomic-embed-text, measured on transcript-style sentences: paraphrases
# 0.75-0.97, unrelated sentences 0.33-0.46, but parallel sentences that make different points
# ("the first reason is X" / "the second reason is Y") up to about 0.82. Only similarity above
# this base counts towards repetition, so balanced repeat removal merges near-paraphrases only.
REPEAT_BASE = 0.70
# Relevance to the user's focus: below this cosine a moment is treated as unrelated.
FOCUS_BASE = 0.45
FOCUS_SPAN = 0.30
# Most a strong focus match adds to a moment's score.
FOCUS_BOOST = 0.12


class Embedder:
    """Embeds texts with one Ollama model, remembering vectors for repeated texts. Disables itself
    after the first failure so a missing model costs one quick request, not one per call."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = (model or "").strip()
        self.available = bool(client is not None and self.model and callable(getattr(client, "embed", None)))
        self.error: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray | None:
        if not self.available:
            return None
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            try:
                rows = self.client.embed(self.model, missing)
            except Exception as exc:  # noqa: BLE001 - any failure means "no embeddings"
                self.available = False
                self.error = str(exc)
                _log.info("Embeddings unavailable (%s); using word-based similarity", exc)
                return None
            for text, row in zip(missing, rows):
                vector = np.asarray(row, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                self._cache[text] = vector / norm if norm > 0 else vector
        return np.stack([self._cache[text] for text in texts]) if texts else np.zeros((0, 0), dtype=np.float32)


def repetition_embedding_fn(embedder: Embedder | None) -> Callable[[list[str]], np.ndarray] | None:
    """An embedding function for RepetitionAnalyzer, or None when embeddings are unavailable."""
    if embedder is None or not embedder.available:
        return None

    def embed(texts: list[str]) -> np.ndarray:
        vectors = embedder(texts)
        if vectors is None:
            raise RuntimeError(embedder.error or "embeddings unavailable")
        return vectors

    return embed


def calibrated_repeat_similarity(cosine: np.ndarray) -> np.ndarray:
    """Map raw cosine to the 0-1 repetition scale used by the word-based measures."""
    return np.clip((cosine - REPEAT_BASE) / (1.0 - REPEAT_BASE), 0.0, 1.0)


def apply_focus(candidates: list[dict], focus: str, embedder: Embedder | None) -> int:
    """Raise the score of moments whose words match the user's focus. Returns how many moments
    were boosted (0 when there is no focus text, no spoken text, or no embedding model)."""
    focus = (focus or "").strip()
    if not focus or embedder is None or not embedder.available:
        return 0
    spoken = [c for c in candidates if str(c.get("text") or "").strip()]
    if not spoken:
        return 0
    vectors = embedder([focus] + [str(c["text"]) for c in spoken])
    if vectors is None:
        return 0
    focus_vector, rows = vectors[0], vectors[1:]
    boosted = 0
    for candidate, row in zip(spoken, rows):
        relevance = float(np.clip((float(row @ focus_vector) - FOCUS_BASE) / FOCUS_SPAN, 0.0, 1.0))
        candidate["focus_match"] = round(relevance, 3)
        if relevance > 0:
            candidate["score"] = round(min(1.0, float(candidate.get("score", 0)) + FOCUS_BOOST * relevance), 4)
            boosted += 1
    candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    return boosted
