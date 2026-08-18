from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "we", "our",
    "is", "are", "was", "be", "this", "that", "it", "with", "as", "at", "can",
    "very", "main",
}

CANON = {
    "quickly": "fast",
    "quick": "fast",
    "rapid": "fast",
    "rapidly": "fast",
    "speed": "fast",
    "faster": "fast",
    "fastest": "fast",
    "delivery": "deliver",
    "delivering": "deliver",
    "products": "product",
    "launched": "launch",
    "launching": "launch",
    "advantage": "benefit",
}


def _stem(token: str) -> str:
    token = CANON.get(token, token)
    for suffix in ("ing", "ers", "er", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [_stem(t) for t in text.split() if t not in STOP and len(t) > 1]
    return " ".join(tokens)


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class RepetitionAnalyzer:
    THRESHOLDS = {"strict": 0.34, "balanced": 0.48, "light": 0.64}

    def __init__(self, embedding_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.embedding_fn = embedding_fn

    def threshold_for(self, mode: str) -> float:
        return self.THRESHOLDS.get(mode, self.THRESHOLDS["balanced"])

    def _matrix(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, 0))
        cleaned = []
        has_text = []
        for index, text in enumerate(texts):
            normalized = normalize_text(text)
            has_text.append(bool(normalized))
            cleaned.append(normalized if normalized else f"blank{index}")
        if self.embedding_fn is not None:
            vectors = self.embedding_fn(texts)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            sim = (vectors / norms) @ (vectors / norms).T
        else:
            try:
                matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(cleaned)
                sim = cosine_similarity(matrix)
            except ValueError:
                sim = np.eye(n)
        for i in range(n):
            for j in range(i, n):
                if i != j and (not has_text[i] or not has_text[j]):
                    sim[i, j] = sim[j, i] = 0.0
                    continue
                lexical = _token_jaccard(cleaned[i], cleaned[j])
                sequence = SequenceMatcher(None, cleaned[i], cleaned[j]).ratio()
                value = max(float(sim[i, j]), lexical, sequence * 0.92)
                sim[i, j] = sim[j, i] = value
        return sim

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        return self._matrix(texts)

    def cluster(self, texts: list[str], threshold: float = 0.5) -> list[list[int]]:
        n = len(texts)
        if n == 0:
            return []
        sim = self._matrix(texts)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        return list(groups.values())

    def deduplicate(
        self,
        candidates: list[dict],
        threshold: float = 0.5,
        text_key: str = "text",
        score_key: str = "score",
    ) -> list[dict]:
        if not candidates:
            return []
        texts = [str(c.get(text_key, "")) for c in candidates]
        groups = self.cluster(texts, threshold=threshold)
        kept: list[dict] = []
        for group in groups:
            best = max(group, key=lambda idx: float(candidates[idx].get(score_key, 0)))
            winner = dict(candidates[best])
            if len(group) > 1:
                winner["repetition_group"] = [candidates[i].get("id", i) for i in group]
            kept.append(winner)
        return kept

    def pair_similarity(self, a: str, b: str) -> float:
        matrix = self._matrix([a, b])
        if matrix.shape != (2, 2):
            return 0.0
        return float(matrix[0, 1])

    def as_similarity_fn(self, candidates: list[dict]):
        texts = [str(c.get("text") or "") for c in candidates]
        matrix = self.similarity_matrix(texts)
        by_id = {str(c.get("id")): i for i, c in enumerate(candidates)}

        def similarity(a: dict, b: dict) -> float:
            ia, ib = by_id.get(str(a.get("id"))), by_id.get(str(b.get("id")))
            if ia is None or ib is None:
                text_sim = self.pair_similarity(str(a.get("text") or ""), str(b.get("text") or ""))
            else:
                text_sim = float(matrix[ia, ib])
            return max(text_sim, _structural_similarity(a, b))

        return similarity


def _scene_id(candidate: dict) -> str | None:
    topic = str(candidate.get("topic") or "")
    match = re.search(r"scene-(\d+)", topic)
    if not match:
        return None
    source = str(candidate.get("source_id") or "")
    return f"{source}:{match.group(1)}" if source else match.group(1)


def _structural_similarity(a: dict, b: dict) -> float:
    scene_a, scene_b = _scene_id(a), _scene_id(b)
    if scene_a is not None and scene_a == scene_b:
        return 0.82
    topic_a = str(a.get("topic") or "").strip().lower()
    topic_b = str(b.get("topic") or "").strip().lower()
    if topic_a and topic_a == topic_b and len(topic_a) > 3 and not topic_a.startswith("scene-"):
        return 0.78
    summary_a = str((a.get("metrics") or {}).get("visual_summary") or a.get("visual_summary") or "").strip().lower()
    summary_b = str((b.get("metrics") or {}).get("visual_summary") or b.get("visual_summary") or "").strip().lower()
    if summary_a and summary_b and summary_a == summary_b:
        return 0.8
    return 0.0
