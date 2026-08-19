from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.analysis.repetition import _structural_similarity
from app.analysis.sequence_limits import auto_cap, fit_candidate, is_low_value, split_span
from app.models.schemas import ClipPlan, ClipPlanItem


SimilarityFn = Callable[[dict, dict], float]
REPEAT_THRESHOLD = 0.48


def _duration(item: dict) -> float:
    return max(0.0, float(item["end"]) - float(item["start"]))


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _source_key(item: dict) -> str:
    return str(item.get("source_id") or "")


def _range_overlaps(item: dict, used: list[tuple[str, float, float]]) -> bool:
    span = (float(item["start"]), float(item["end"]))
    sid = _source_key(item)
    return any(sid == used_sid and _overlaps(span, (start, end)) for used_sid, start, end in used)


def _default_similarity(a: dict, b: dict) -> float:
    topic_a = str(a.get("topic") or "").strip().lower()
    topic_b = str(b.get("topic") or "").strip().lower()
    if topic_a and topic_a == topic_b and len(topic_a) > 3:
        return 0.9
    text_a = (a.get("text") or "").lower()
    text_b = (b.get("text") or "").lower()
    if not text_a or not text_b:
        return _structural_similarity(a, b)
    words_a, words_b = set(text_a.split()), set(text_b.split())
    if not words_a or not words_b:
        return _structural_similarity(a, b)
    lexical = len(words_a & words_b) / len(words_a | words_b)
    return max(lexical, _structural_similarity(a, b))


@dataclass
class ClipOptimizer:
    quality_floor: float = 0.35
    preferred_slack: float = 0.08
    acceptable_slack: float = 0.2

    def select(
        self,
        candidates: list[dict],
        min_duration: float,
        max_duration: float,
        must_include: list[tuple[float, float]] | None = None,
        must_exclude: list[tuple[float, float]] | None = None,
        similarity_fn: SimilarityFn | None = None,
        order_mode: str = "chronological",
        source_duration: float | None = None,
        target: float | None = None,
        max_sequence_duration: float = 5.0,
        min_sequence_duration: float | None = None,
        sequence_duration_mode: str = "custom",
        allow_repeated_content: bool = False,
        max_repeated_duration: float = 15.0,
        repeat_threshold: float = REPEAT_THRESHOLD,
        content_priorities: list[str] | None = None,
    ) -> ClipPlan:
        target = target or (min_duration + max_duration) / 2
        notes = ""
        similarity_fn = similarity_fn or _default_similarity
        must_include = must_include or []
        must_exclude = must_exclude or []
        max_sequence_duration = max(1.0, float(max_sequence_duration))
        if min_sequence_duration is not None:
            min_sequence_duration = max(0.5, float(min_sequence_duration))
        if sequence_duration_mode == "custom" and min_sequence_duration is not None:
            max_sequence_duration = max(max_sequence_duration, min_sequence_duration)
        max_repeated_duration = max(0.0, float(max_repeated_duration))
        sequence_duration_mode = "auto" if sequence_duration_mode == "auto" else "custom"
        if sequence_duration_mode == "auto":
            max_sequence_duration = auto_cap(min_duration, max_duration)

        def fit(item: dict, cap: float | None = None) -> dict:
            return fit_candidate(
                item,
                cap if cap is not None else max_sequence_duration,
                sequence_duration_mode,
                priorities=content_priorities,
                min_len=min_sequence_duration,
            )

        repeat_budget = 0.0 if not allow_repeated_content else max_repeated_duration

        if source_duration is not None and 0 < source_duration <= min_duration:
            return self._short_source_plan(
                candidates,
                source_duration,
                target,
                order_mode,
                max_sequence_duration,
                sequence_duration_mode,
                min_sequence_duration,
            )

        pool = []
        for candidate in candidates:
            span = (float(candidate["start"]), float(candidate["end"]))
            if any(_overlaps(span, excluded) for excluded in must_exclude):
                continue
            pool.append(fit(candidate))

        selected: list[dict] = []
        used_ranges: list[tuple[str, float, float]] = []
        repeated_so_far = 0.0

        def is_repeat(item: dict) -> bool:
            if not selected:
                return False
            return max(similarity_fn(item, s) for s in selected) >= repeat_threshold

        def add(item: dict, *, required: bool = False) -> bool:
            nonlocal repeated_so_far
            candidate = dict(item)
            cap = max_sequence_duration
            repeated = is_repeat(candidate)
            if repeated and not required:
                if not allow_repeated_content or repeat_budget <= 0:
                    return False
                remaining = repeat_budget - repeated_so_far
                if remaining < 0.8:
                    return False
                cap = min(cap, remaining)
            candidate = fit(candidate, cap)
            if _duration(candidate) < 0.8:
                return False
            span = (float(candidate["start"]), float(candidate["end"]))
            if _range_overlaps(candidate, used_ranges):
                return False
            if repeated:
                candidate["is_repeat"] = True
                repeated_so_far += _duration(candidate)
            selected.append(candidate)
            used_ranges.append((_source_key(candidate), span[0], span[1]))
            return True

        for required in must_include:
            match = self._best_covering(pool, required)
            if match is not None:
                add(match, required=True)

        remaining = [c for c in pool if c not in selected]
        remaining.sort(key=lambda c: float(c.get("score", 0)), reverse=True)

        def current_duration() -> float:
            return sum(_duration(c) for c in selected)

        while remaining:
            if current_duration() >= min_duration and current_duration() >= target * 0.92:
                break
            best_idx = None
            best_value = -1e9
            need_more = current_duration() < min_duration
            for idx, cand in enumerate(remaining):
                prepared = fit(cand)
                if _range_overlaps(prepared, used_ranges):
                    continue
                repeated = is_repeat(prepared)
                if repeated and not allow_repeated_content:
                    continue
                if repeated and not need_more:
                    continue
                if repeated and (repeat_budget - repeated_so_far) < 0.8:
                    continue
                dur = _duration(prepared)
                if repeated:
                    dur = min(dur, repeat_budget - repeated_so_far)
                if current_duration() + dur > max_duration + 1e-6:
                    continue
                if is_low_value(cand) and current_duration() >= min_duration * 0.55:
                    continue
                if float(cand.get("score", 0)) < self.quality_floor and current_duration() >= min_duration * 0.7:
                    continue
                novelty = 1.0
                if selected:
                    novelty = 1.0 - max(similarity_fn(cand, s) for s in selected)
                value = float(cand.get("score", 0)) + 0.55 * novelty
                if novelty < 0.35 and not need_more:
                    continue
                if value > best_value:
                    best_value = value
                    best_idx = idx
            if best_idx is None:
                break
            chosen = remaining.pop(best_idx)
            if float(chosen.get("score", 0)) < self.quality_floor and current_duration() >= min_duration * 0.8:
                continue
            add(chosen)

        if current_duration() > max_duration:
            selected = self._trim_to_max(selected, max_duration, must_include)

        if not selected:
            if pool:
                selected = [max(pool, key=lambda c: float(c.get("score", 0)))]
            elif source_duration:
                span = min(max_duration, source_duration, max_sequence_duration)
                selected = [
                    {
                        "id": "source-span",
                        "start": 0.0,
                        "end": span,
                        "score": 0.4,
                        "reason": "Coverage window",
                        "text": "",
                        "topic": "coverage",
                    }
                ]
                notes = "No high-value candidates were available; using a coverage window so a highlight can still be reviewed."

        selected = [fit(c) for c in selected]

        source_ids = {str(c.get("source_id") or "") for c in selected if c.get("source_id")}
        if len(source_ids) > 1:
            selected = self._multi_source_order(selected)
        elif order_mode == "narrative":
            selected = self._narrative_order(selected)
        else:
            selected.sort(key=lambda c: float(c["start"]))

        clips = [self._as_item(c) for c in selected]
        estimated = sum(c.duration for c in clips)
        if estimated < min_duration * 0.85 and remaining:
            notes = "Could not reach the target duration without adding low-value content. Preferring a shorter high-quality result."
        return ClipPlan(
            target_duration=target,
            estimated_duration=round(estimated, 3),
            clips=clips,
            notes=notes,
            order_mode=order_mode,
        )

    def _short_source_plan(
        self,
        candidates: list[dict],
        source_duration: float,
        target: float,
        order_mode: str,
        max_sequence_duration: float,
        sequence_duration_mode: str = "custom",
        min_sequence_duration: float | None = None,
    ) -> ClipPlan:
        if candidates:
            selected = []
            used: list[tuple[float, float]] = []
            for candidate in sorted(candidates, key=lambda c: c["start"]):
                item = fit_candidate(
                    candidate,
                    max_sequence_duration,
                    sequence_duration_mode,
                    min_len=min_sequence_duration,
                )
                span = (float(item["start"]), float(item["end"]))
                if any(_overlaps(span, used_span) for used_span in used):
                    continue
                selected.append(item)
                used.append(span)
        else:
            selected = []
        covered = sum(_duration(c) for c in selected)
        if covered + 0.5 < source_duration:
            selected = []
            for i, (start, end) in enumerate(split_span(0.0, source_duration, max_sequence_duration)):
                selected.append(
                    {
                        "id": f"source-{i}",
                        "start": start,
                        "end": end,
                        "score": 0.5,
                        "reason": "Source coverage",
                        "text": "",
                        "topic": "coverage",
                        "duration_reason": f"{end - start:.1f}s — source coverage window",
                    }
                )
        clips = [self._as_item(c) for c in selected]
        duration = sum(c.duration for c in clips) or min(source_duration, max_sequence_duration)
        return ClipPlan(
            target_duration=target,
            estimated_duration=min(duration, source_duration),
            clips=clips,
            notes="Source is already shorter than the requested highlight. Returning a minimally edited version.",
            order_mode=order_mode,
        )

    def _best_covering(self, pool: list[dict], required: tuple[float, float]) -> dict | None:
        covering = [c for c in pool if c["start"] <= required[0] + 0.25 and c["end"] >= required[1] - 0.25]
        if covering:
            return max(covering, key=lambda c: float(c.get("score", 0)))
        overlapping = [c for c in pool if _overlaps((c["start"], c["end"]), required)]
        if overlapping:
            return max(overlapping, key=lambda c: float(c.get("score", 0)))
        return {
            "id": f"must-{required[0]}-{required[1]}",
            "start": required[0],
            "end": required[1],
            "score": 1.0,
            "reason": "Manually required moment",
            "text": "",
            "topic": "must-include",
        }

    def _trim_to_max(
        self,
        selected: list[dict],
        max_duration: float,
        must_include: list[tuple[float, float]],
    ) -> list[dict]:
        protected = set()
        for item in selected:
            if any(_overlaps((item["start"], item["end"]), req) for req in must_include):
                protected.add(id(item))
        ordered = sorted(selected, key=lambda c: float(c.get("score", 0)))
        total = sum(_duration(c) for c in selected)
        kept = list(selected)
        for item in ordered:
            if total <= max_duration:
                break
            if id(item) in protected:
                continue
            kept.remove(item)
            total -= _duration(item)
        return kept

    def _narrative_order(self, selected: list[dict]) -> list[dict]:
        def rank(item: dict) -> tuple[int, float]:
            reason = (item.get("reason") or "").lower()
            if "hook" in reason:
                return (0, -float(item.get("score", 0)))
            if "conclusion" in reason or "ending" in reason:
                return (3, -float(item.get("score", 0)))
            if "demonstrat" in reason or "example" in reason:
                return (2, -float(item.get("score", 0)))
            return (1, item["start"])

        return sorted(selected, key=rank)

    def _multi_source_order(self, selected: list[dict]) -> list[dict]:
        remaining = list(selected)
        ordered: list[dict] = []

        def take(item: dict) -> None:
            remaining.remove(item)
            ordered.append(item)

        if remaining:
            opening = max(
                remaining,
                key=lambda c: (
                    1 if "hook" in (c.get("reason") or "").lower() else 0,
                    float(c.get("score") or 0),
                ),
            )
            take(opening)
        while remaining:
            last = ordered[-1]
            last_id = str(last.get("source_id") or "")
            follow = [
                c
                for c in remaining
                if str(c.get("source_id") or "") == last_id
                and float(c["start"]) >= float(last["end"]) - 0.4
                and float(c["start"]) - float(last["end"]) < 10
            ]
            if follow:
                take(min(follow, key=lambda c: float(c["start"])))
                continue
            forward = [
                c
                for c in remaining
                if str(c.get("source_id") or "") != last_id
                or float(c["start"]) >= float(last["end"]) - 0.4
            ]
            pool = forward or remaining
            take(max(pool, key=lambda c: float(c.get("score") or 0)))
        return ordered

    def _as_item(self, candidate: dict) -> ClipPlanItem:
        return ClipPlanItem(
            start=float(candidate["start"]),
            end=float(candidate["end"]),
            reason=str(candidate.get("reason") or "Meaningful moment"),
            score=float(candidate.get("score") or 0),
            candidate_id=str(candidate.get("id")) if candidate.get("id") is not None else None,
            transcript=str(candidate.get("text") or ""),
            duration_reason=str(candidate.get("duration_reason") or ""),
            source_id=str(candidate.get("source_id") or ""),
            source_name=str(candidate.get("source_name") or ""),
            source_path=str(candidate.get("source_path") or ""),
        )
