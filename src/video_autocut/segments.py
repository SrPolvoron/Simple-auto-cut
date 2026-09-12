from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    reason: str = ""
    score: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict:
        data = {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration": round(self.duration, 6),
        }
        if self.reason:
            data["reason"] = self.reason
        if self.score is not None:
            data["score"] = round(float(self.score), 6)
        return data


def normalize(segments: list[Segment], duration: float) -> list[Segment]:
    normalized: list[Segment] = []
    for segment in segments:
        start = max(0.0, min(duration, segment.start))
        end = max(start, min(duration, segment.end))
        if end > start:
            normalized.append(Segment(start, end, segment.reason, segment.score))
    return normalized


def complement(excluded: list[Segment], duration: float, min_duration: float = 0.0) -> list[Segment]:
    if duration <= 0:
        return []
    excluded = sorted(normalize(excluded, duration), key=lambda s: (s.start, s.end))

    merged: list[Segment] = []
    for segment in excluded:
        if not merged or segment.start > merged[-1].end:
            merged.append(segment)
        else:
            previous = merged[-1]
            merged[-1] = Segment(previous.start, max(previous.end, segment.end), "excluded")

    kept: list[Segment] = []
    cursor = 0.0
    for segment in merged:
        if segment.start - cursor >= min_duration:
            kept.append(Segment(cursor, segment.start, "non_black"))
        cursor = max(cursor, segment.end)
    if duration - cursor >= min_duration:
        kept.append(Segment(cursor, duration, "non_black"))
    return kept


def merge_short_scenes(scenes: list[Segment], min_duration: float) -> list[Segment]:
    """Merge very short scene fragments into the following scene where possible."""
    if min_duration <= 0 or len(scenes) <= 1:
        return scenes

    result: list[Segment] = []
    index = 0
    while index < len(scenes):
        current = scenes[index]
        if current.duration >= min_duration or index == len(scenes) - 1:
            result.append(current)
            index += 1
            continue

        following = scenes[index + 1]
        scenes[index + 1] = Segment(current.start, following.end, following.reason)
        index += 1
    return result
