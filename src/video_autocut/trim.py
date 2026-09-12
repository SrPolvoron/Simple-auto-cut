from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .media import probe_media
from .segments import Segment, complement


@dataclass(frozen=True, slots=True)
class TrimPlan:
    duration: float
    removed: list[Segment]
    keep: list[Segment]


def build_trim_plan(
    duration: float,
    *,
    remove_start_s: float = 0.0,
    remove_end_s: float = 0.0,
    remove_middle_s: float = 0.0,
    min_keep_duration_s: float = 0.01,
) -> TrimPlan:
    """Build keep/remove intervals for simple batch trimming.

    ``remove_middle_s`` removes a window centered on the exact video midpoint.
    Options can be combined, e.g. remove 3 s from both the beginning and end.
    """
    if duration <= 0:
        raise ValueError("video duration must be greater than 0")

    values = {
        "remove_start_s": remove_start_s,
        "remove_end_s": remove_end_s,
        "remove_middle_s": remove_middle_s,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    if not any(value > 0 for value in values.values()):
        raise ValueError("at least one trim operation must be greater than 0 seconds")

    removed: list[Segment] = []
    if remove_start_s > 0:
        removed.append(Segment(0.0, min(duration, remove_start_s), "trim_start"))

    if remove_end_s > 0:
        removed.append(
            Segment(max(0.0, duration - remove_end_s), duration, "trim_end")
        )

    if remove_middle_s > 0:
        half = remove_middle_s / 2.0
        midpoint = duration / 2.0
        removed.append(
            Segment(
                max(0.0, midpoint - half),
                min(duration, midpoint + half),
                "trim_middle",
            )
        )

    # complement() also merges overlapping exclusions, which is useful for short
    # videos where start/end/middle operations overlap.
    keep = complement(removed, duration, min_duration=min_keep_duration_s)
    return TrimPlan(duration=duration, removed=removed, keep=keep)


def plan_video_trim(
    video: Path,
    *,
    remove_start_s: float = 0.0,
    remove_end_s: float = 0.0,
    remove_middle_s: float = 0.0,
    min_keep_duration_s: float = 0.01,
) -> TrimPlan:
    info = probe_media(video)
    return build_trim_plan(
        info.duration,
        remove_start_s=remove_start_s,
        remove_end_s=remove_end_s,
        remove_middle_s=remove_middle_s,
        min_keep_duration_s=min_keep_duration_s,
    )
