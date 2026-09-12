from __future__ import annotations

from pathlib import Path

from .ffmpeg_tools import detect_scenes
from .media import probe_media
from .segments import Segment


def split_scenes(
    video: Path,
    *,
    threshold: float = 0.30,
    min_scene_duration_s: float = 0.25,
) -> list[Segment]:
    """Return chronological scene intervals detected from hard/strong visual cuts."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("scene threshold must be between 0 and 1")
    if min_scene_duration_s < 0:
        raise ValueError("minimum scene duration must be >= 0")

    info = probe_media(video)
    scenes = detect_scenes(
        video,
        duration=info.duration,
        threshold=threshold,
        min_scene_duration=min_scene_duration_s,
    )
    return [Segment(item.start, item.end, "scene") for item in scenes]
