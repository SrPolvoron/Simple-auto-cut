from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import PersonConfig
from .faces import FaceMatcher
from .ffmpeg_tools import detect_scenes
from .media import probe_media
from .segments import Segment


@dataclass(frozen=True, slots=True)
class SceneDecision:
    segment: Segment
    matching_samples: int
    sampled_frames: int
    best_score: float


def _sample_times(segment: Segment, interval: float) -> list[float]:
    if interval <= 0:
        raise ValueError("sample_interval_s must be greater than 0")
    duration = segment.duration
    if duration <= 0:
        return []

    # Sample near the beginning and then at a fixed interval. Also include a point near
    # the end so short appearances at scene boundaries are less likely to be missed.
    epsilon = min(0.05, duration / 4)
    start = segment.start + epsilon
    end = max(start, segment.end - epsilon)
    times: list[float] = []
    current = start
    while current <= end + 1e-9:
        times.append(current)
        current += interval
    if not times or end - times[-1] > min(interval * 0.5, 0.20):
        times.append(end)
    return sorted({round(t, 6) for t in times})


def find_person_scenes(
    video: Path,
    matcher: FaceMatcher,
    config: PersonConfig,
) -> tuple[list[Segment], list[SceneDecision]]:
    info = probe_media(video)
    scenes = detect_scenes(
        video,
        duration=info.duration,
        threshold=config.scene_threshold,
        min_scene_duration=config.min_scene_duration_s,
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video with OpenCV: {video}")

    decisions: list[SceneDecision] = []
    selected: list[Segment] = []
    try:
        for scene in scenes:
            matches = 0
            sampled = 0
            best_score = -1.0
            for timestamp in _sample_times(scene, config.sample_interval_s):
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                sampled += 1
                result = matcher.match_frame(frame)
                best_score = max(best_score, result.best_score)
                if result.matched:
                    matches += 1
                    if matches >= config.min_matching_samples:
                        # Recall-oriented: once the scene is confirmed, no need to decode the rest.
                        break

            decision = SceneDecision(scene, matches, sampled, best_score)
            decisions.append(decision)
            if matches >= config.min_matching_samples:
                selected.append(
                    Segment(scene.start, scene.end, "reference_person", best_score)
                )
    finally:
        cap.release()

    return selected, decisions
