from __future__ import annotations

import re
from pathlib import Path
import subprocess

from .media import require_binary
from .segments import Segment, merge_short_scenes

_SCENE_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_BLACK_RE = re.compile(
    r"black_start:(?P<start>[0-9]+(?:\.[0-9]+)?)\s+"
    r"black_end:(?P<end>[0-9]+(?:\.[0-9]+)?)\s+"
    r"black_duration:(?P<duration>[0-9]+(?:\.[0-9]+)?)"
)


def detect_scenes(
    video: Path,
    duration: float,
    threshold: float,
    min_scene_duration: float,
) -> list[Segment]:
    ffmpeg = require_binary("ffmpeg")
    # showinfo writes timestamps for frames selected by FFmpeg's scene score.
    filter_expr = f"select='gt(scene,{threshold})',showinfo"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "info",
        "-i", str(video),
        "-an",
        "-vf", filter_expr,
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg scene detection failed for {video}:\n{result.stderr[-2000:]}")

    cuts = [float(match.group(1)) for match in _SCENE_TIME_RE.finditer(result.stderr)]
    cuts = sorted({cut for cut in cuts if 0.0 < cut < duration})
    boundaries = [0.0, *cuts, duration]
    scenes = [
        Segment(boundaries[i], boundaries[i + 1], "scene")
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    return merge_short_scenes(scenes, min_scene_duration)


def detect_black_intervals(
    video: Path,
    min_duration: float,
    picture_black_ratio: float,
    pixel_black_threshold: float,
) -> list[Segment]:
    ffmpeg = require_binary("ffmpeg")
    filter_expr = (
        "blackdetect="
        f"d={min_duration}:"
        f"pic_th={picture_black_ratio}:"
        f"pix_th={pixel_black_threshold}"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "info",
        "-i", str(video),
        "-an",
        "-vf", filter_expr,
        "-f", "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg black detection failed for {video}:\n{result.stderr[-2000:]}")

    return [
        Segment(float(m.group("start")), float(m.group("end")), "black")
        for m in _BLACK_RE.finditer(result.stderr)
    ]


def extract_copy(video: Path, segment: Segment, destination: Path, overwrite: bool) -> None:
    ffmpeg = require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-ss", f"{segment.start:.6f}",
        "-i", str(video),
        "-t", f"{segment.duration:.6f}",
        "-map", "0:v?",
        "-map", "0:a?",
        "-map", "0:s?",
        "-map_metadata", "0",
        "-c", "copy",
        "-reset_timestamps", "1",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extraction failed for {destination}:\n{result.stderr[-2000:]}")
