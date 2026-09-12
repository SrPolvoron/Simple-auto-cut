from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required executable '{name}' was not found in PATH")
    return binary


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        denominator_f = float(denominator)
        return float(numerator) / denominator_f if denominator_f else None
    return float(rate)


def probe_media(path: Path) -> MediaInfo:
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)

    duration = None
    if payload.get("format", {}).get("duration") is not None:
        duration = float(payload["format"]["duration"])

    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if duration is None and video_stream and video_stream.get("duration") is not None:
        duration = float(video_stream["duration"])
    if duration is None:
        raise RuntimeError(f"Unable to determine duration for {path}")

    return MediaInfo(
        duration=duration,
        width=int(video_stream["width"]) if video_stream and video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream and video_stream.get("height") else None,
        fps=_parse_rate(video_stream.get("avg_frame_rate") if video_stream else None),
    )


def discover_videos(input_path: Path, extensions: list[str]) -> list[Path]:
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if input_path.is_file():
        if input_path.suffix.lower() not in normalized:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path.resolve()]
    if input_path.is_dir():
        return sorted(
            path.resolve()
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in normalized
        )
    raise FileNotFoundError(f"Input does not exist: {input_path}")


def output_directory(root: Path, video: Path) -> Path:
    return root / video.stem


def clip_path(root: Path, video: Path, index: int) -> Path:
    directory = output_directory(root, video)
    return directory / f"c{index:03d}_{video.stem}{video.suffix.lower()}"


def montage_path(root: Path, video: Path) -> Path:
    """Return the per-video montage path using the original filename."""
    return output_directory(root, video) / video.name
