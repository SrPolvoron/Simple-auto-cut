from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

from .ffmpeg_tools import extract_copy
from .media import clip_path, montage_path, output_directory, require_binary
from .segments import Segment

EXPORT_MODES = ("clips", "montage", "both")


@dataclass(frozen=True, slots=True)
class ExportArtifacts:
    mode: str
    clip_paths: list[Path]
    montage_path: Path | None
    dry_run: bool

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "dry_run": self.dry_run,
            "clips": [path.name for path in self.clip_paths],
            "montage": self.montage_path.name if self.montage_path is not None else None,
        }


def _ffconcat_path(path: Path) -> str:
    # FFmpeg's concat demuxer accepts forward-slash absolute paths on Windows too.
    # Escape characters that have meaning in ffconcat single-quoted strings.
    value = path.resolve().as_posix()
    value = value.replace("\\", "\\\\").replace("'", "'\\''")
    return f"file '{value}'"


def concat_copy(inputs: list[Path], destination: Path, overwrite: bool) -> None:
    """Concatenate already compatible media files without re-encoding."""
    if not inputs:
        raise ValueError("At least one input clip is required for a montage")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )

    if len(inputs) == 1:
        if destination.exists():
            destination.unlink()
        shutil.copy2(inputs[0], destination)
        return

    ffmpeg = require_binary("ffmpeg")
    list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffconcat",
            prefix="video-autocut-",
            delete=False,
        ) as fh:
            list_path = Path(fh.name)
            fh.write("ffconcat version 1.0\n")
            for item in inputs:
                fh.write(_ffconcat_path(item) + "\n")

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-map", "0:v?",
            "-map", "0:a?",
            "-map", "0:s?",
            "-c", "copy",
            "-reset_timestamps", "1",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg montage concatenation failed for {destination}:\n{result.stderr[-3000:]}"
            )
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)


def _montage_from_source(
    video: Path,
    segments: list[Segment],
    destination: Path,
    overwrite: bool,
) -> None:
    """Build a montage from source segments without leaving intermediate clips behind."""
    if not segments:
        return
    if destination.resolve() == video.resolve():
        raise RuntimeError("Montage destination would overwrite the source video")

    if len(segments) == 1:
        extract_copy(video, segments[0], destination, overwrite)
        return

    with tempfile.TemporaryDirectory(prefix="video-autocut-montage-") as temp_name:
        temp_dir = Path(temp_name)
        temporary_clips: list[Path] = []
        for index, segment in enumerate(segments, start=1):
            temp_clip = temp_dir / f"segment_{index:05d}{video.suffix.lower()}"
            extract_copy(video, segment, temp_clip, overwrite=True)
            temporary_clips.append(temp_clip)
        concat_copy(temporary_clips, destination, overwrite)


def export_segments(
    video: Path,
    segments: list[Segment],
    *,
    output_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> ExportArtifacts:
    """Export selected segments as individual clips, a montage, or both."""
    if mode not in EXPORT_MODES:
        raise ValueError(f"Unknown export mode: {mode}")

    out_dir = output_directory(output_root, video)
    out_dir.mkdir(parents=True, exist_ok=True)

    planned_clips = [clip_path(output_root, video, i) for i in range(1, len(segments) + 1)]
    montage = montage_path(output_root, video) if mode in {"montage", "both"} and segments else None

    if mode in {"clips", "both"}:
        for index, (segment, destination) in enumerate(zip(segments, planned_clips), start=1):
            print(
                f"  c{index:03d}: {segment.start:.3f}s -> {segment.end:.3f}s "
                f"({segment.duration:.3f}s) -> {destination.name}"
            )
            if not dry_run:
                extract_copy(video, segment, destination, overwrite)
    else:
        # Still print the selected timeline so montage-only jobs are auditable.
        for index, segment in enumerate(segments, start=1):
            print(
                f"  s{index:03d}: {segment.start:.3f}s -> {segment.end:.3f}s "
                f"({segment.duration:.3f}s)"
            )

    clip_outputs = planned_clips if mode in {"clips", "both"} else []

    if montage is not None:
        print(f"  montage: {montage.name}")
        if not dry_run:
            if mode == "both":
                concat_copy(planned_clips, montage, overwrite)
            else:
                _montage_from_source(video, segments, montage, overwrite)
    elif mode in {"montage", "both"}:
        print("  montage: skipped (no selected segments)")

    return ExportArtifacts(
        mode=mode,
        clip_paths=clip_outputs,
        montage_path=montage,
        dry_run=dry_run,
    )
