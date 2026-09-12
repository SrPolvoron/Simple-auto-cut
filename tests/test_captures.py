import json
from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.captures import (
    CaptureCandidate,
    CaptureMetrics,
    CaptureResult,
    analyze_captures,
    create_capture_directory,
    export_captures,
    write_capture_manifest,
)
from video_autocut.config import CaptureConfig
from video_autocut.preview import generate_capture_preview


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_quality_captures_end_to_end(tmp_path: Path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=15:d=6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
    )
    config = CaptureConfig(
        sample_interval_s=1.0,
        analysis_max_side=320,
        hardware_decode="cpu",
        refine_window_s=0.2,
        refine_fps=5.0,
    )
    result = analyze_captures(
        video,
        config,
        count=3,
        kind="quality",
        progress=False,
    )
    assert len(result.selected) == 3
    assert result.decoder == "cpu"
    assert [item.timestamp for item in result.selected] == sorted(item.timestamp for item in result.selected)

    out = tmp_path / "out"
    files = export_captures(video, result, out, overwrite=True)
    assert len(files) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in files)

    pages = generate_capture_preview(
        files,
        [item.timestamp for item in result.selected],
        [item.final_score for item in result.selected],
        out,
        width=160,
    )
    assert pages and pages[0].exists() and pages[0].stat().st_size > 0


def test_capture_config_defaults_are_sane():
    config = CaptureConfig()
    assert config.sample_interval_s > 0
    assert config.refine_fps > 0
    assert config.jpeg_qscale >= 1


def test_capture_directories_are_numbered_after_the_highest_existing_one(tmp_path: Path):
    (tmp_path / "captures001").mkdir()
    (tmp_path / "captures003").mkdir()
    (tmp_path / "captures-not-a-batch").mkdir()

    directory = create_capture_directory(tmp_path)

    assert directory == tmp_path / "captures004"
    assert directory.is_dir()


def test_capture_manifest_accumulates_batches_and_uses_relative_paths(tmp_path: Path):
    result = CaptureResult(
        selected=[
            CaptureCandidate(
                timestamp=1.0,
                metrics=CaptureMetrics(
                    10.0, 0.5, 0.1, 0.1, 0.2, 0.5,
                    0.1, 0.1, 0.1, 0.0, 0.0, False,
                ),
            )
        ],
        elapsed_s=0.5,
        decoder="cpu",
        gpu_scale=False,
        sampled_frames=1,
        sample_fps=1.0,
        min_gap_s=0.0,
    )
    first_directory = create_capture_directory(tmp_path)
    first_file = first_directory / "s001_video.jpg"
    first_file.touch()
    manifest = tmp_path / "manifest.json"

    write_capture_manifest(
        manifest,
        source=Path("video.mp4"),
        kind="quality",
        result=result,
        files=[first_file],
    )
    write_capture_manifest(
        manifest,
        source=Path("other-video.mp4"),
        kind="landscape",
        result=result,
        files=[],
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["mode"] == "captures"
    assert len(payload["batches"]) == 2
    assert payload["batches"][0]["captures"][0]["file"] == "captures001/s001_video.jpg"
    assert payload["batches"][1]["captures"][0]["file"] is None
