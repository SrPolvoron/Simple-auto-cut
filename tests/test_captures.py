from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.captures import analyze_captures, export_captures
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
