from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.exporter import export_segments
from video_autocut.media import probe_media
from video_autocut.trim import build_trim_plan, plan_video_trim


def _pairs(segments):
    return [(round(item.start, 6), round(item.end, 6)) for item in segments]


def test_remove_start_three_seconds():
    plan = build_trim_plan(10.0, remove_start_s=3.0)
    assert _pairs(plan.keep) == [(3.0, 10.0)]


def test_remove_end_three_seconds():
    plan = build_trim_plan(10.0, remove_end_s=3.0)
    assert _pairs(plan.keep) == [(0.0, 7.0)]


def test_remove_middle_three_seconds():
    plan = build_trim_plan(10.0, remove_middle_s=3.0)
    assert _pairs(plan.keep) == [(0.0, 3.5), (6.5, 10.0)]


def test_trim_options_can_be_combined():
    plan = build_trim_plan(20.0, remove_start_s=2.0, remove_end_s=3.0, remove_middle_s=4.0)
    assert _pairs(plan.keep) == [(2.0, 8.0), (12.0, 17.0)]


def test_trim_requires_an_operation():
    with pytest.raises(ValueError):
        build_trim_plan(10.0)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_middle_trim_can_be_montaged_to_original_filename(tmp_path: Path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
            str(video),
        ],
        check=True,
    )
    plan = plan_video_trim(video, remove_middle_s=2.0)
    artifacts = export_segments(
        video,
        plan.keep,
        output_root=tmp_path / "exports",
        mode="montage",
        overwrite=True,
        dry_run=False,
    )
    assert artifacts.montage_path is not None
    assert artifacts.montage_path.name == "source.mp4"
    assert artifacts.montage_path.exists()
    assert 7.5 <= probe_media(artifacts.montage_path).duration <= 8.5
