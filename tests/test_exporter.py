from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.exporter import export_segments
from video_autocut.media import montage_path, probe_media
from video_autocut.segments import Segment


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_export_both_creates_clips_and_same_name_montage(tmp_path: Path):
    video = tmp_path / "source video.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=6",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
            "-c:a", "aac", "-shortest",
            str(video),
        ],
        check=True,
    )

    output_root = tmp_path / "exports"
    segments = [Segment(0.0, 2.0), Segment(4.0, 6.0)]
    artifacts = export_segments(
        video,
        segments,
        output_root=output_root,
        mode="both",
        overwrite=True,
        dry_run=False,
    )

    assert len(artifacts.clip_paths) == 2
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.clip_paths)
    assert artifacts.montage_path == montage_path(output_root, video)
    assert artifacts.montage_path is not None and artifacts.montage_path.exists()
    assert artifacts.montage_path.name == video.name
    assert 3.5 <= probe_media(artifacts.montage_path).duration <= 4.5


def test_montage_only_layout_does_not_plan_clip_outputs(tmp_path: Path):
    video = Path("video_001.mp4")
    artifacts = export_segments(
        video,
        [Segment(1.0, 2.0)],
        output_root=tmp_path,
        mode="montage",
        overwrite=False,
        dry_run=True,
    )
    assert artifacts.clip_paths == []
    assert artifacts.montage_path == tmp_path / "video_001" / "video_001.mp4"
