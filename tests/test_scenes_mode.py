from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.scenes import split_scenes


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_split_scenes_detects_strong_compilation_cuts(tmp_path: Path):
    video = tmp_path / "compilation.mp4"
    # Three visually distinct 2-second sections with hard cuts.
    filter_complex = (
        "color=c=red:s=320x180:d=2:r=15[v0];"
        "testsrc2=s=320x180:d=2:r=15[v1];"
        "color=c=white:s=320x180:d=2:r=15[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[out]"
    )
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
    )

    scenes = split_scenes(video, threshold=0.20, min_scene_duration_s=0.2)
    assert len(scenes) >= 3
    assert scenes[0].start == pytest.approx(0.0, abs=0.05)
    assert scenes[-1].end == pytest.approx(6.0, abs=0.1)
