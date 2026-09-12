from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from video_autocut.black import (
    DarkFrameMetrics,
    FrameDecision,
    _coarse_candidates,
    _is_unusable_dark_fast,
    analyze_black,
    is_unusable_dark,
    measure_dark_frame,
)
from video_autocut.config import BlackConfig


def test_uniform_black_is_unusable():
    frame = np.zeros((180, 320), dtype=np.uint8)
    assert is_unusable_dark(measure_dark_frame(frame), BlackConfig())
    assert _is_unusable_dark_fast(frame, BlackConfig())


def test_dark_sensor_noise_is_unusable():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 11, size=(180, 320), dtype=np.uint8)
    assert is_unusable_dark(measure_dark_frame(frame), BlackConfig())
    assert _is_unusable_dark_fast(frame, BlackConfig())


def test_dark_frame_with_small_bright_subject_is_kept():
    frame = np.full((180, 320), 5, dtype=np.uint8)
    frame[:, 150:162] = 120
    assert not is_unusable_dark(measure_dark_frame(frame), BlackConfig())
    assert not _is_unusable_dark_fast(frame, BlackConfig())


def test_merely_dim_frame_is_kept():
    metrics = DarkFrameMetrics(mean_luma=24.0, p95_luma=40.0, p99_luma=60.0, edge_ratio=0.001)
    assert not is_unusable_dark(metrics, BlackConfig())


def test_coarse_candidates_bridge_one_sample_gap():
    decisions = [
        FrameDecision(0.0, False),
        FrameDecision(0.5, True),
        FrameDecision(1.0, False),
        FrameDecision(1.5, True),
        FrameDecision(2.0, False),
    ]
    candidates = _coarse_candidates(
        decisions,
        sample_interval_s=0.5,
        duration_s=2.5,
        bridge_gap_s=0.5,
    )
    assert [(item.start, item.end) for item in candidates] == [(0.5, 2.0)]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_adaptive_black_analysis_keeps_dim_scene(tmp_path: Path):
    video = tmp_path / "synthetic.mp4"
    # 2 s white, 2 s black, 2 s dim blue. The dim section is deliberately low
    # luminance but still contains visible structure/color and must be retained.
    filter_complex = (
        "color=c=white:s=320x180:d=2:r=30[v0];"
        "color=c=black:s=320x180:d=2:r=30[v1];"
        "color=c=0x081020:s=320x180:d=2:r=30," \
        "drawbox=x=130:y=40:w=60:h=100:color=0x4060a0:t=fill[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[out]"
    )
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    config = BlackConfig(hardware_decode="cpu")
    result = analyze_black(video, config)

    assert len(result.dark) == 1
    dark = result.dark[0]
    assert dark.start == pytest.approx(2.0, abs=0.15)
    assert dark.end == pytest.approx(4.0, abs=0.15)
    assert len(result.keep) == 2
    assert result.stats.coarse_samples > 0
    assert result.stats.refined_samples > 0
    assert result.stats.decoder == "cpu"


def test_refine_windows_merge_nearby_boundaries():
    from video_autocut.black import _build_refine_windows
    from video_autocut.segments import Segment

    candidates = [Segment(10.0, 12.0), Segment(14.0, 16.0)]
    windows = _build_refine_windows(
        candidates,
        duration_s=30.0,
        margin_s=1.0,
        batch_gap_s=2.0,
    )
    assert [(item.start, item.end) for item in windows] == [(9.0, 17.0)]
