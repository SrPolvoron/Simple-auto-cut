from pathlib import Path
import shutil
import subprocess

import pytest

from video_autocut.preview import format_timestamp, generate_contact_sheet, generate_cut_preview, write_html_report
from video_autocut.segments import Segment


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(65.25) == "01:05.250"
    assert format_timestamp(3661.5) == "01:01:01.500"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_preview_outputs(tmp_path: Path):
    video = tmp_path / "preview.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=10:d=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    out = tmp_path / "out"
    cuts = generate_cut_preview(video, [Segment(1.0, 2.0)], out, width=160)
    sheets = generate_contact_sheet(video, out, every_s=1.0, width=160, columns=2, rows_per_page=2)
    report = out / "report.html"
    write_html_report(
        report,
        source=video,
        keep_segments=[Segment(0.0, 1.0), Segment(2.0, 4.0)],
        dark_segments=[Segment(1.0, 2.0)],
        analysis={"decoder": "cpu", "realtime_factor": 10.0},
        cut_preview_pages=cuts,
        contact_sheet_pages=sheets,
    )

    assert cuts and all(path.exists() and path.stat().st_size > 0 for path in cuts)
    assert sheets and all(path.exists() and path.stat().st_size > 0 for path in sheets)
    assert report.exists()
    assert "Removed dark intervals" in report.read_text(encoding="utf-8")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_selected_segment_preview_and_report(tmp_path: Path):
    from video_autocut.preview import generate_segment_preview, write_selection_report

    video = tmp_path / "selected.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
    )
    out = tmp_path / "out_selected"
    segments = [Segment(0.0, 1.0), Segment(2.0, 3.0)]
    pages = generate_segment_preview(video, segments, out, width=160, columns=2, rows_per_page=1)
    report = out / "report.html"
    write_selection_report(
        report,
        source=video,
        mode="person",
        selected_segments=segments,
        export_info={"mode": "both", "montage": "selected.mp4"},
        preview_pages=pages,
    )
    assert pages and all(path.exists() and path.stat().st_size > 0 for path in pages)
    assert report.exists()
    assert "Selected intervals" in report.read_text(encoding="utf-8")

@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_cut_preview_can_include_interval_ending_at_video_end(tmp_path: Path):
    video = tmp_path / "end_trim.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
    )
    pages = generate_cut_preview(video, [Segment(2.0, 3.0)], tmp_path / "preview_end", width=160)
    assert pages and pages[0].exists() and pages[0].stat().st_size > 0
