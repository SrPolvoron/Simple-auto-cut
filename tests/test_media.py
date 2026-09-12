from pathlib import Path

from video_autocut.media import clip_path, discover_videos


def test_discover_videos_from_directory(tmp_path: Path):
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.MOV").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x")

    videos = discover_videos(tmp_path, [".mp4", ".mov"])
    assert [path.name for path in videos] == ["a.MOV", "b.mp4"]


def test_clip_path_matches_requested_layout(tmp_path: Path):
    video = Path("video_001.mp4")
    path = clip_path(tmp_path, video, 2)
    assert path == tmp_path / "video_001" / "c002_video_001.mp4"


def test_montage_path_uses_original_filename(tmp_path: Path):
    from video_autocut.media import montage_path

    video = Path("video_001.MP4")
    path = montage_path(tmp_path, video)
    assert path == tmp_path / "video_001" / "video_001.MP4"
