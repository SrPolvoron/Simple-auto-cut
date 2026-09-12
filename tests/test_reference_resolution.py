from pathlib import Path

from video_autocut.faces import resolve_reference_path


def test_relative_reference_can_live_next_to_input_folder_videos(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "podcast"
    input_dir.mkdir()
    reference = input_dir / "person.jpg"
    reference.write_bytes(b"not-an-image-needed-for-path-test")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_reference_path(Path("person.jpg"), input_dir)
    assert resolved == reference.resolve()


def test_relative_reference_can_live_next_to_single_video(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "podcast"
    input_dir.mkdir()
    video = input_dir / "episode.mp4"
    video.write_bytes(b"")
    reference = input_dir / "person.png"
    reference.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_reference_path(Path("person.png"), video)
    assert resolved == reference.resolve()
