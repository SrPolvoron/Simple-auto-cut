from pathlib import Path

from video_autocut.cli import build_parser


def test_output_option_is_accepted_after_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "black",
        "video.mp4",
        "--output",
        "D:/exports",
        "--export",
        "both",
        "--overwrite",
    ])
    assert args.output == Path("D:/exports")
    assert args.export_mode == "both"
    assert args.overwrite is True


def test_person_reference_filename_can_be_relative():
    parser = build_parser()
    args = parser.parse_args([
        "person",
        "C:/Videos/Podcast",
        "--reference",
        "person.jpg",
        "--export",
        "montage",
    ])
    assert args.reference == Path("person.jpg")
    assert args.export_mode == "montage"


def test_scenes_mode_options():
    parser = build_parser()
    args = parser.parse_args([
        "scenes",
        "C:/Videos/compilation.mp4",
        "--threshold", "0.25",
        "--min-scene", "0.4",
        "--preview",
    ])
    assert args.threshold == 0.25
    assert args.min_scene == 0.4
    assert args.preview is True


def test_trim_defaults_to_montage_and_supports_combined_removals():
    parser = build_parser()
    args = parser.parse_args([
        "trim",
        "C:/Videos",
        "--remove-start", "3",
        "--remove-end", "3",
        "--remove-middle", "2",
        "--output", "D:/trimmed",
    ])
    assert args.export_mode == "montage"
    assert args.remove_start == 3.0
    assert args.remove_end == 3.0
    assert args.remove_middle == 2.0
    assert args.output == Path("D:/trimmed")


def test_captures_mode_options():
    parser = build_parser()
    args = parser.parse_args([
        "captures",
        "C:/Videos/Trip",
        "--count", "10",
        "--kind", "landscape",
        "--min-gap", "4",
        "--format", "png",
        "--preview",
    ])
    assert args.count == 10
    assert args.kind == "landscape"
    assert args.min_gap == 4.0
    assert args.image_format == "png"
    assert args.preview is True


def test_captures_specific_person_reference_can_be_relative():
    parser = build_parser()
    args = parser.parse_args([
        "screenshots",
        "C:/Videos/Podcast",
        "--count", "5",
        "--kind", "person",
        "--reference", "himar.jpg",
    ])
    assert args.reference == Path("himar.jpg")
    assert args.kind == "person"
