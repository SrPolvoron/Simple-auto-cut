from video_autocut.segments import Segment, complement, merge_short_scenes


def test_complement_returns_non_black_regions():
    black = [Segment(2.0, 4.0), Segment(7.0, 8.0)]
    keep = complement(black, duration=10.0, min_duration=0.1)
    assert [(item.start, item.end) for item in keep] == [
        (0.0, 2.0),
        (4.0, 7.0),
        (8.0, 10.0),
    ]


def test_complement_merges_overlapping_exclusions():
    black = [Segment(2.0, 5.0), Segment(4.0, 7.0)]
    keep = complement(black, duration=10.0)
    assert [(item.start, item.end) for item in keep] == [(0.0, 2.0), (7.0, 10.0)]


def test_merge_short_scene_into_following_scene():
    scenes = [Segment(0.0, 0.2), Segment(0.2, 3.0), Segment(3.0, 5.0)]
    merged = merge_short_scenes(scenes, min_duration=0.35)
    assert [(item.start, item.end) for item in merged] == [(0.0, 3.0), (3.0, 5.0)]
