from video_autocut.person import _sample_times
from video_autocut.segments import Segment


def test_short_scene_gets_samples_near_both_ends():
    samples = _sample_times(Segment(10.0, 10.4), 0.5)
    assert samples[0] > 10.0
    assert samples[-1] < 10.4
    assert len(samples) >= 2


def test_long_scene_uses_fixed_interval():
    samples = _sample_times(Segment(0.0, 2.0), 0.5)
    assert len(samples) >= 4
