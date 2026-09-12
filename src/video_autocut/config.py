from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(slots=True)
class GeneralConfig:
    output_dir: str = "exports"
    video_extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"]
    )
    overwrite: bool = False
    write_manifest: bool = True


@dataclass(slots=True)
class PersonConfig:
    scene_threshold: float = 0.30
    min_scene_duration_s: float = 0.35
    sample_interval_s: float = 0.50
    face_detection_threshold: float = 0.80
    analysis_max_side: int = 1280
    face_similarity_threshold: float = 0.363
    min_matching_samples: int = 1


@dataclass(slots=True)
class BlackConfig:
    sample_interval_s: float = 0.50
    analysis_max_side: int = 320
    refine_enabled: bool = True
    refine_fps: float = 20.0
    refine_margin_s: float = 1.0
    refine_stability_s: float = 0.15
    refine_batch_gap_s: float = 2.0
    hardware_decode: str = "auto"
    min_black_duration_s: float = 0.75
    bridge_gap_s: float = 0.50
    min_keep_duration_s: float = 0.75
    extreme_mean_luma: float = 6.0
    extreme_p99_luma: float = 20.0
    dark_mean_luma: float = 16.0
    dark_p95_luma: float = 26.0
    dark_p99_luma: float = 45.0
    max_edge_ratio: float = 0.006


@dataclass(slots=True)
class SceneConfig:
    threshold: float = 0.30
    min_scene_duration_s: float = 0.25


@dataclass(slots=True)
class TrimConfig:
    min_keep_duration_s: float = 0.01


@dataclass(slots=True)
class CaptureConfig:
    # Coarse candidate scan. One frame per second is usually enough to rank moments.
    sample_interval_s: float = 1.0
    analysis_max_side: int = 720
    hardware_decode: str = "auto"

    # Final local search around every selected moment.
    refine_window_s: float = 0.50
    refine_fps: float = 8.0

    # Keep captures spread across the video instead of returning near-duplicates.
    min_gap_s: float = 0.0  # 0 = derive automatically from duration and requested count
    edge_guard_s: float = 0.25

    # JPEG export quality (FFmpeg qscale: 2 is high quality).
    jpeg_qscale: int = 2

    # Quality weights. Metrics are rank-normalized per video where appropriate.
    sharpness_weight: float = 0.42
    exposure_weight: float = 0.18
    clipping_weight: float = 0.12
    contrast_weight: float = 0.10
    entropy_weight: float = 0.08
    stability_weight: float = 0.10


@dataclass(slots=True)
class AppConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    person: PersonConfig = field(default_factory=PersonConfig)
    black: BlackConfig = field(default_factory=BlackConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    trim: TrimConfig = field(default_factory=TrimConfig)
    captures: CaptureConfig = field(default_factory=CaptureConfig)


def _update_dataclass(instance, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key: {key}")
        setattr(instance, key, value)


def load_config(path: Path | None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    allowed_sections = {"general", "person", "black", "scenes", "trim", "captures"}
    unknown = set(data) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")

    _update_dataclass(config.general, data.get("general", {}))
    _update_dataclass(config.person, data.get("person", {}))
    _update_dataclass(config.black, data.get("black", {}))
    _update_dataclass(config.scenes, data.get("scenes", {}))
    _update_dataclass(config.trim, data.get("trim", {}))
    _update_dataclass(config.captures, data.get("captures", {}))
    return config
