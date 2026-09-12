from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import platform
import subprocess
import time

import cv2
import numpy as np

from .config import BlackConfig
from .media import MediaInfo, probe_media, require_binary
from .segments import Segment, complement


@dataclass(frozen=True, slots=True)
class DarkFrameMetrics:
    mean_luma: float
    p95_luma: float
    p99_luma: float
    edge_ratio: float


@dataclass(frozen=True, slots=True)
class FrameDecision:
    timestamp: float
    dark: bool


@dataclass(frozen=True, slots=True)
class ScanBackend:
    hwaccel: str | None
    gpu_scale: bool = False

    @property
    def decoder(self) -> str:
        return self.hwaccel or "cpu"


@dataclass(frozen=True, slots=True)
class BlackAnalysisStats:
    video_duration_s: float
    elapsed_s: float
    coarse_elapsed_s: float
    refine_elapsed_s: float
    realtime_factor: float
    decoder: str
    gpu_scale: bool
    hardware_fallback: bool
    coarse_fps: float
    refine_fps: float
    coarse_samples: int
    refined_samples: int
    candidate_intervals: int
    refine_windows: int
    detected_intervals: int

    def as_dict(self) -> dict:
        return {
            "video_duration_s": round(self.video_duration_s, 6),
            "elapsed_s": round(self.elapsed_s, 6),
            "coarse_elapsed_s": round(self.coarse_elapsed_s, 6),
            "refine_elapsed_s": round(self.refine_elapsed_s, 6),
            "realtime_factor": round(self.realtime_factor, 3),
            "decoder": self.decoder,
            "gpu_scale": self.gpu_scale,
            "hardware_fallback": self.hardware_fallback,
            "coarse_fps": round(self.coarse_fps, 6),
            "refine_fps": round(self.refine_fps, 6),
            "coarse_samples": self.coarse_samples,
            "refined_samples": self.refined_samples,
            "candidate_intervals": self.candidate_intervals,
            "refine_windows": self.refine_windows,
            "detected_intervals": self.detected_intervals,
        }


@dataclass(frozen=True, slots=True)
class BlackAnalysisResult:
    keep: list[Segment]
    dark: list[Segment]
    stats: BlackAnalysisStats


def _scaled_size(width: int | None, height: int | None, max_side: int) -> tuple[int, int]:
    if not width or not height or width <= 0 or height <= 0:
        return max_side, max_side
    scale = min(1.0, max_side / max(width, height))
    out_w = max(2, int(round(width * scale)))
    out_h = max(2, int(round(height * scale)))
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    return max(2, out_w), max(2, out_h)


def _histogram_percentile(gray: np.ndarray, percentile: float) -> float:
    hist = np.bincount(gray.ravel(), minlength=256)
    cumulative = np.cumsum(hist)
    target = int(np.ceil((percentile / 100.0) * gray.size))
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(min(255, max(0, index)))


def _edge_ratio(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 12, 36)
    return float(np.count_nonzero(edges)) / float(edges.size)


def measure_dark_frame(gray: np.ndarray) -> DarkFrameMetrics:
    if gray.ndim != 2:
        raise ValueError("measure_dark_frame expects a grayscale frame")

    return DarkFrameMetrics(
        mean_luma=float(np.mean(gray)),
        p95_luma=_histogram_percentile(gray, 95.0),
        p99_luma=_histogram_percentile(gray, 99.0),
        edge_ratio=_edge_ratio(gray),
    )


def is_unusable_dark(metrics: DarkFrameMetrics, config: BlackConfig) -> bool:
    extremely_black = (
        metrics.mean_luma <= config.extreme_mean_luma
        and metrics.p99_luma <= config.extreme_p99_luma
    )
    if extremely_black:
        return True

    return (
        metrics.mean_luma <= config.dark_mean_luma
        and metrics.p95_luma <= config.dark_p95_luma
        and metrics.p99_luma <= config.dark_p99_luma
        and metrics.edge_ratio <= config.max_edge_ratio
    )


def _is_unusable_dark_fast(gray: np.ndarray, config: BlackConfig) -> bool:
    """Classify a frame using cheap exits before structural edge analysis."""
    mean_luma = float(np.mean(gray))

    if mean_luma > config.dark_mean_luma:
        return False

    total = float(gray.size)

    if mean_luma <= config.extreme_mean_luma:
        above_extreme_p99 = float(np.count_nonzero(gray > config.extreme_p99_luma)) / total
        if above_extreme_p99 <= 0.01:
            return True

    above_dark_p95 = float(np.count_nonzero(gray > config.dark_p95_luma)) / total
    if above_dark_p95 > 0.05:
        return False

    above_dark_p99 = float(np.count_nonzero(gray > config.dark_p99_luma)) / total
    if above_dark_p99 > 0.01:
        return False

    return _edge_ratio(gray) <= config.max_edge_ratio


@lru_cache(maxsize=1)
def _available_hwaccels() -> set[str]:
    ffmpeg = require_binary("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-hwaccels"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {
        line.strip().lower()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    }


@lru_cache(maxsize=1)
def _available_filters() -> set[str]:
    ffmpeg = require_binary("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()

    filters: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) >= 3 and parts[0][0] in "T." and parts[0][1] in "S." and parts[0][2] in "C.":
            filters.add(parts[1].lower())
    return filters


def _select_hwaccel(mode: str) -> str | None:
    normalized = mode.strip().lower()
    if normalized == "cpu":
        return None
    if normalized not in {"auto", "cuda", "d3d11va"}:
        raise ValueError("black.hardware_decode must be one of: auto, cpu, cuda, d3d11va")

    available = _available_hwaccels()
    if normalized != "auto":
        if normalized not in available:
            raise RuntimeError(
                f"FFmpeg does not report '{normalized}' hardware acceleration support. "
                f"Available: {', '.join(sorted(available)) or 'none'}"
            )
        return normalized

    if "cuda" in available:
        return "cuda"
    if platform.system() == "Windows" and "d3d11va" in available:
        return "d3d11va"
    return None


def _cuda_scale_available() -> bool:
    return "scale_cuda" in _available_filters()


def _format_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_scan_progress(
    label: str,
    sample_index: int,
    expected_samples: int,
    fps: float,
    started: float,
) -> None:
    elapsed = max(1e-9, time.perf_counter() - started)
    processed_s = sample_index / fps
    total_s = expected_samples / fps
    ratio = min(1.0, sample_index / max(1, expected_samples))
    speed = processed_s / elapsed
    remaining_s = max(0.0, total_s - processed_s)
    eta = remaining_s / speed if speed > 0 else 0.0
    print(
        f"\r  {label}: {ratio * 100:5.1f}%  "
        f"{_format_clock(processed_s)}/{_format_clock(total_s)}  "
        f"{speed:5.1f}x  ETA {_format_clock(eta)}",
        end="",
        flush=True,
    )


def _build_scan_command(
    video: Path,
    info: MediaInfo,
    config: BlackConfig,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    backend: ScanBackend,
) -> tuple[list[str], int, int]:
    ffmpeg = require_binary("ffmpeg")
    out_w, out_h = _scaled_size(info.width, info.height, config.analysis_max_side)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if backend.hwaccel is not None:
        command.extend(["-hwaccel", backend.hwaccel])
        if backend.hwaccel == "cuda" and backend.gpu_scale:
            command.extend(["-hwaccel_output_format", "cuda"])
    if start_s > 0:
        command.extend(["-ss", f"{start_s:.6f}"])
    command.extend(["-i", str(video)])
    if duration_s < info.duration - 1e-6:
        command.extend(["-t", f"{duration_s:.6f}"])

    if backend.hwaccel == "cuda" and backend.gpu_scale:
        # Keep decoded frames on the GPU, shrink them there, and only then copy the
        # small NV12 frames back to system RAM. Grayscale conversion is cheap on CPU.
        filter_chain = (
            f"scale_cuda={out_w}:{out_h}:format=nv12,"
            f"hwdownload,format=nv12,fps={fps:.8f},format=gray"
        )
    else:
        filter_chain = f"fps={fps:.8f},scale={out_w}:{out_h}:flags=area,format=gray"

    command.extend(
        [
            "-an",
            "-sn",
            "-vf",
            filter_chain,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    return command, out_w, out_h


def _scan_frames(
    video: Path,
    info: MediaInfo,
    config: BlackConfig,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    backend: ScanBackend,
    progress: bool = False,
    progress_label: str = "scan",
) -> tuple[list[FrameDecision], float]:
    if duration_s <= 0 or fps <= 0:
        return [], 0.0

    command, out_w, out_h = _build_scan_command(
        video,
        info,
        config,
        start_s=start_s,
        duration_s=duration_s,
        fps=fps,
        backend=backend,
    )

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Unable to start FFmpeg dark-frame analysis")

    frame_size = out_w * out_h
    decisions: list[FrameDecision] = []
    sample_index = 0
    expected_samples = max(1, int(math.ceil(duration_s * fps)))
    started = time.perf_counter()
    last_progress = 0.0

    try:
        while True:
            payload = process.stdout.read(frame_size)
            if not payload:
                break
            if len(payload) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete analysis frame")

            gray = np.frombuffer(payload, dtype=np.uint8).reshape((out_h, out_w))
            timestamp = min(info.duration, start_s + (sample_index / fps))
            decisions.append(FrameDecision(timestamp, _is_unusable_dark_fast(gray, config)))
            sample_index += 1

            if progress:
                now = time.perf_counter()
                if now - last_progress >= 0.5 or sample_index >= expected_samples:
                    _print_scan_progress(progress_label, sample_index, expected_samples, fps, started)
                    last_progress = now
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    elapsed = max(1e-9, time.perf_counter() - started)
    if progress:
        _print_scan_progress(progress_label, sample_index, max(sample_index, expected_samples), fps, started)
        print()

    if return_code != 0:
        suffix = "+gpu-scale" if backend.gpu_scale else ""
        raise RuntimeError(
            f"FFmpeg dark-frame analysis failed using decoder={backend.decoder}{suffix} for {video}:\n"
            f"{stderr[-3000:]}"
        )
    return decisions, elapsed


def _backend_candidates(mode: str) -> list[ScanBackend]:
    selected = _select_hwaccel(mode)
    normalized = mode.strip().lower()

    if selected == "cuda":
        candidates: list[ScanBackend] = []
        if _cuda_scale_available():
            candidates.append(ScanBackend("cuda", True))
        candidates.append(ScanBackend("cuda", False))
        if normalized == "auto":
            candidates.append(ScanBackend(None, False))
        return candidates

    if selected == "d3d11va":
        candidates = [ScanBackend("d3d11va", False)]
        if normalized == "auto":
            candidates.append(ScanBackend(None, False))
        return candidates

    return [ScanBackend(None, False)]


def _scan_with_backend_fallback(
    video: Path,
    info: MediaInfo,
    config: BlackConfig,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    backends: list[ScanBackend],
    progress: bool,
    progress_label: str,
) -> tuple[list[FrameDecision], float, ScanBackend, bool]:
    last_error: RuntimeError | None = None
    for index, backend in enumerate(backends):
        try:
            decisions, elapsed = _scan_frames(
                video,
                info,
                config,
                start_s=start_s,
                duration_s=duration_s,
                fps=fps,
                backend=backend,
                progress=progress,
                progress_label=progress_label,
            )
            return decisions, elapsed, backend, index > 0
        except RuntimeError as exc:
            last_error = exc
            if index + 1 < len(backends):
                suffix = "+gpu-scale" if backend.gpu_scale else ""
                print(f"  decoder {backend.decoder}{suffix} failed; trying fallback...")
                continue
            raise
    assert last_error is not None
    raise last_error


def _merge_dark_windows(
    windows: list[Segment],
    *,
    bridge_gap_s: float,
    min_duration_s: float,
) -> list[Segment]:
    if not windows:
        return []

    windows = sorted(windows, key=lambda item: (item.start, item.end))
    merged: list[Segment] = [windows[0]]
    for item in windows[1:]:
        previous = merged[-1]
        if item.start - previous.end <= bridge_gap_s + 1e-9:
            merged[-1] = Segment(previous.start, max(previous.end, item.end), "unusable_dark")
        else:
            merged.append(item)

    return [item for item in merged if item.duration >= min_duration_s]


def _coarse_candidates(
    decisions: list[FrameDecision],
    *,
    sample_interval_s: float,
    duration_s: float,
    bridge_gap_s: float,
) -> list[Segment]:
    windows = [
        Segment(
            item.timestamp,
            min(duration_s, item.timestamp + sample_interval_s),
            "unusable_dark",
        )
        for item in decisions
        if item.dark and item.timestamp < duration_s
    ]
    return _merge_dark_windows(
        windows,
        bridge_gap_s=bridge_gap_s,
        min_duration_s=0.0,
    )


def _stable_run_start(
    decisions: list[FrameDecision],
    *,
    desired_dark: bool,
    stable_frames: int,
    require_opposite_before: bool,
) -> float | None:
    if not decisions:
        return None
    stable_frames = max(1, stable_frames)
    seen_opposite = not require_opposite_before

    for index in range(len(decisions)):
        if decisions[index].dark != desired_dark:
            seen_opposite = True
            continue
        if not seen_opposite:
            continue
        end = min(len(decisions), index + stable_frames)
        if end - index < stable_frames:
            break
        if all(item.dark == desired_dark for item in decisions[index:end]):
            return decisions[index].timestamp
    return None


def _merge_refine_windows(windows: list[Segment], gap_s: float) -> list[Segment]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda item: (item.start, item.end))
    merged = [windows[0]]
    for item in windows[1:]:
        previous = merged[-1]
        if item.start - previous.end <= gap_s + 1e-9:
            merged[-1] = Segment(previous.start, max(previous.end, item.end), "refine")
        else:
            merged.append(item)
    return merged


def _build_refine_windows(
    candidates: list[Segment],
    *,
    duration_s: float,
    margin_s: float,
    batch_gap_s: float,
) -> list[Segment]:
    windows: list[Segment] = []
    for candidate in candidates:
        if candidate.start > 0:
            windows.append(
                Segment(
                    max(0.0, candidate.start - margin_s),
                    min(duration_s, candidate.start + margin_s),
                    "refine",
                )
            )
        if candidate.end < duration_s:
            windows.append(
                Segment(
                    max(0.0, candidate.end - margin_s),
                    min(duration_s, candidate.end + margin_s),
                    "refine",
                )
            )
    return _merge_refine_windows(windows, batch_gap_s)


def _slice_decisions(
    decisions: list[FrameDecision],
    start_s: float,
    end_s: float,
) -> list[FrameDecision]:
    return [item for item in decisions if start_s - 1e-9 <= item.timestamp <= end_s + 1e-9]


def _refine_candidates_batched(
    video: Path,
    info: MediaInfo,
    config: BlackConfig,
    candidates: list[Segment],
    *,
    backend: ScanBackend,
    auto_fallback: bool,
    progress: bool,
) -> tuple[list[Segment], int, int, float, ScanBackend, bool]:
    if not config.refine_enabled or not candidates:
        return candidates, 0, 0, 0.0, backend, False

    windows = _build_refine_windows(
        candidates,
        duration_s=info.duration,
        margin_s=config.refine_margin_s,
        batch_gap_s=config.refine_batch_gap_s,
    )
    if not windows:
        return candidates, 0, 0, 0.0, backend, False

    backend_options = [backend]
    if auto_fallback and backend.hwaccel is not None:
        if backend.hwaccel == "cuda" and backend.gpu_scale:
            backend_options.append(ScanBackend("cuda", False))
        backend_options.append(ScanBackend(None, False))

    all_decisions: list[FrameDecision] = []
    total_samples = 0
    total_elapsed = 0.0
    fallback_used = False
    active_backend = backend

    for index, window in enumerate(windows, start=1):
        if progress:
            print(
                f"\r  refine: {index:3d}/{len(windows):3d} windows "
                f"({_format_clock(window.start)}-{_format_clock(window.end)})",
                end="",
                flush=True,
            )
        decisions, elapsed, used_backend, fell_back = _scan_with_backend_fallback(
            video,
            info,
            config,
            start_s=window.start,
            duration_s=window.duration,
            fps=config.refine_fps,
            backends=backend_options,
            progress=False,
            progress_label="refine",
        )
        if fell_back:
            fallback_used = True
            active_backend = used_backend
            backend_options = [used_backend]
        all_decisions.extend(decisions)
        total_samples += len(decisions)
        total_elapsed += elapsed

    if progress:
        print()

    all_decisions.sort(key=lambda item: item.timestamp)
    stable_frames = max(1, int(round(config.refine_stability_s * config.refine_fps)))
    refined: list[Segment] = []

    for candidate in candidates:
        refined_start = candidate.start
        if candidate.start > 0:
            start_window = _slice_decisions(
                all_decisions,
                max(0.0, candidate.start - config.refine_margin_s),
                min(info.duration, candidate.start + config.refine_margin_s),
            )
            found_start = _stable_run_start(
                start_window,
                desired_dark=True,
                stable_frames=stable_frames,
                require_opposite_before=False,
            )
            if found_start is not None:
                refined_start = found_start

        refined_end = candidate.end
        if candidate.end < info.duration:
            end_window = _slice_decisions(
                all_decisions,
                max(0.0, candidate.end - config.refine_margin_s),
                min(info.duration, candidate.end + config.refine_margin_s),
            )
            found_end = _stable_run_start(
                end_window,
                desired_dark=False,
                stable_frames=stable_frames,
                require_opposite_before=True,
            )
            if found_end is not None:
                refined_end = found_end

        if refined_end <= refined_start:
            refined.append(candidate)
        else:
            refined.append(Segment(refined_start, refined_end, "unusable_dark"))

    return refined, total_samples, len(windows), total_elapsed, active_backend, fallback_used


def analyze_black(
    video: Path,
    config: BlackConfig,
    *,
    progress: bool = True,
) -> BlackAnalysisResult:
    """Analyze unusable near-black footage using a low-rate scan and boundary refinement."""
    started = time.perf_counter()
    info = probe_media(video)
    coarse_fps = 1.0 / config.sample_interval_s
    backend_options = _backend_candidates(config.hardware_decode)

    coarse, coarse_elapsed, backend, coarse_fallback = _scan_with_backend_fallback(
        video,
        info,
        config,
        start_s=0.0,
        duration_s=info.duration,
        fps=coarse_fps,
        backends=backend_options,
        progress=progress,
        progress_label="coarse",
    )

    candidates = _coarse_candidates(
        coarse,
        sample_interval_s=config.sample_interval_s,
        duration_s=info.duration,
        bridge_gap_s=config.bridge_gap_s,
    )

    refined, refined_samples, refine_windows, refine_elapsed, final_backend, refine_fallback = (
        _refine_candidates_batched(
            video,
            info,
            config,
            candidates,
            backend=backend,
            auto_fallback=config.hardware_decode.strip().lower() == "auto",
            progress=progress,
        )
    )

    dark = _merge_dark_windows(
        refined,
        bridge_gap_s=config.bridge_gap_s,
        min_duration_s=config.min_black_duration_s,
    )
    keep = complement(dark, info.duration, config.min_keep_duration_s)

    elapsed = max(1e-9, time.perf_counter() - started)
    stats = BlackAnalysisStats(
        video_duration_s=info.duration,
        elapsed_s=elapsed,
        coarse_elapsed_s=coarse_elapsed,
        refine_elapsed_s=refine_elapsed,
        realtime_factor=info.duration / elapsed,
        decoder=final_backend.decoder,
        gpu_scale=final_backend.gpu_scale,
        hardware_fallback=coarse_fallback or refine_fallback,
        coarse_fps=coarse_fps,
        refine_fps=config.refine_fps if config.refine_enabled else 0.0,
        coarse_samples=len(coarse),
        refined_samples=refined_samples,
        candidate_intervals=len(candidates),
        refine_windows=refine_windows,
        detected_intervals=len(dark),
    )
    return BlackAnalysisResult(keep=keep, dark=dark, stats=stats)


def detect_unusable_dark_intervals(video: Path, config: BlackConfig) -> list[Segment]:
    return analyze_black(video, config).dark


def find_non_black_segments(video: Path, config: BlackConfig) -> tuple[list[Segment], list[Segment]]:
    result = analyze_black(video, config)
    return result.keep, result.dark
