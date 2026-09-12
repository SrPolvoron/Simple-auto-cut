from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np

from .black import ScanBackend, _backend_candidates, _scaled_size
from .config import CaptureConfig
from .faces import FaceMatcher
from .media import MediaInfo, probe_media, require_binary


CAPTURE_KINDS = ("quality", "person", "landscape", "situation")
IMAGE_FORMATS = ("jpg", "png")
_CAPTURE_DIRECTORY_PATTERN = re.compile(r"captures(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    sharpness: float
    mean_luma: float
    exposure: float
    clipping: float
    contrast: float
    entropy: float
    motion: float
    edge_ratio: float
    saturation: float
    person_score: float
    reference_score: float
    reference_matched: bool


@dataclass(slots=True)
class CaptureCandidate:
    timestamp: float
    metrics: CaptureMetrics
    quality_score: float = 0.0
    subject_score: float = 0.0
    final_score: float = 0.0

    def as_dict(self) -> dict:
        payload = asdict(self.metrics)
        payload.update(
            {
                "timestamp": round(self.timestamp, 6),
                "quality_score": round(self.quality_score, 6),
                "subject_score": round(self.subject_score, 6),
                "final_score": round(self.final_score, 6),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class CaptureResult:
    selected: list[CaptureCandidate]
    elapsed_s: float
    decoder: str
    gpu_scale: bool
    sampled_frames: int
    sample_fps: float
    min_gap_s: float

    def as_dict(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed_s, 6),
            "decoder": self.decoder,
            "gpu_scale": self.gpu_scale,
            "sampled_frames": self.sampled_frames,
            "sample_fps": round(self.sample_fps, 6),
            "min_gap_s": round(self.min_gap_s, 6),
            "selected": [item.as_dict() for item in self.selected],
        }


class GenericPersonDetector:
    """Lightweight generic face-presence detector using OpenCV bundled cascades.

    It is intentionally cheap enough for coarse ranking. Specific-person selection uses
    SFace through FaceMatcher instead.
    """

    def __init__(self) -> None:
        base = Path(cv2.data.haarcascades)
        self.frontal = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self.profile = cv2.CascadeClassifier(str(base / "haarcascade_profileface.xml"))

    @staticmethod
    def _resize(gray: np.ndarray, max_side: int = 640) -> np.ndarray:
        height, width = gray.shape[:2]
        longest = max(height, width)
        if longest <= max_side:
            return gray
        scale = max_side / float(longest)
        return cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def score(self, gray: np.ndarray) -> float:
        small = self._resize(gray)
        height, width = small.shape[:2]
        area = float(height * width)
        if area <= 0:
            return 0.0
        faces: list[tuple[int, int, int, int]] = []
        for classifier, image in (
            (self.frontal, small),
            (self.profile, small),
            (self.profile, cv2.flip(small, 1)),
        ):
            if classifier.empty():
                continue
            detected = classifier.detectMultiScale(
                image,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(28, 28),
            )
            faces.extend(tuple(int(v) for v in row) for row in detected)
        if not faces:
            return 0.0
        max_area = max(float(w * h) for _, _, w, h in faces) / area
        count_bonus = min(0.25, 0.05 * max(0, len(faces) - 1))
        # A reasonably sized visible face should score strongly, while tiny false positives
        # remain weak. Clamp to [0, 1].
        return float(min(1.0, max_area * 18.0 + count_bonus))


def _entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probabilities = hist[hist > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _edge_ratio(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _measure_frame(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    *,
    person_detector: GenericPersonDetector | None,
    matcher: FaceMatcher | None,
) -> tuple[CaptureMetrics, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.count_nonzero((gray <= 4) | (gray >= 251))) / float(gray.size)
    exposure = float(math.exp(-((mean_luma - 125.0) / 72.0) ** 2))
    clipping = max(0.0, min(1.0, 1.0 - clipped / 0.16))
    entropy = _entropy(gray)
    edges = _edge_ratio(gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    motion = 0.0
    if previous_gray is not None and previous_gray.shape == gray.shape:
        motion = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0

    person_score = person_detector.score(gray) if person_detector is not None else 0.0
    reference_score = -1.0
    reference_matched = False
    if matcher is not None:
        match = matcher.match_frame(frame)
        reference_score = match.best_score
        reference_matched = match.matched

    metrics = CaptureMetrics(
        sharpness=sharpness,
        mean_luma=mean_luma,
        exposure=exposure,
        clipping=clipping,
        contrast=contrast,
        entropy=entropy,
        motion=motion,
        edge_ratio=edges,
        saturation=saturation,
        person_score=person_score,
        reference_score=reference_score,
        reference_matched=reference_matched,
    )
    return metrics, gray


def _rank(values: list[float], *, reverse: bool = False) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    if np.allclose(array, array[0]):
        return [0.5] * len(values)
    order = np.argsort(array)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    ranks /= max(1, len(values) - 1)
    if reverse:
        ranks = 1.0 - ranks
    return [float(value) for value in ranks]


def _score_candidates(candidates: list[CaptureCandidate], kind: str, config: CaptureConfig) -> None:
    if not candidates:
        return

    sharp = _rank([math.log1p(item.metrics.sharpness) for item in candidates])
    contrast = _rank([item.metrics.contrast for item in candidates])
    entropy = _rank([item.metrics.entropy for item in candidates])
    stability = _rank([item.metrics.motion for item in candidates], reverse=True)
    motion = _rank([item.metrics.motion for item in candidates])
    edges = _rank([item.metrics.edge_ratio for item in candidates])
    saturation = _rank([item.metrics.saturation for item in candidates])
    people = _rank([item.metrics.person_score for item in candidates])
    refs = _rank([item.metrics.reference_score for item in candidates])

    for index, item in enumerate(candidates):
        quality = (
            config.sharpness_weight * sharp[index]
            + config.exposure_weight * item.metrics.exposure
            + config.clipping_weight * item.metrics.clipping
            + config.contrast_weight * contrast[index]
            + config.entropy_weight * entropy[index]
            + config.stability_weight * stability[index]
        )
        item.quality_score = float(max(0.0, min(1.0, quality)))

        if kind == "quality":
            subject = 0.5
            final = quality
        elif kind == "person":
            if item.metrics.reference_score >= 0:
                subject = refs[index] if item.metrics.reference_matched else 0.0
            else:
                subject = people[index] if item.metrics.person_score > 0 else 0.0
            final = 0.58 * quality + 0.42 * subject
        elif kind == "landscape":
            # Scenic heuristic: no prominent people, detailed image and useful colour.
            no_people = 1.0 - people[index] if item.metrics.person_score > 0 else 1.0
            subject = 0.50 * no_people + 0.28 * edges[index] + 0.22 * saturation[index]
            final = 0.64 * quality + 0.36 * subject
        elif kind == "situation":
            # "Situation" means action/event-like moments: visible motion and scene detail,
            # but final quality still strongly favours a sharp usable frame.
            subject = 0.52 * motion[index] + 0.30 * edges[index] + 0.18 * saturation[index]
            final = 0.62 * quality + 0.38 * subject
        else:
            raise ValueError(f"Unknown capture kind: {kind}")

        item.subject_score = float(max(0.0, min(1.0, subject)))
        item.final_score = float(max(0.0, min(1.0, final)))


def _build_scan_command(
    video: Path,
    info: MediaInfo,
    config: CaptureConfig,
    backend: ScanBackend,
) -> tuple[list[str], int, int]:
    ffmpeg = require_binary("ffmpeg")
    out_w, out_h = _scaled_size(info.width, info.height, config.analysis_max_side)
    fps = 1.0 / config.sample_interval_s

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if backend.hwaccel is not None:
        command.extend(["-hwaccel", backend.hwaccel])
        if backend.hwaccel == "cuda" and backend.gpu_scale:
            command.extend(["-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(video), "-an", "-sn"])

    if backend.hwaccel == "cuda" and backend.gpu_scale:
        filters = (
            f"scale_cuda={out_w}:{out_h}:format=nv12,hwdownload,format=nv12,"
            f"fps={fps:.8f},format=bgr24"
        )
    else:
        filters = f"fps={fps:.8f},scale={out_w}:{out_h}:flags=area,format=bgr24"

    command.extend(["-vf", filters, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    return command, out_w, out_h


def _scan_candidates(
    video: Path,
    info: MediaInfo,
    config: CaptureConfig,
    *,
    kind: str,
    matcher: FaceMatcher | None,
    progress: bool,
) -> tuple[list[CaptureCandidate], ScanBackend, float]:
    need_people = kind in {"person", "landscape"} and matcher is None
    person_detector = GenericPersonDetector() if need_people else None
    last_error: RuntimeError | None = None

    for backend in _backend_candidates(config.hardware_decode):
        command, out_w, out_h = _build_scan_command(video, info, config, backend)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Unable to start capture candidate analysis")
        frame_size = out_w * out_h * 3
        candidates: list[CaptureCandidate] = []
        previous_gray: np.ndarray | None = None
        index = 0
        started = time.perf_counter()
        expected = max(1, int(math.ceil(info.duration / config.sample_interval_s)))
        try:
            while True:
                payload = process.stdout.read(frame_size)
                if not payload:
                    break
                if len(payload) != frame_size:
                    raise RuntimeError("FFmpeg returned an incomplete capture-analysis frame")
                frame = np.frombuffer(payload, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
                timestamp = min(info.duration, index * config.sample_interval_s)
                metrics, previous_gray = _measure_frame(
                    frame,
                    previous_gray,
                    person_detector=person_detector,
                    matcher=matcher,
                )
                candidates.append(CaptureCandidate(timestamp=timestamp, metrics=metrics))
                index += 1
                if progress and (index % 25 == 0 or index >= expected):
                    elapsed = max(1e-9, time.perf_counter() - started)
                    done_s = min(info.duration, index * config.sample_interval_s)
                    speed = done_s / elapsed
                    percent = min(100.0, 100.0 * index / expected)
                    eta = max(0.0, info.duration - done_s) / speed if speed > 0 else 0.0
                    print(
                        f"\r  captures scan: {percent:5.1f}%  {speed:5.1f}x  ETA {eta:5.0f}s",
                        end="",
                        flush=True,
                    )
        finally:
            process.stdout.close()

        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        elapsed = time.perf_counter() - started
        if progress:
            print()
        if return_code == 0:
            return candidates, backend, elapsed
        last_error = RuntimeError(
            f"FFmpeg capture analysis failed using decoder={backend.decoder}:\n{stderr[-3000:]}"
        )
        if progress:
            print(f"  capture hardware path failed ({backend.decoder}); trying fallback...")

    assert last_error is not None
    raise last_error


def _automatic_gap(duration: float, count: int) -> float:
    if count <= 1:
        return 0.0
    # Broad enough to avoid near-identical shots, but permissive enough to fill the quota.
    return max(1.0, duration / max(1.0, count * 2.5))


def _select_diverse(
    candidates: list[CaptureCandidate],
    *,
    count: int,
    min_gap_s: float,
    edge_guard_s: float,
    duration: float,
    kind: str,
    specific_reference: bool,
) -> list[CaptureCandidate]:
    eligible = [
        item
        for item in candidates
        if edge_guard_s <= item.timestamp <= max(edge_guard_s, duration - edge_guard_s)
    ]
    if kind == "person":
        if specific_reference:
            eligible = [item for item in eligible if item.metrics.reference_matched]
        else:
            eligible = [item for item in eligible if item.metrics.person_score > 0.0]

    ranked = sorted(eligible, key=lambda item: item.final_score, reverse=True)
    for gap_factor in (1.0, 0.70, 0.45, 0.20, 0.0):
        selected: list[CaptureCandidate] = []
        gap = min_gap_s * gap_factor
        for item in ranked:
            if all(abs(item.timestamp - other.timestamp) >= gap for other in selected):
                selected.append(item)
                if len(selected) >= count:
                    return selected
        if len(selected) >= count or gap_factor == 0.0:
            return selected[:count]
    return []


def _absolute_quality(metrics: CaptureMetrics) -> float:
    sharp = 1.0 - math.exp(-max(0.0, metrics.sharpness) / 260.0)
    contrast = min(1.0, metrics.contrast / 70.0)
    entropy = min(1.0, metrics.entropy / 7.5)
    stability = math.exp(-metrics.motion * 8.0)
    return float(
        0.46 * sharp
        + 0.19 * metrics.exposure
        + 0.12 * metrics.clipping
        + 0.09 * contrast
        + 0.07 * entropy
        + 0.07 * stability
    )


def _read_frame(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def _refine_candidate(
    video: Path,
    candidate: CaptureCandidate,
    info: MediaInfo,
    config: CaptureConfig,
    *,
    kind: str,
    matcher: FaceMatcher | None,
) -> CaptureCandidate:
    window = max(0.0, config.refine_window_s)
    fps = max(1.0, config.refine_fps)
    start = max(config.edge_guard_s, candidate.timestamp - window)
    end = min(max(0.0, info.duration - config.edge_guard_s), candidate.timestamp + window)
    if end <= start + 1e-9:
        return candidate

    person_detector = GenericPersonDetector() if kind in {"person", "landscape"} and matcher is None else None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return candidate
    local: list[CaptureCandidate] = []
    previous_gray: np.ndarray | None = None
    try:
        timestamp = start
        step = 1.0 / fps
        while timestamp <= end + 1e-9:
            frame = _read_frame(cap, timestamp)
            if frame is not None:
                # Keep refinement reasonably cheap while preserving blur/detail cues.
                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > 1280:
                    scale = 1280.0 / longest
                    frame = cv2.resize(
                        frame,
                        (max(2, int(width * scale)), max(2, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                metrics, previous_gray = _measure_frame(
                    frame,
                    previous_gray,
                    person_detector=person_detector,
                    matcher=matcher,
                )
                local.append(CaptureCandidate(timestamp=timestamp, metrics=metrics))
            timestamp += step
    finally:
        cap.release()

    if not local:
        return candidate

    best: CaptureCandidate | None = None
    best_score = -1.0
    for item in local:
        if kind == "person" and matcher is not None and not item.metrics.reference_matched:
            continue
        if kind == "person" and matcher is None and item.metrics.person_score <= 0:
            continue
        subject = 0.5
        if kind == "person":
            if matcher is not None:
                subject = max(0.0, min(1.0, (item.metrics.reference_score + 1.0) / 2.0))
            else:
                subject = item.metrics.person_score
        elif kind == "landscape":
            subject = 0.55 * (1.0 - item.metrics.person_score) + 0.25 * min(1.0, item.metrics.edge_ratio / 0.12) + 0.20 * item.metrics.saturation
        elif kind == "situation":
            subject = 0.55 * min(1.0, item.metrics.motion / 0.10) + 0.30 * min(1.0, item.metrics.edge_ratio / 0.12) + 0.15 * item.metrics.saturation
        score = 0.82 * _absolute_quality(item.metrics) + 0.18 * subject
        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return candidate
    best.quality_score = _absolute_quality(best.metrics)
    best.subject_score = candidate.subject_score
    best.final_score = candidate.final_score
    return best


def analyze_captures(
    video: Path,
    config: CaptureConfig,
    *,
    count: int,
    kind: str,
    matcher: FaceMatcher | None = None,
    min_gap_s: float | None = None,
    progress: bool = True,
) -> CaptureResult:
    if count <= 0:
        raise ValueError("capture count must be greater than zero")
    if kind not in CAPTURE_KINDS:
        raise ValueError(f"capture kind must be one of: {', '.join(CAPTURE_KINDS)}")
    if config.sample_interval_s <= 0:
        raise ValueError("captures.sample_interval_s must be greater than zero")

    info = probe_media(video)
    started = time.perf_counter()
    candidates, backend, _scan_elapsed = _scan_candidates(
        video,
        info,
        config,
        kind=kind,
        matcher=matcher,
        progress=progress,
    )
    _score_candidates(candidates, kind, config)
    gap = min_gap_s if min_gap_s is not None else config.min_gap_s
    if gap <= 0:
        gap = _automatic_gap(info.duration, count)

    selected = _select_diverse(
        candidates,
        count=count,
        min_gap_s=gap,
        edge_guard_s=config.edge_guard_s,
        duration=info.duration,
        kind=kind,
        specific_reference=matcher is not None,
    )

    refined = [
        _refine_candidate(video, item, info, config, kind=kind, matcher=matcher)
        for item in selected
    ]
    refined.sort(key=lambda item: item.timestamp)
    elapsed = time.perf_counter() - started
    return CaptureResult(
        selected=refined,
        elapsed_s=elapsed,
        decoder=backend.decoder,
        gpu_scale=backend.gpu_scale,
        sampled_frames=len(candidates),
        sample_fps=1.0 / config.sample_interval_s,
        min_gap_s=gap,
    )


def extract_capture(
    video: Path,
    timestamp: float,
    destination: Path,
    *,
    image_format: str,
    jpeg_qscale: int,
    overwrite: bool,
) -> None:
    image_format = image_format.lower()
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"image format must be one of: {', '.join(IMAGE_FORMATS)}")
    ffmpeg = require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{max(0.0, timestamp):.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-sn",
    ]
    if image_format == "jpg":
        command.extend(["-q:v", str(max(1, min(31, jpeg_qscale)))])
    command.append(str(destination))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Unable to export capture {destination}:\n{result.stderr[-2000:]}")


def export_captures(
    video: Path,
    result: CaptureResult,
    output_dir: Path,
    *,
    image_format: str = "jpg",
    jpeg_qscale: int = 2,
    overwrite: bool = False,
    start_index: int = 1,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, item in enumerate(result.selected, start=start_index):
        destination = output_dir / f"s{index:03d}_{video.stem}.{image_format}"
        extract_capture(
            video,
            item.timestamp,
            destination,
            image_format=image_format,
            jpeg_qscale=jpeg_qscale,
            overwrite=overwrite,
        )
        paths.append(destination)
    return paths


def create_capture_directory(output_root: Path) -> Path:
    """Create the next ``capturesNNN`` directory directly below *output_root*."""
    output_root.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in output_root.iterdir():
        match = _CAPTURE_DIRECTORY_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            highest = max(highest, int(match.group(1)))

    index = highest + 1
    while True:
        directory = output_root / f"captures{index:03d}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            # Another invocation may have created the same number after our scan.
            index += 1


def write_capture_manifest(
    destination: Path,
    *,
    source: Path,
    kind: str,
    result: CaptureResult,
    files: list[Path],
    reference: list[Path] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("mode") != "captures":
            raise ValueError(f"Cannot append capture results to {destination}: incompatible manifest")
        batches = payload.setdefault("batches", [])
        if not isinstance(batches, list):
            raise ValueError(f"Cannot append capture results to {destination}: invalid batches field")
    else:
        payload = {"mode": "captures", "batches": []}
        batches = payload["batches"]

    def relative_file(index: int) -> str | None:
        if index >= len(files):
            return None
        try:
            return files[index].relative_to(destination.parent).as_posix()
        except ValueError:
            return str(files[index])

    batches.append({
        "source": str(source),
        "kind": kind,
        "reference_images": [str(path) for path in (reference or [])],
        "analysis": result.as_dict(),
        "captures": [
            {
                **item.as_dict(),
                "file": relative_file(index),
            }
            for index, item in enumerate(result.selected)
        ],
    })
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
