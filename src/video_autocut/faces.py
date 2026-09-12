from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FaceMatchResult:
    matched: bool
    best_score: float
    faces_detected: int


class FaceMatcher:
    def __init__(
        self,
        detector_model: Path,
        recognizer_model: Path,
        reference_paths: list[Path],
        detection_threshold: float = 0.80,
        similarity_threshold: float = 0.363,
        analysis_max_side: int = 1280,
    ) -> None:
        if not detector_model.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {detector_model}. Run python scripts/download_models.py"
            )
        if not recognizer_model.exists():
            raise FileNotFoundError(
                f"SFace model not found: {recognizer_model}. Run python scripts/download_models.py"
            )
        if not reference_paths:
            raise ValueError("At least one reference image is required")

        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), detection_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")
        self.similarity_threshold = similarity_threshold
        self.analysis_max_side = analysis_max_side
        self.reference_features = self._load_reference_features(reference_paths)


    @staticmethod
    def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
        if max_side <= 0:
            return frame
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest <= max_side:
            return frame
        scale = max_side / float(longest)
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces

    def _feature(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned)
        return feature.clone() if hasattr(feature, "clone") else feature.copy()

    def _load_reference_features(self, paths: list[Path]) -> list[np.ndarray]:
        features: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unable to read reference image: {path}")
            # Try a compact reference image first so a close-up face does not become
            # excessively large for YuNet. Fall back to the configured analysis size.
            candidates = [self._resize(image, min(self.analysis_max_side, 640))]
            if self.analysis_max_side > 640:
                candidates.append(self._resize(image, self.analysis_max_side))
            candidates.append(image)
            selected_image = None
            faces = None
            for candidate in candidates:
                candidate_faces = self._detect(candidate)
                if len(candidate_faces) > 0:
                    selected_image = candidate
                    faces = candidate_faces
                    break
            if selected_image is None or faces is None:
                raise ValueError(f"No face detected in reference image: {path}")
            # Reference images should contain one clear target face. If there are several,
            # use the largest face to avoid silently learning a background person.
            face = max(faces, key=lambda row: float(row[2] * row[3]))
            features.append(self._feature(selected_image, face))
        return features

    def match_frame(self, frame: np.ndarray) -> FaceMatchResult:
        frame = self._resize(frame, self.analysis_max_side)
        faces = self._detect(frame)
        best = -1.0
        for face in faces:
            feature = self._feature(frame, face)
            for reference in self.reference_features:
                score = float(
                    self.recognizer.match(
                        reference,
                        feature,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                )
                best = max(best, score)
                if score >= self.similarity_threshold:
                    return FaceMatchResult(True, best, len(faces))
        return FaceMatchResult(False, best, len(faces))


def discover_references(path: Path) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if path.is_file():
        if path.suffix.lower() not in image_extensions:
            raise ValueError(f"Unsupported reference image extension: {path.suffix}")
        return [path.resolve()]
    if path.is_dir():
        references = sorted(
            item.resolve()
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in image_extensions
        )
        if not references:
            raise ValueError(f"No reference images found in: {path}")
        return references
    raise FileNotFoundError(f"Reference path does not exist: {path}")


def resolve_reference_path(reference: Path, input_path: Path) -> Path:
    """Resolve a reference, preferring a relative path beside the input media."""
    reference = reference.expanduser()

    if not reference.is_absolute():
        base = input_path if input_path.is_dir() else input_path.parent
        candidate = (base / reference).expanduser()
        if candidate.exists():
            return candidate.resolve()

    if reference.exists():
        return reference.resolve()

    raise FileNotFoundError(
        f"Reference path does not exist: {reference}. "
        "If the image is next to the input videos, pass its filename with --reference."
    )
