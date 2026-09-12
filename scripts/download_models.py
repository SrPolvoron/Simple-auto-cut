from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FILES = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=true",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?download=true",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, url: str, expected: str | None) -> None:
    destination = MODELS / name
    if destination.exists():
        if expected is None or sha256(destination) == expected:
            print(f"[ok] {name} already exists")
            return
        print(f"[warn] checksum mismatch for existing {name}; downloading again")

    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {name}")
    request = urllib.request.Request(url, headers={"User-Agent": "simple-video-autocut/0.1"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as fh:
        while chunk := response.read(1024 * 1024):
            fh.write(chunk)

    if expected is not None:
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )

    tmp.replace(destination)
    print(f"[ok] {destination}")


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        for name, metadata in FILES.items():
            download(name, metadata["url"], metadata["sha256"])
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
