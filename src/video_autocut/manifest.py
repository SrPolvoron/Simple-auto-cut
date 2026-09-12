from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .segments import Segment


def write_manifest(
    destination: Path,
    source: Path,
    mode: str,
    segments: list[Segment],
    extra: dict | None = None,
) -> None:
    payload = {
        "source": str(source),
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments": [segment.as_dict() for segment in segments],
    }
    if extra:
        payload.update(extra)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
