from __future__ import annotations

from pathlib import Path
import html
import math
import subprocess

import cv2
import numpy as np

from .media import probe_media, require_binary
from .segments import Segment


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000.0))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _scaled_size(width: int | None, height: int | None, target_width: int) -> tuple[int, int]:
    if not width or not height or width <= 0 or height <= 0:
        return target_width, max(2, int(target_width * 9 / 16))
    target_height = max(2, int(round(height * target_width / width)))
    if target_height % 2:
        target_height += 1
    return target_width, target_height


def _extract_thumbnail(video: Path, timestamp: float, width: int) -> np.ndarray:
    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp):.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-vf",
        f"scale={width}:-2:flags=area",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Unable to extract preview frame at {timestamp:.3f}s:\n{stderr[-2000:]}")
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode preview frame at {timestamp:.3f}s")
    return image


def _labeled_thumbnail(image: np.ndarray, label: str, bar_height: int = 28) -> np.ndarray:
    height, width = image.shape[:2]
    canvas = np.zeros((height + bar_height, width, 3), dtype=np.uint8)
    canvas[:height] = image
    cv2.putText(
        canvas,
        label,
        (8, height + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _save_grid_page(
    cells: list[np.ndarray],
    destination: Path,
    *,
    columns: int,
    gap: int = 8,
    margin: int = 12,
) -> None:
    if not cells:
        return
    cell_h = max(cell.shape[0] for cell in cells)
    cell_w = max(cell.shape[1] for cell in cells)
    rows = int(math.ceil(len(cells) / columns))
    page_h = margin * 2 + rows * cell_h + max(0, rows - 1) * gap
    page_w = margin * 2 + columns * cell_w + max(0, columns - 1) * gap
    page = np.full((page_h, page_w, 3), 24, dtype=np.uint8)

    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        y = margin + row * (cell_h + gap)
        x = margin + col * (cell_w + gap)
        page[y : y + cell.shape[0], x : x + cell.shape[1]] = cell

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), page, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise RuntimeError(f"Unable to write preview image: {destination}")


def generate_cut_preview(
    video: Path,
    dark_segments: list[Segment],
    output_dir: Path,
    *,
    width: int = 320,
    context_s: float = 0.75,
    cuts_per_page: int = 8,
) -> list[Path]:
    """Create pages with before / middle-of-cut / after thumbnails for each cut."""
    if not dark_segments:
        return []
    info = probe_media(video)
    duration = info.duration
    pages: list[Path] = []
    cells: list[np.ndarray] = []
    page_index = 1

    # Seeking too close to the container duration can land after the last decodable
    # frame. Keep at least one nominal frame interval of margin when FPS is known.
    frame_guard = (1.0 / info.fps) if info.fps and info.fps > 0 else 0.10
    latest_frame_time = max(0.0, duration - max(0.05, frame_guard))

    for cut_index, segment in enumerate(dark_segments, start=1):
        before_time = min(latest_frame_time, max(0.0, segment.start - context_s))
        middle_time = min(latest_frame_time, max(0.0, (segment.start + segment.end) / 2.0))
        after_time = min(latest_frame_time, max(0.0, segment.end + context_s))
        points = [
            (before_time, f"C{cut_index:03d} before {format_timestamp(before_time)}"),
            (middle_time, f"C{cut_index:03d} CUT {format_timestamp(segment.start)} - {format_timestamp(segment.end)}"),
            (after_time, f"C{cut_index:03d} after  {format_timestamp(after_time)}"),
        ]
        for timestamp, label in points:
            cells.append(_labeled_thumbnail(_extract_thumbnail(video, timestamp, width), label))

        if cut_index % cuts_per_page == 0 or cut_index == len(dark_segments):
            destination = output_dir / f"preview_cuts_{page_index:03d}.jpg"
            _save_grid_page(cells, destination, columns=3)
            pages.append(destination)
            cells = []
            page_index += 1

    return pages


def _contact_sheet_command(
    video: Path,
    *,
    every_s: float,
    width: int,
    height: int,
    decoder: str,
    gpu_scale: bool,
) -> list[str]:
    ffmpeg = require_binary("ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if decoder in {"cuda", "d3d11va"}:
        command.extend(["-hwaccel", decoder])
        if decoder == "cuda" and gpu_scale:
            command.extend(["-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(video), "-an", "-sn"])

    fps = 1.0 / every_s
    if decoder == "cuda" and gpu_scale:
        filters = (
            f"scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12,"
            f"fps={fps:.8f},format=bgr24"
        )
    else:
        filters = f"fps={fps:.8f},scale={width}:{height}:flags=area,format=bgr24"

    command.extend(
        [
            "-vf",
            filters,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )
    return command


def generate_contact_sheet(
    video: Path,
    output_dir: Path,
    *,
    every_s: float,
    width: int = 320,
    columns: int = 5,
    rows_per_page: int = 8,
    decoder: str = "cpu",
    gpu_scale: bool = False,
) -> list[Path]:
    """Generate paginated color thumbnails sampled at a fixed interval in one pass."""
    if every_s <= 0:
        raise ValueError("contact-sheet interval must be greater than zero")

    info = probe_media(video)
    out_w, out_h = _scaled_size(info.width, info.height, width)
    command = _contact_sheet_command(
        video,
        every_s=every_s,
        width=out_w,
        height=out_h,
        decoder=decoder,
        gpu_scale=gpu_scale,
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Unable to start contact-sheet extraction")

    frame_size = out_w * out_h * 3
    cells: list[np.ndarray] = []
    pages: list[Path] = []
    page_capacity = columns * rows_per_page
    sample_index = 0
    page_index = 1

    try:
        while True:
            payload = process.stdout.read(frame_size)
            if not payload:
                break
            if len(payload) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete contact-sheet frame")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
            timestamp = sample_index * every_s
            cells.append(_labeled_thumbnail(frame, format_timestamp(timestamp)))
            sample_index += 1

            if len(cells) >= page_capacity:
                destination = output_dir / f"contact_sheet_{page_index:03d}.jpg"
                _save_grid_page(cells, destination, columns=columns)
                pages.append(destination)
                cells = []
                page_index += 1
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Contact-sheet generation failed:\n{stderr[-3000:]}")

    if cells:
        destination = output_dir / f"contact_sheet_{page_index:03d}.jpg"
        _save_grid_page(cells, destination, columns=columns)
        pages.append(destination)

    return pages


def write_html_report(
    destination: Path,
    *,
    source: Path,
    keep_segments: list[Segment],
    dark_segments: list[Segment],
    analysis: dict,
    cut_preview_pages: list[Path] | None = None,
    contact_sheet_pages: list[Path] | None = None,
    export_info: dict | None = None,
) -> None:
    cut_preview_pages = cut_preview_pages or []
    contact_sheet_pages = contact_sheet_pages or []
    export_info = export_info or {}

    def rows(segments: list[Segment]) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(format_timestamp(segment.start))}</td>"
            f"<td>{html.escape(format_timestamp(segment.end))}</td>"
            f"<td>{segment.duration:.3f}s</td>"
            "</tr>"
            for index, segment in enumerate(segments, start=1)
        )

    def image_section(title: str, paths: list[Path]) -> str:
        if not paths:
            return ""
        images = "\n".join(
            f'<a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" alt="{html.escape(path.name)}"></a>'
            for path in paths
        )
        return f"<h2>{html.escape(title)}</h2><div class=\"images\">{images}</div>"

    stats_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in analysis.items()
    )
    export_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in export_info.items()
    )
    export_section = f"<h2>Export</h2><table>{export_rows}</table>" if export_rows else ""

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simple Video Autocut report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#111;color:#eee;max-width:1200px;margin:0 auto;padding:24px}}
a{{color:#9dc4ff}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}} th,td{{border:1px solid #444;padding:8px;text-align:left}} th{{background:#222}} .images{{display:grid;grid-template-columns:1fr;gap:16px}} img{{max-width:100%;height:auto;border:1px solid #444}} code{{background:#222;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body>
<h1>Simple Video Autocut report</h1>
<p><strong>Source:</strong> <code>{html.escape(str(source))}</code></p>
{export_section}
<h2>Analysis</h2><table>{stats_rows}</table>
<h2>Removed dark intervals ({len(dark_segments)})</h2>
<table><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th></tr>{rows(dark_segments)}</table>
<h2>Exported keep intervals ({len(keep_segments)})</h2>
<table><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th></tr>{rows(keep_segments)}</table>
{image_section('Cut previews', cut_preview_pages)}
{image_section('Contact sheets', contact_sheet_pages)}
</body>
</html>
"""
    destination.write_text(document, encoding="utf-8")


def generate_segment_preview(
    video: Path,
    segments: list[Segment],
    output_dir: Path,
    *,
    width: int = 320,
    columns: int = 4,
    rows_per_page: int = 6,
    prefix: str = "preview_selected",
) -> list[Path]:
    """Create paginated midpoint thumbnails for selected/kept segments."""
    if not segments:
        return []

    pages: list[Path] = []
    cells: list[np.ndarray] = []
    page_capacity = columns * rows_per_page
    page_index = 1

    for index, segment in enumerate(segments, start=1):
        timestamp = (segment.start + segment.end) / 2.0
        label = (
            f"S{index:03d} {format_timestamp(segment.start)} - "
            f"{format_timestamp(segment.end)}"
        )
        cells.append(_labeled_thumbnail(_extract_thumbnail(video, timestamp, width), label))
        if len(cells) >= page_capacity:
            destination = output_dir / f"{prefix}_{page_index:03d}.jpg"
            _save_grid_page(cells, destination, columns=columns)
            pages.append(destination)
            cells = []
            page_index += 1

    if cells:
        destination = output_dir / f"{prefix}_{page_index:03d}.jpg"
        _save_grid_page(cells, destination, columns=columns)
        pages.append(destination)

    return pages


def write_selection_report(
    destination: Path,
    *,
    source: Path,
    mode: str,
    selected_segments: list[Segment],
    export_info: dict,
    preview_pages: list[Path] | None = None,
    extra_rows: dict | None = None,
) -> None:
    preview_pages = preview_pages or []
    extra_rows = extra_rows or {}

    segment_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(format_timestamp(segment.start))}</td>"
        f"<td>{html.escape(format_timestamp(segment.end))}</td>"
        f"<td>{segment.duration:.3f}s</td>"
        "</tr>"
        for index, segment in enumerate(selected_segments, start=1)
    )
    export_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in export_info.items()
    )
    extra_html = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in extra_rows.items()
    )
    images = "\n".join(
        f'<a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" alt="{html.escape(path.name)}"></a>'
        for path in preview_pages
    )
    image_section = f'<h2>Preview</h2><div class="images">{images}</div>' if images else ""

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simple Video Autocut report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#111;color:#eee;max-width:1200px;margin:0 auto;padding:24px}}
a{{color:#9dc4ff}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}} th,td{{border:1px solid #444;padding:8px;text-align:left}} th{{background:#222}} .images{{display:grid;grid-template-columns:1fr;gap:16px}} img{{max-width:100%;height:auto;border:1px solid #444}} code{{background:#222;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body>
<h1>Simple Video Autocut report</h1>
<p><strong>Mode:</strong> {html.escape(mode)}</p>
<p><strong>Source:</strong> <code>{html.escape(str(source))}</code></p>
<h2>Export</h2><table>{export_rows}{extra_html}</table>
<h2>Selected intervals ({len(selected_segments)})</h2>
<table><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th></tr>{segment_rows}</table>
{image_section}
</body>
</html>
"""
    destination.write_text(document, encoding="utf-8")


def generate_capture_preview(
    capture_paths: list[Path],
    timestamps: list[float],
    scores: list[float],
    output_dir: Path,
    *,
    width: int = 320,
    columns: int = 4,
    rows_per_page: int = 5,
) -> list[Path]:
    """Build contact-sheet pages from already exported still captures."""
    if not capture_paths:
        return []
    page_capacity = max(1, columns * rows_per_page)
    pages: list[Path] = []
    cells: list[np.ndarray] = []
    page_index = 1

    for index, path in enumerate(capture_paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, source_width = image.shape[:2]
        target_height = max(2, int(round(height * width / max(1, source_width))))
        resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
        timestamp = timestamps[index] if index < len(timestamps) else 0.0
        score = scores[index] if index < len(scores) else 0.0
        label = f"S{index + 1:03d} {format_timestamp(timestamp)} score={score:.3f}"
        cells.append(_labeled_thumbnail(resized, label))

        if len(cells) >= page_capacity:
            destination = output_dir / f"captures_preview_{page_index:03d}.jpg"
            _save_grid_page(cells, destination, columns=columns)
            pages.append(destination)
            cells = []
            page_index += 1

    if cells:
        destination = output_dir / f"captures_preview_{page_index:03d}.jpg"
        _save_grid_page(cells, destination, columns=columns)
        pages.append(destination)

    return pages
