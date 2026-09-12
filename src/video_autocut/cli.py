from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .black import analyze_black
from .captures import (
    CAPTURE_KINDS,
    IMAGE_FORMATS,
    analyze_captures,
    create_capture_directory,
    export_captures,
    write_capture_manifest,
)
from .config import AppConfig, load_config
from .exporter import EXPORT_MODES, ExportArtifacts, export_segments
from .faces import FaceMatcher, discover_references, resolve_reference_path
from .manifest import write_manifest
from .media import discover_videos, output_directory, require_binary
from .person import find_person_scenes
from .scenes import split_scenes
from .trim import plan_video_trim
from .preview import (
    generate_contact_sheet,
    generate_capture_preview,
    generate_cut_preview,
    generate_segment_preview,
    write_html_report,
    write_selection_report,
)


def _add_after_mode_job_options(parser: argparse.ArgumentParser) -> None:
    """Allow common job options both before and after the subcommand."""
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=argparse.SUPPRESS,
        help="Override output root directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Analyze only; do not create video outputs",
    )


def _add_export_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        dest="export_mode",
        choices=EXPORT_MODES,
        default="clips",
        help=(
            "Video output mode: clips = c001/c002..., montage = one joined video with the "
            "original filename, both = create both (default: clips)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-autocut",
        description=(
            "Detect useful video intervals and export individual stream-copy clips, "
            "a joined montage, or both."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration file (default: ./config.toml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Override output root directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not create video outputs")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    person = subparsers.add_parser("person", help="Keep complete shots containing a reference person")
    _add_after_mode_job_options(person)
    _add_export_option(person)
    person.add_argument("input", type=Path, help="Video file or directory containing videos")
    person.add_argument(
        "--reference",
        "-r",
        required=True,
        type=Path,
        help=(
            "Reference image or directory of reference images. A relative filename is also "
            "looked up next to the input video(s)."
        ),
    )
    person.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override SFace cosine similarity threshold",
    )
    person.add_argument(
        "--preview",
        action="store_true",
        help="Generate thumbnail pages for selected person scenes plus report.html",
    )
    person.add_argument(
        "--report",
        action="store_true",
        help="Generate report.html for selected person scenes",
    )
    person.add_argument(
        "--thumbnail-width",
        type=int,
        default=320,
        help="Preview thumbnail width in pixels (default: 320)",
    )

    black = subparsers.add_parser(
        "black",
        help="Remove near-black intervals and keep the remaining sections",
    )
    _add_after_mode_job_options(black)
    _add_export_option(black)
    black.add_argument("input", type=Path, help="Video file or directory containing videos")
    black.add_argument(
        "--hwaccel",
        choices=["auto", "cpu", "cuda", "d3d11va"],
        default=None,
        help="Override decoder acceleration for black analysis",
    )
    black.add_argument(
        "--scan-fps",
        type=float,
        default=None,
        help="Override coarse scan FPS (default: 2)",
    )
    black.add_argument(
        "--refine-fps",
        type=float,
        default=None,
        help="Override boundary refinement FPS (default: 20)",
    )
    black.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip high-FPS boundary refinement for maximum analysis speed",
    )
    black.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress and ETA output",
    )
    black.add_argument(
        "--preview",
        action="store_true",
        help="Generate cut-preview JPG pages and report.html",
    )
    black.add_argument(
        "--preview-cuts",
        action="store_true",
        help="Generate before/cut/after thumbnail pages for every removed interval",
    )
    black.add_argument(
        "--contact-sheet",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Generate color thumbnails across the whole video every N seconds",
    )
    black.add_argument(
        "--report",
        action="store_true",
        help="Generate report.html with timestamps, statistics and any preview images",
    )
    black.add_argument(
        "--thumbnail-width",
        type=int,
        default=320,
        help="Preview thumbnail width in pixels (default: 320)",
    )

    scenes = subparsers.add_parser(
        "scenes",
        aliases=["scene"],
        help="Split compilations into individual scene clips",
    )
    _add_after_mode_job_options(scenes)
    scenes.add_argument("input", type=Path, help="Video file or directory containing videos")
    scenes.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override scene-change sensitivity (lower = more cuts, default: 0.30)",
    )
    scenes.add_argument(
        "--min-scene",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Merge scene fragments shorter than this duration (default: 0.25)",
    )
    scenes.add_argument(
        "--preview",
        action="store_true",
        help="Generate scene thumbnail pages plus report.html",
    )
    scenes.add_argument(
        "--report",
        action="store_true",
        help="Generate report.html with scene timestamps",
    )
    scenes.add_argument(
        "--thumbnail-width",
        type=int,
        default=320,
        help="Preview thumbnail width in pixels (default: 320)",
    )

    trim = subparsers.add_parser(
        "trim",
        help="Batch-remove seconds from the beginning, end and/or exact middle of videos",
    )
    _add_after_mode_job_options(trim)
    _add_export_option(trim)
    trim.set_defaults(export_mode="montage")
    trim.add_argument("input", type=Path, help="Video file or directory containing videos")
    trim.add_argument(
        "--remove-start",
        type=float,
        metavar="SECONDS",
        default=0.0,
        help="Remove N seconds from the beginning of every video",
    )
    trim.add_argument(
        "--remove-end",
        type=float,
        metavar="SECONDS",
        default=0.0,
        help="Remove N seconds from the end of every video",
    )
    trim.add_argument(
        "--remove-middle",
        type=float,
        metavar="SECONDS",
        default=0.0,
        help="Remove N seconds centered on the exact midpoint of every video",
    )
    trim.add_argument(
        "--preview",
        action="store_true",
        help="Generate before/cut/after thumbnail pages plus report.html",
    )
    trim.add_argument(
        "--report",
        action="store_true",
        help="Generate report.html with kept timestamps",
    )
    trim.add_argument(
        "--thumbnail-width",
        type=int,
        default=320,
        help="Preview thumbnail width in pixels (default: 320)",
    )

    captures = subparsers.add_parser(
        "captures",
        aliases=["capture", "screenshots", "screenshot"],
        help="Select and export the best still frames from each video",
    )
    _add_after_mode_job_options(captures)
    captures.add_argument("input", type=Path, help="Video file or directory containing videos")
    captures.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of captures per video (default: 5)",
    )
    captures.add_argument(
        "--kind",
        choices=CAPTURE_KINDS,
        default="quality",
        help=(
            "Selection target: quality, person, landscape or situation. "
            "Situation favours action/event-like moments."
        ),
    )
    captures.add_argument(
        "--reference",
        "-r",
        type=Path,
        default=None,
        help=(
            "Optional reference image/directory for --kind person. Relative paths are "
            "looked up beside the input videos."
        ),
    )
    captures.add_argument(
        "--scan-every",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Coarse candidate interval in seconds (default: 1.0)",
    )
    captures.add_argument(
        "--refine-window",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Search +/- N seconds around every selected candidate (default: 0.50)",
    )
    captures.add_argument(
        "--refine-fps",
        type=float,
        default=None,
        help="FPS used for local best-frame refinement (default: 8)",
    )
    captures.add_argument(
        "--min-gap",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Minimum time between selected captures; default is automatic",
    )
    captures.add_argument(
        "--hwaccel",
        choices=["auto", "cpu", "cuda", "d3d11va"],
        default=None,
        help="Decoder acceleration for the coarse scan",
    )
    captures.add_argument(
        "--format",
        dest="image_format",
        choices=IMAGE_FORMATS,
        default="jpg",
        help="Output image format (default: jpg)",
    )
    captures.add_argument(
        "--preview",
        action="store_true",
        help="Generate contact-sheet pages with the selected captures and timestamps",
    )
    captures.add_argument(
        "--thumbnail-width",
        type=int,
        default=320,
        help="Preview thumbnail width in pixels (default: 320)",
    )
    captures.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable capture-analysis progress output",
    )

    return parser


def _load(args: argparse.Namespace) -> AppConfig:
    config_path = args.config
    if config_path == Path("config.toml") and not config_path.exists():
        config = load_config(None)
    else:
        config = load_config(config_path)

    if args.output is not None:
        config.general.output_dir = str(args.output)
    if args.overwrite:
        config.general.overwrite = True

    if args.mode == "person" and args.threshold is not None:
        config.person.face_similarity_threshold = args.threshold

    if args.mode == "black":
        if args.hwaccel is not None:
            config.black.hardware_decode = args.hwaccel
        if args.scan_fps is not None:
            if args.scan_fps <= 0:
                raise ValueError("--scan-fps must be greater than 0")
            config.black.sample_interval_s = 1.0 / args.scan_fps
        if args.refine_fps is not None:
            if args.refine_fps <= 0:
                raise ValueError("--refine-fps must be greater than 0")
            config.black.refine_fps = args.refine_fps
        if args.no_refine:
            config.black.refine_enabled = False
        if args.contact_sheet is not None and args.contact_sheet <= 0:
            raise ValueError("--contact-sheet must be greater than 0 seconds")

    if args.mode in {"scenes", "scene"}:
        if args.threshold is not None:
            if not 0.0 < args.threshold < 1.0:
                raise ValueError("--threshold must be between 0 and 1")
            config.scenes.threshold = args.threshold
        if args.min_scene is not None:
            if args.min_scene < 0:
                raise ValueError("--min-scene must be >= 0")
            config.scenes.min_scene_duration_s = args.min_scene

    if args.mode == "trim":
        for name in ("remove_start", "remove_end", "remove_middle"):
            if getattr(args, name) < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
        if not any(getattr(args, name) > 0 for name in ("remove_start", "remove_end", "remove_middle")):
            raise ValueError("trim requires --remove-start, --remove-end and/or --remove-middle")

    if args.mode in {"captures", "capture", "screenshots", "screenshot"}:
        if args.count <= 0:
            raise ValueError("--count must be greater than 0")
        if args.reference is not None and args.kind != "person":
            raise ValueError("--reference is only valid with --kind person")
        if args.scan_every is not None:
            if args.scan_every <= 0:
                raise ValueError("--scan-every must be greater than 0")
            config.captures.sample_interval_s = args.scan_every
        if args.refine_window is not None:
            if args.refine_window < 0:
                raise ValueError("--refine-window must be >= 0")
            config.captures.refine_window_s = args.refine_window
        if args.refine_fps is not None:
            if args.refine_fps <= 0:
                raise ValueError("--refine-fps must be greater than 0")
            config.captures.refine_fps = args.refine_fps
        if args.min_gap is not None and args.min_gap < 0:
            raise ValueError("--min-gap must be >= 0")
        if args.hwaccel is not None:
            config.captures.hardware_decode = args.hwaccel

    if getattr(args, "thumbnail_width", 320) < 120:
        raise ValueError("--thumbnail-width must be at least 120")

    return config


def _export(
    video: Path,
    segments,
    config: AppConfig,
    output_root: Path,
    dry_run: bool,
    export_mode: str,
) -> ExportArtifacts:
    return export_segments(
        video,
        list(segments),
        output_root=output_root,
        mode=export_mode,
        overwrite=config.general.overwrite,
        dry_run=dry_run,
    )


def _person_mode(args: argparse.Namespace, config: AppConfig, videos: list[Path], output_root: Path) -> None:
    reference_path = resolve_reference_path(args.reference, args.input)
    references = discover_references(reference_path)

    project_root = Path(__file__).resolve().parents[2]
    detector = project_root / "models" / "face_detection_yunet_2023mar.onnx"
    recognizer = project_root / "models" / "face_recognition_sface_2021dec.onnx"
    if not detector.exists():
        detector = Path("models/face_detection_yunet_2023mar.onnx")
    if not recognizer.exists():
        recognizer = Path("models/face_recognition_sface_2021dec.onnx")

    matcher = FaceMatcher(
        detector,
        recognizer,
        references,
        detection_threshold=config.person.face_detection_threshold,
        similarity_threshold=config.person.face_similarity_threshold,
        analysis_max_side=config.person.analysis_max_side,
    )

    print("[person] reference:", ", ".join(str(path) for path in references))
    for video in videos:
        print(f"[person] {video}")
        segments, decisions = find_person_scenes(video, matcher, config.person)
        artifacts = _export(
            video,
            segments,
            config,
            output_root,
            args.dry_run,
            args.export_mode,
        )
        out_dir = output_directory(output_root, video)
        preview_pages: list[Path] = []
        if args.preview:
            print("  generating selected-scene preview pages...")
            preview_pages = generate_segment_preview(
                video,
                segments,
                out_dir,
                width=args.thumbnail_width,
                prefix="preview_person",
            )
            print(f"  preview pages: {len(preview_pages)}")

        if args.preview or args.report:
            report_path = out_dir / "report.html"
            write_selection_report(
                report_path,
                source=video,
                mode="person",
                selected_segments=segments,
                export_info=artifacts.as_dict(),
                preview_pages=preview_pages,
                extra_rows={
                    "reference_images": ", ".join(path.name for path in references),
                    "selected_scenes": len(segments),
                },
            )
            print(f"  report: {report_path.name}")

        if config.general.write_manifest:
            write_manifest(
                out_dir / "manifest.json",
                source=video,
                mode="person",
                segments=segments,
                extra={
                    "reference_images": [str(path) for path in references],
                    "exports": artifacts.as_dict(),
                    "preview": {
                        "selected_pages": [path.name for path in preview_pages],
                        "report": "report.html" if (args.preview or args.report) else None,
                    },
                    "scene_decisions": [
                        {
                            "start": round(item.segment.start, 6),
                            "end": round(item.segment.end, 6),
                            "matching_samples": item.matching_samples,
                            "sampled_frames": item.sampled_frames,
                            "best_score": round(item.best_score, 6),
                        }
                        for item in decisions
                    ],
                },
            )
        print(f"  selected scenes: {len(segments)}")
        print(f"  export mode: {args.export_mode}")


def _generate_black_previews(
    args: argparse.Namespace,
    video: Path,
    out_dir: Path,
    result,
    artifacts: ExportArtifacts,
) -> tuple[list[Path], list[Path]]:
    cut_pages: list[Path] = []
    contact_pages: list[Path] = []

    if args.preview or args.preview_cuts:
        print("  generating cut preview pages...")
        cut_pages = generate_cut_preview(
            video,
            result.dark,
            out_dir,
            width=args.thumbnail_width,
        )
        print(f"  cut preview pages: {len(cut_pages)}")

    if args.contact_sheet is not None:
        print(f"  generating contact sheet every {args.contact_sheet:g}s...")
        try:
            contact_pages = generate_contact_sheet(
                video,
                out_dir,
                every_s=args.contact_sheet,
                width=args.thumbnail_width,
                decoder=result.stats.decoder,
                gpu_scale=result.stats.gpu_scale,
            )
        except RuntimeError:
            if result.stats.decoder == "cpu":
                raise
            print("  preview hardware path failed; retrying contact sheet on CPU...")
            contact_pages = generate_contact_sheet(
                video,
                out_dir,
                every_s=args.contact_sheet,
                width=args.thumbnail_width,
                decoder="cpu",
                gpu_scale=False,
            )
        print(f"  contact-sheet pages: {len(contact_pages)}")

    if args.preview or args.report:
        report_path = out_dir / "report.html"
        write_html_report(
            report_path,
            source=video,
            keep_segments=result.keep,
            dark_segments=result.dark,
            analysis=result.stats.as_dict(),
            cut_preview_pages=cut_pages,
            contact_sheet_pages=contact_pages,
            export_info=artifacts.as_dict(),
        )
        print(f"  report: {report_path.name}")

    return cut_pages, contact_pages


def _black_mode(args: argparse.Namespace, config: AppConfig, videos: list[Path], output_root: Path) -> None:
    for video in videos:
        print(f"[black] {video}")
        result = analyze_black(video, config.black, progress=not args.no_progress)
        artifacts = _export(
            video,
            result.keep,
            config,
            output_root,
            args.dry_run,
            args.export_mode,
        )

        out_dir = output_directory(output_root, video)
        cut_pages, contact_pages = _generate_black_previews(args, video, out_dir, result, artifacts)

        if config.general.write_manifest:
            write_manifest(
                out_dir / "manifest.json",
                source=video,
                mode="black",
                segments=result.keep,
                extra={
                    "detected_unusable_dark_intervals": [item.as_dict() for item in result.dark],
                    "analysis": result.stats.as_dict(),
                    "exports": artifacts.as_dict(),
                    "preview": {
                        "cut_pages": [path.name for path in cut_pages],
                        "contact_sheet_pages": [path.name for path in contact_pages],
                        "report": "report.html" if (args.preview or args.report) else None,
                    },
                },
            )

        stats = result.stats
        fallback = " (fallback used)" if stats.hardware_fallback else ""
        gpu_scale = ", gpu-scale=yes" if stats.gpu_scale else ", gpu-scale=no"
        print(
            f"  analysis: {stats.elapsed_s:.2f}s for {stats.video_duration_s:.2f}s "
            f"({stats.realtime_factor:.1f}x realtime), decoder={stats.decoder}{gpu_scale}{fallback}"
        )
        print(
            f"  timing: coarse={stats.coarse_elapsed_s:.2f}s, "
            f"refine={stats.refine_elapsed_s:.2f}s"
        )
        print(
            f"  samples: coarse={stats.coarse_samples} @ {stats.coarse_fps:g} FPS, "
            f"refined={stats.refined_samples} @ {stats.refine_fps:g} FPS, "
            f"refine-windows={stats.refine_windows}"
        )
        print(f"  detected unusable-dark intervals: {len(result.dark)}")
        print(f"  selected non-black intervals: {len(result.keep)}")
        print(f"  export mode: {args.export_mode}")


def _scenes_mode(args: argparse.Namespace, config: AppConfig, videos: list[Path], output_root: Path) -> None:
    for video in videos:
        print(f"[scenes] {video}")
        scenes = split_scenes(
            video,
            threshold=config.scenes.threshold,
            min_scene_duration_s=config.scenes.min_scene_duration_s,
        )
        artifacts = _export(video, scenes, config, output_root, args.dry_run, "clips")
        out_dir = output_directory(output_root, video)
        preview_pages: list[Path] = []
        if args.preview:
            print("  generating scene preview pages...")
            preview_pages = generate_segment_preview(
                video, scenes, out_dir, width=args.thumbnail_width, prefix="preview_scenes"
            )
            print(f"  preview pages: {len(preview_pages)}")
        if args.preview or args.report:
            report_path = out_dir / "report.html"
            write_selection_report(
                report_path,
                source=video,
                mode="scenes",
                selected_segments=scenes,
                export_info=artifacts.as_dict(),
                preview_pages=preview_pages,
                extra_rows={
                    "scene_threshold": config.scenes.threshold,
                    "min_scene_duration_s": config.scenes.min_scene_duration_s,
                    "detected_scenes": len(scenes),
                },
            )
            print(f"  report: {report_path.name}")
        if config.general.write_manifest:
            write_manifest(
                out_dir / "manifest.json",
                source=video,
                mode="scenes",
                segments=scenes,
                extra={
                    "scene_threshold": config.scenes.threshold,
                    "min_scene_duration_s": config.scenes.min_scene_duration_s,
                    "exports": artifacts.as_dict(),
                    "preview": {
                        "scene_pages": [path.name for path in preview_pages],
                        "report": "report.html" if (args.preview or args.report) else None,
                    },
                },
            )
        print(f"  detected scenes: {len(scenes)}")


def _trim_mode(args: argparse.Namespace, config: AppConfig, videos: list[Path], output_root: Path) -> None:
    for video in videos:
        print(f"[trim] {video}")
        plan = plan_video_trim(
            video,
            remove_start_s=args.remove_start,
            remove_end_s=args.remove_end,
            remove_middle_s=args.remove_middle,
            min_keep_duration_s=config.trim.min_keep_duration_s,
        )
        artifacts = _export(
            video, plan.keep, config, output_root, args.dry_run, args.export_mode
        )
        out_dir = output_directory(output_root, video)
        preview_pages: list[Path] = []
        if args.preview:
            print("  generating trim preview pages...")
            preview_pages = generate_cut_preview(
                video, plan.removed, out_dir, width=args.thumbnail_width
            )
            print(f"  preview pages: {len(preview_pages)}")
        if args.preview or args.report:
            report_path = out_dir / "report.html"
            write_selection_report(
                report_path,
                source=video,
                mode="trim",
                selected_segments=plan.keep,
                export_info=artifacts.as_dict(),
                preview_pages=preview_pages,
                extra_rows={
                    "source_duration_s": round(plan.duration, 6),
                    "remove_start_s": args.remove_start,
                    "remove_end_s": args.remove_end,
                    "remove_middle_s": args.remove_middle,
                    "removed_intervals": len(plan.removed),
                },
            )
            print(f"  report: {report_path.name}")
        if config.general.write_manifest:
            write_manifest(
                out_dir / "manifest.json",
                source=video,
                mode="trim",
                segments=plan.keep,
                extra={
                    "removed_intervals": [item.as_dict() for item in plan.removed],
                    "trim": {
                        "remove_start_s": args.remove_start,
                        "remove_end_s": args.remove_end,
                        "remove_middle_s": args.remove_middle,
                    },
                    "exports": artifacts.as_dict(),
                    "preview": {
                        "cut_pages": [path.name for path in preview_pages],
                        "report": "report.html" if (args.preview or args.report) else None,
                    },
                },
            )
        print(f"  kept intervals: {len(plan.keep)}")
        print(f"  export mode: {args.export_mode}")



def _capture_matcher(args: argparse.Namespace, config: AppConfig) -> tuple[FaceMatcher | None, list[Path]]:
    if args.reference is None:
        return None, []
    reference_path = resolve_reference_path(args.reference, args.input)
    references = discover_references(reference_path)
    project_root = Path(__file__).resolve().parents[2]
    detector = project_root / "models" / "face_detection_yunet_2023mar.onnx"
    recognizer = project_root / "models" / "face_recognition_sface_2021dec.onnx"
    if not detector.exists():
        detector = Path("models/face_detection_yunet_2023mar.onnx")
    if not recognizer.exists():
        recognizer = Path("models/face_recognition_sface_2021dec.onnx")
    matcher = FaceMatcher(
        detector,
        recognizer,
        references,
        detection_threshold=config.person.face_detection_threshold,
        similarity_threshold=config.person.face_similarity_threshold,
        analysis_max_side=config.person.analysis_max_side,
    )
    return matcher, references


def _captures_mode(args: argparse.Namespace, config: AppConfig, videos: list[Path], output_root: Path) -> None:
    matcher, references = _capture_matcher(args, config)
    capture_dir: Path | None = None
    next_capture_index = 1
    if references:
        print("[captures] reference:", ", ".join(str(path) for path in references))

    for video in videos:
        print(f"[captures] {video}")
        result = analyze_captures(
            video,
            config.captures,
            count=args.count,
            kind=args.kind,
            matcher=matcher,
            min_gap_s=args.min_gap,
            progress=not args.no_progress,
        )
        files: list[Path] = []
        preview_pages: list[Path] = []
        if not args.dry_run:
            if capture_dir is None:
                capture_dir = create_capture_directory(output_root)
            files = export_captures(
                video,
                result,
                capture_dir,
                image_format=args.image_format,
                jpeg_qscale=config.captures.jpeg_qscale,
                overwrite=config.general.overwrite,
                start_index=next_capture_index,
            )
            next_capture_index += len(files)
            if args.preview:
                preview_pages = generate_capture_preview(
                    files,
                    [item.timestamp for item in result.selected],
                    [item.final_score for item in result.selected],
                    capture_dir,
                    width=args.thumbnail_width,
                )
        elif args.preview:
            print("  --preview ignored during --dry-run because no still images are exported")

        write_capture_manifest(
            output_root / "manifest.json",
            source=video,
            kind=args.kind,
            result=result,
            files=files,
            reference=references,
        )
        print(
            f"  selected captures: {len(result.selected)}/{args.count}, "
            f"analysis={result.elapsed_s:.2f}s, decoder={result.decoder}, "
            f"gpu-scale={'yes' if result.gpu_scale else 'no'}"
        )
        if files:
            print(f"  images: {capture_dir}")
        if preview_pages:
            print(f"  preview pages: {len(preview_pages)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_binary("ffmpeg")
        require_binary("ffprobe")
        config = _load(args)
        input_path = args.input
        videos = discover_videos(input_path, config.general.video_extensions)
        if not videos:
            print(f"No supported videos found in {input_path}")
            return 0

        output_root = Path(config.general.output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        if args.mode == "person":
            _person_mode(args, config, videos, output_root)
        elif args.mode == "black":
            _black_mode(args, config, videos, output_root)
        elif args.mode in {"scenes", "scene"}:
            _scenes_mode(args, config, videos, output_root)
        elif args.mode == "trim":
            _trim_mode(args, config, videos, output_root)
        elif args.mode in {"captures", "capture", "screenshots", "screenshot"}:
            _captures_mode(args, config, videos, output_root)
        else:
            parser.error(f"Unknown mode: {args.mode}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
