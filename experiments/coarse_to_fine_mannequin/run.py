"""粗探索・詳細照合マネキン検出器を、実カメラ入力で検証する。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from ClsImageViewerUDP import ClsImageViewerUDP  # noqa: E402
from experiments.coarse_to_fine_mannequin.src.artifacts import (  # noqa: E402
    AnnotatedVideoWriter,
    ArtifactSaveOptions,
    ArtifactWriter,
)
from experiments.coarse_to_fine_mannequin.src.performance import (  # noqa: E402
    FrameTiming,
    PerformanceOptions,
    PerformanceTracker,
)
from experiments.coarse_to_fine_mannequin.src.pipeline import (  # noqa: E402
    CoarseToFineMatcher,
    FrameResult,
)
from experiments.coarse_to_fine_mannequin.src.tracking import (  # noqa: E402
    TemporalMatchTracker,
)


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _create_matcher(template: np.ndarray, config: dict[str, object]) -> CoarseToFineMatcher:
    """メインプログラムと同じ検出パラメータから照合器を作る。"""
    return CoarseToFineMatcher(
        template,
        device=str(config.get("device", "cuda")),
        frame_downscale=float(config.get("frame_downscale", 0.25)),
        coarse_template_scales=config.get("coarse_template_scales", [0.2]),
        detail_template_scales=config.get("detail_template_scales", [1.0, 0.8, 0.6, 0.4]),
        coarse_stride=int(config.get("coarse_stride", 2)),
        detail_stride_base=int(config.get("detail_stride_base", config.get("detail_stride", 4))),
        detail_stride_min=int(config.get("detail_stride_min", 2)),
        coarse_top_k=int(config.get("coarse_top_k", 3)),
        coarse_candidates_per_template=int(config.get("coarse_candidates_per_template", 20)),
        nms_distance_original_px=int(config.get("nms_distance_original_px", 120)),
        roi_margin_px=int(config.get("roi_margin_px", 32)),
        final_max_error=float(config.get("final_max_error", 27.0)),
        brightness_weights=tuple(
            config.get("brightness_weights", {}).get(key, default)
            for key, default in (("r", 0.299), ("g", 0.587), ("b", 0.114))
        ),
        channel_weights=tuple(
            config.get("channel_weights", {}).get(key, default)
            for key, default in (("rg", 0.5), ("by", 0.5), ("y", 1.0))
        ),
        variance_channel_weights=tuple(
            config.get("variance_channel_weights", {}).get(key, default)
            for key, default in (("rg", 1.0), ("by", 1.0), ("y", 1.0))
        ),
        variance_log_distance_max=float(config.get("variance_log_distance_max", 3.0)),
        variance_epsilon=float(config.get("variance_epsilon", 1.0)),
        min_chroma_variance=float(config.get("min_chroma_variance", 0.0)),
        min_brightness_variance=float(config.get("min_brightness_variance", 0.0)),
        min_weight=float(config.get("min_weight", 0.05)),
    )


def _create_tracker(config: dict[str, object]) -> TemporalMatchTracker:
    """メインプログラムと同じ履歴条件で、生検出を確定する追跡器を作る。"""
    return TemporalMatchTracker(
        window_frames=int(config.get("target_confirm_window_frames", 10)),
        confirm_frames=int(config.get("target_confirm_frames", 5)),
        match_iou_thresh=float(config.get("match_iou_thresh", 0.8)),
        max_track_age=int(config.get("max_track_age", 30)),
    )


def apply_temporal_tracking(
    raw_result: FrameResult, tracker: TemporalMatchTracker
) -> FrameResult:
    """生の照合結果を履歴で MATCH / MAYBE / NO MATCH に確定する。

    ``FrameResult.detected`` はメインプログラムと同じく、確定済みMATCHだけを示す。
    ``temporal_status`` にはMAYBEを含む表示・保存用の状態を残す。
    """
    best = raw_result.best_match
    raw_match = raw_result.detected and best is not None
    bbox_xyxy = (
        (best.x, best.y, best.x + best.template.width, best.y + best.template.height)
        if raw_match and best is not None
        else None
    )
    decision = tracker.update(bbox_xyxy)
    return replace(
        raw_result,
        detected=raw_match and decision.confirmed,
        temporal_status=decision.status,
        track_id=decision.track_id,
        positive_frames=decision.positive_frames,
        history_window_frames=decision.history_window_frames,
    )


def _display_frame(
    frame_number: int,
    frame: np.ndarray,
    result: FrameResult,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    fps: float | None = None,
) -> np.ndarray:
    """表示専用の検出枠と、固定文字サイズのHUDを描く。"""
    image = frame.copy()
    status = result.temporal_status or ("MATCH" if result.detected else "NO MATCH")
    colour = (
        (0, 255, 0)
        if status == "MATCH"
        else ((0, 165, 255) if status == "MAYBE" else (0, 0, 255))
    )
    best = result.best_match
    if best is not None:
        cv2.rectangle(
            image,
            (round(best.x * scale_x), round(best.y * scale_y)),
            (
                round((best.x + best.template.width) * scale_x),
                round((best.y + best.template.height) * scale_y),
            ),
            colour,
            2,
        )
        score_text = f"score={best.score:.2f} scale={best.template.scale:.2f}"
    else:
        score_text = "score=N/A scale=N/A"
    if result.track_id is not None:
        score_text += f" track={result.track_id} history={result.positive_frames}/{result.history_window_frames}"

    status_text = f"{frame_number}: {status}"
    hud_rows: list[tuple[str, int]] = [(status_text, 28), (score_text, 52)]
    if fps is not None:
        hud_rows.append((f"FPS: {fps:.1f}", 76))
    for text, y in hud_rows:
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            image, (5, y - height - baseline - 3), (width + 11, y + baseline + 3), (0, 0, 0), -1
        )
        cv2.putText(
            image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA
        )
    return image


def _input_size(config: dict[str, object]) -> tuple[int, int] | None:
    width, height = config.get("frame_width"), config.get("frame_height")
    if (width is None) != (height is None):
        raise ValueError("frame_width and frame_height must be specified together")
    if width is None:
        return None
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    return width, height


def _display_size(config: dict[str, object]) -> tuple[int, int] | None:
    width, height = config.get("display_width"), config.get("display_height")
    if (width is None) != (height is None):
        raise ValueError("display_width and display_height must be specified together")
    if width is None:
        return None
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("display_width and display_height must be positive")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="粗探索・詳細照合マネキン位置推定の実験ランナー")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, help="設定値を上書きする処理フレーム数")
    parser.add_argument("--no-display", action="store_true", help="OpenCVウィンドウを表示しない")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    template_path = _resolve_path(str(config["template_path"]), config_path)
    template = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"cannot read template image: {template_path}")
    matcher = _create_matcher(template, config)
    tracker = _create_tracker(config)
    input_size = _input_size(config)
    display_size = _display_size(config)

    results_root = _resolve_path(str(config.get("results_root", "../results")), config_path)
    save_options = ArtifactSaveOptions.from_config(config)
    performance_options = PerformanceOptions.from_config(config)
    performance_tracker = PerformanceTracker(performance_options) if performance_options.enabled else None
    writer = ArtifactWriter(results_root, save_options)
    writer.save_detail_templates(matcher.detail_templates)

    input_source = str(config.get("input_source", "video"))
    capture: cv2.VideoCapture | None = None
    sensor: ClsImageViewerUDP | None = None
    if input_source == "udp":
        sensor = ClsImageViewerUDP()
    elif input_source in ("camera", "video"):
        source: int | str
        source = int(config.get("camera_index", 0)) if input_source == "camera" else str(_resolve_path(str(config["video_path"]), config_path))
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {input_source} input: {source}")
    else:
        raise ValueError("input_source must be 'udp', 'camera', or 'video'")

    max_frames = args.max_frames if args.max_frames is not None else config.get("max_frames")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive when specified")
    show_window = bool(config.get("show_window", True)) and not args.no_display
    window_name = str(config.get("window_name", "Coarse-to-fine mannequin matcher"))
    frame_number = 0
    annotated_video_writer: AnnotatedVideoWriter | None = None
    try:
        while True:
            loop_started = perf_counter()
            decode_started = perf_counter()
            if sensor is not None:
                sensor.receive_one_set()
                frame = sensor.get_center_surround_bgr_image()
                ok = frame is not None
            else:
                assert capture is not None
                ok, frame = capture.read()
            decode_ended = perf_counter()
            if not ok or frame is None:
                break
            if input_size is not None:
                frame = cv2.resize(frame, input_size)
            frame_number += 1

            detection_started = perf_counter()
            raw_result = matcher.process(
                frame,
                collect_coarse_image=(save_options.enabled and (save_options.coarse_frame or save_options.coarse_candidates)),
                collect_detail_variance_filters=(save_options.enabled and (save_options.variance_filters or save_options.scores_json)),
            )
            result = apply_temporal_tracking(raw_result, tracker)
            detection_ended = perf_counter()
            fps = 1.0 / max(detection_ended - detection_started, 1e-6)

            artifact_started = perf_counter()
            if save_options.enabled and save_options.annotated_video:
                if annotated_video_writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else 0.0
                    annotated_video_writer = AnnotatedVideoWriter(
                        writer.require_run_dir(), fps, frame.shape[1], frame.shape[0]
                    )
                annotated_video_writer.write(frame_number, frame, result)
            if save_options.should_save_frame(frame_number):
                writer.save_frame(frame_number, frame, result)
            artifact_ended = perf_counter()

            display_started = perf_counter()
            quit_requested = False
            if show_window:
                if display_size is not None:
                    interpolation = cv2.INTER_LINEAR if display_size[0] >= frame.shape[1] else cv2.INTER_AREA
                    display = cv2.resize(frame, display_size, interpolation=interpolation)
                    display = _display_frame(
                        frame_number,
                        display,
                        result,
                        scale_x=display_size[0] / frame.shape[1],
                        scale_y=display_size[1] / frame.shape[0],
                        fps=fps,
                    )
                else:
                    display = _display_frame(frame_number, frame, result, fps=fps)
                cv2.imshow(window_name, display)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            display_ended = perf_counter()

            if performance_tracker is not None:
                performance_tracker.record(
                    FrameTiming(
                        frame_number=frame_number,
                        decode_ms=(decode_ended - decode_started) * 1000.0,
                        detection_ms=(detection_ended - detection_started) * 1000.0,
                        artifact_write_ms=(artifact_ended - artifact_started) * 1000.0,
                        display_ms=(display_ended - display_started) * 1000.0,
                        total_ms=(display_ended - loop_started) * 1000.0,
                    )
                )
                if performance_tracker.should_report(frame_number):
                    print(performance_tracker.format_summary())
            if quit_requested or (max_frames is not None and frame_number >= int(max_frames)):
                break
    finally:
        if capture is not None:
            capture.release()
        if sensor is not None:
            sensor.receiver.close()
        if annotated_video_writer is not None:
            annotated_video_writer.close()
        writer.close()
        cv2.destroyAllWindows()

    print(f"saved_results={writer.run_dir if writer.run_dir is not None else 'disabled'}")
    if annotated_video_writer is not None:
        print(f"saved_annotated_video={annotated_video_writer.path}")
    if performance_tracker is not None:
        performance_summary = performance_tracker.summary()
        print(performance_tracker.format_summary())
        performance_path = writer.save_performance_summary(performance_summary)
        if performance_path is not None:
            print(f"saved_performance_summary={performance_path}")


if __name__ == "__main__":
    main()
