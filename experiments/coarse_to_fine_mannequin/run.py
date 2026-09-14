"""粗探索・詳細照合の二段階テンプレート照合を実行する。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from experiments.coarse_to_fine_mannequin.src.artifacts import (  # noqa: E402
    AnnotatedVideoWriter,
    ArtifactSaveOptions,
    ArtifactWriter,
)
from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher  # noqa: E402
from experiments.coarse_to_fine_mannequin.src.performance import (  # noqa: E402
    FrameTiming,
    PerformanceOptions,
    PerformanceTracker,
)


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="粗探索・詳細照合マネキン位置推定")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    template_path = _resolve_path(str(config["template_path"]), config_path)
    template = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"cannot read template image: {template_path}")

    matcher = CoarseToFineMatcher(
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
        brightness_weights=tuple(config.get("brightness_weights", {}).get(k, d) for k, d in (("r", 0.299), ("g", 0.587), ("b", 0.114))),
        channel_weights=tuple(config.get("channel_weights", {}).get(k, d) for k, d in (("rg", 0.5), ("by", 0.5), ("y", 1.0))),
        variance_channel_weights=tuple(config.get("variance_channel_weights", {}).get(k, d) for k, d in (("rg", 1.0), ("by", 1.0), ("y", 1.0))),
        variance_log_distance_max=float(config.get("variance_log_distance_max", 3.0)),
        variance_epsilon=float(config.get("variance_epsilon", 1.0)),
        min_chroma_variance=float(config.get("min_chroma_variance", 0.0)),
        min_brightness_variance=float(config.get("min_brightness_variance", 0.0)),
        min_weight=float(config.get("min_weight", 0.05)),
    )

    results_root = _resolve_path(str(config.get("results_root", "../results")), config_path)
    save_options = ArtifactSaveOptions.from_config(config)
    performance_options = PerformanceOptions.from_config(config)
    performance_tracker = (
        PerformanceTracker(performance_options) if performance_options.enabled else None
    )
    writer = ArtifactWriter(results_root, save_options)
    writer.save_detail_templates(matcher.detail_templates)

    source = str(config.get("input_source", "video"))
    if source == "camera":
        capture = cv2.VideoCapture(int(config.get("camera_index", 0)))
    elif source == "video":
        capture = cv2.VideoCapture(str(_resolve_path(str(config["video_path"]), config_path)))
    else:
        raise ValueError("input_source must be 'video' or 'camera'")
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source} input")

    max_frames = config.get("max_frames")
    show_window = bool(config.get("show_window", True))
    display_width = config.get("display_width")
    display_height = config.get("display_height")
    if (display_width is None) != (display_height is None):
        raise ValueError("display_width and display_height must be specified together")
    if display_width is not None:
        display_width = int(display_width)
        display_height = int(display_height)
        if display_width <= 0 or display_height <= 0:
            raise ValueError("display_width and display_height must be positive")
    frame_number = 0
    annotated_video_writer: AnnotatedVideoWriter | None = None
    try:
        while True:
            loop_started = perf_counter()
            decode_started = perf_counter()
            ok, frame = capture.read()
            decode_ended = perf_counter()
            if not ok:
                break
            frame_number += 1
            # 保存しない中間データはGPU上の検出判定にだけ使用し、CPUへは転送しない。
            detection_started = perf_counter()
            result = matcher.process(
                frame,
                collect_coarse_image=(
                    save_options.enabled
                    and (save_options.coarse_frame or save_options.coarse_candidates)
                ),
                collect_detail_variance_filters=(
                    save_options.enabled
                    and (save_options.variance_filters or save_options.scores_json)
                ),
            )
            detection_ended = perf_counter()

            artifact_started = perf_counter()
            if save_options.enabled and save_options.annotated_video:
                if annotated_video_writer is None:
                    annotated_video_writer = AnnotatedVideoWriter(
                        writer.require_run_dir(),
                        capture.get(cv2.CAP_PROP_FPS),
                        frame.shape[1],
                        frame.shape[0],
                    )
                annotated_video_writer.write(frame_number, frame, result)

            if save_options.should_save_frame(frame_number):
                writer.save_frame(frame_number, frame, result)
            artifact_ended = perf_counter()

            display_started = perf_counter()
            quit_requested = False
            if show_window:
                # 検出は上の matcher.process(frame) で元フレームのまま完了している。
                # ここで作る display は表示専用であり、判定値や保存元フレームを変えない。
                if display_width is None:
                    display = frame.copy()
                    scale_x = 1.0
                    scale_y = 1.0
                else:
                    interpolation = cv2.INTER_LINEAR if display_width >= frame.shape[1] else cv2.INTER_AREA
                    display = cv2.resize(frame, (display_width, display_height), interpolation=interpolation)
                    scale_x = display_width / frame.shape[1]
                    scale_y = display_height / frame.shape[0]
                best = result.best_match
                colour = (0, 255, 0) if result.detected else (0, 0, 255)
                if best is not None:
                    cv2.rectangle(
                        display,
                        (round(best.x * scale_x), round(best.y * scale_y)),
                        (
                            round((best.x + best.template.width) * scale_x),
                            round((best.y + best.template.height) * scale_y),
                        ),
                        colour,
                        max(1, round(2 * min(scale_x, scale_y))),
                    )
                score_text = (
                    f"score={best.score:.2f} var={best.variance_distance:.2f}"
                    if best is not None
                    else "all detail candidates rejected by variance"
                )
                cv2.putText(
                    display,
                    f"{'MANNEQUIN' if result.detected else 'NO MATCH'} {score_text}",
                    # スコア文字は映像倍率に連動させず、表示領域を占有しない。
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Coarse-to-fine mannequin matcher", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_requested = True
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
            if quit_requested:
                break
            if max_frames is not None and frame_number >= int(max_frames):
                break
    finally:
        capture.release()
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
