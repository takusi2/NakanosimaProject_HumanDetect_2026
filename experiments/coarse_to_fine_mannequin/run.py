"""粗探索・詳細照合マネキン検出器を、実カメラ入力で検証する。"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml

from sensor_io.frame_sources import (
    CaptureFrameSource,
    SensorFrameSource,
)
from src.artifacts import (
    AnnotatedVideoWriter,
    ArtifactWriter,
)
from src.performance import (
    FrameTiming,
    PerformanceTracker,
)
from src.display import draw_display_frame
from src.pipeline import CoarseToFineMatcher
from src.tracking import (
    TemporalMatchTracker,
    apply_temporal_tracking,
)
from src.settings import RunSettings


# 使用する設定ファイルはここだけで指定する。
config_path_name = "run_20260902_1552_right_motor.yaml"
CONFIG_PATH = Path(__file__).parent / "config" / config_path_name


def main() -> None:
    settings = RunSettings(CONFIG_PATH)
    template = cv2.imdecode(
        np.fromfile(str(settings.input.template_path), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if template is None:
        raise FileNotFoundError(f"cannot read template image: {settings.input.template_path}")

    # 検出器・追跡器・保存器・パフォーマンス計測器の初期化
    save_options = settings.save
    performance_options = settings.performance
    collect_coarse_image = (
        save_options.enabled
        and (save_options.coarse_frame or save_options.coarse_candidates)
    )
    collect_detail_variance_filters = (
        save_options.enabled
        and (save_options.variance_filters or save_options.scores_json)
    )
    matcher = CoarseToFineMatcher(template, settings.matcher)
    tracker = TemporalMatchTracker(settings.tracking)
    writer = ArtifactWriter(settings.input.results_root, save_options)
    writer.save_detail_templates(matcher.detail_templates)
    performance_tracker = (
        PerformanceTracker(performance_options) if performance_options.enabled else None
    )

    # 入力元の初期化
    input_size = settings.input.input_size
    display_size = settings.display.display_size
    if settings.input.input_source == "udp":
        source = SensorFrameSource()
    elif settings.input.input_source == "camera":
        source = CaptureFrameSource(settings.input.camera_index)
    elif settings.input.input_source == "video":
        source = CaptureFrameSource(settings.input.video_path)
    else:
        raise ValueError(f"unknown input source type: {settings.input.input_source}")

    processing_frame_limit = settings.input.processing_frame_limit
    show_window = settings.display.show_window
    window_name = settings.display.window_name
    frame_number = 0
    annotated_video_writer: AnnotatedVideoWriter | None = None
    try:
        while True:
            loop_started = perf_counter()
            decode_started = perf_counter()
            frame = source.read()
            decode_ended = perf_counter()
            if frame is None:
                break
            frame = cv2.resize(frame, input_size)
            frame_number += 1

            detection_started = perf_counter()
            raw_result = matcher.process(
                frame,
                collect_coarse_image=collect_coarse_image,
                collect_detail_variance_filters=collect_detail_variance_filters,
            )
            result = apply_temporal_tracking(raw_result, tracker)
            detection_ended = perf_counter()
            fps = 1.0 / max(detection_ended - detection_started, 1e-6)

            artifact_started = perf_counter()
            if save_options.enabled and save_options.annotated_video:
                if annotated_video_writer is None:
                    annotated_video_writer = AnnotatedVideoWriter(
                        writer.require_run_dir(), source.get_fps(), frame.shape[1], frame.shape[0]
                    )
                annotated_video_writer.write(frame_number, frame, result)
            if save_options.should_save_frame(frame_number):
                writer.save_frame(frame_number, frame, result)
            artifact_ended = perf_counter()

            display_started = perf_counter()
            quit_requested = False
            if show_window:
                interpolation = cv2.INTER_LINEAR if display_size[0] >= frame.shape[1] else cv2.INTER_AREA
                display = cv2.resize(frame, display_size)
                display = draw_display_frame(
                    frame_number,
                    display,
                    result,
                    scale_x=display_size[0] / frame.shape[1],
                    scale_y=display_size[1] / frame.shape[0],
                    fps=fps,
                )
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
            if (
                quit_requested
                or (
                    processing_frame_limit is not None
                    and frame_number >= processing_frame_limit
                )
            ):
                break
    finally:
        source.close()
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
