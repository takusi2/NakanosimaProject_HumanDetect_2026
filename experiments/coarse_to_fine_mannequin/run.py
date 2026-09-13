"""粗探索・詳細照合の二段階テンプレート照合を実行する。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from experiments.coarse_to_fine_mannequin.src.artifacts import (  # noqa: E402
    AnnotatedVideoWriter,
    ArtifactWriter,
)
from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher  # noqa: E402


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
        min_weight=float(config.get("min_weight", 0.05)),
    )

    results_root = _resolve_path(str(config.get("results_root", "../results")), config_path)
    writer = ArtifactWriter(results_root)
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

    save_every_n_frames = int(config.get("save_every_n_frames", 1))
    save_annotated_video = bool(config.get("save_annotated_video", True))
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
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            result = matcher.process(frame)

            if save_annotated_video:
                if annotated_video_writer is None:
                    annotated_video_writer = AnnotatedVideoWriter(
                        writer.run_dir,
                        capture.get(cv2.CAP_PROP_FPS),
                        frame.shape[1],
                        frame.shape[0],
                    )
                annotated_video_writer.write(frame_number, frame, result)

            if frame_number % save_every_n_frames == 0:
                writer.save_frame(frame_number, frame, result)
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
                    break
            if max_frames is not None and frame_number >= int(max_frames):
                break
    finally:
        capture.release()
        if annotated_video_writer is not None:
            annotated_video_writer.close()
        writer.close()
        cv2.destroyAllWindows()

    print(f"saved_results={writer.run_dir}")
    if annotated_video_writer is not None:
        print(f"saved_annotated_video={annotated_video_writer.path}")


if __name__ == "__main__":
    main()
