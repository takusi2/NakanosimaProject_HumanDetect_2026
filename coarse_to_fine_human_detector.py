"""ロボットから呼び出す粗探索・詳細照合マネキン検出器。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from ClsImageViewerUDP import ClsImageViewerUDP
from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher
from experiments.coarse_to_fine_mannequin.src.tracking import TemporalMatchTracker


@dataclass(frozen=True)
class _DisplayOverlay:
    """表示画像へ最後に重ねる、解像度に依存しないHUD情報。"""

    status: str
    score_text: str
    fps: float
    colour: tuple[int, int, int]


class CoarseToFineHumanDetector:
    """CoarseToFineMatcher を HumanDetector 互換のAPIで提供する。"""

    def __init__(self, config_path: str = "config/coarse_to_fine_detector.yaml") -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")

        # 従来のHumanDetectorと同じ名前・既定値で、位置の時系列確認を設定する。
        config.setdefault("target_confirm_window_frames", 10)
        config.setdefault("target_confirm_frames", 5)
        config.setdefault("match_iou_thresh", 0.8)
        config.setdefault("max_track_age", 30)
        config.setdefault("maybe_target_color", [0, 165, 255])
        config.setdefault("confirmed_target_color", [0, 255, 0])
        config.setdefault("no_match_color", [0, 0, 255])
        for name in ("maybe_target_color", "confirmed_target_color", "no_match_color"):
            config[name] = tuple(config[name])

        template_path = self._resolve_path(str(config["template_path"]))
        template = cv2.imdecode(
            np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if template is None:
            raise FileNotFoundError(f"cannot read template image: {template_path}")

        # main でも同じ解決済みパスを使えるよう、設定値を置き換える。
        config["template_path"] = str(template_path)
        if config.get("video_path"):
            config["video_path"] = str(self._resolve_path(str(config["video_path"])))
        self.conf = SimpleNamespace(**config)
        self.matcher = CoarseToFineMatcher(template, **self._matcher_kwargs(config))
        self.frame_idx = 0
        self.tracker = TemporalMatchTracker(
            window_frames=int(config["target_confirm_window_frames"]),
            confirm_frames=int(config["target_confirm_frames"]),
            match_iou_thresh=float(config["match_iou_thresh"]),
            max_track_age=int(config["max_track_age"]),
        )
        self._display_base: np.ndarray | None = None
        self._display_overlay: _DisplayOverlay | None = None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.config_path.parent / path).resolve()

    @staticmethod
    def _matcher_kwargs(config: dict) -> dict:
        def channel_values(name: str, defaults: tuple[float, float, float]) -> tuple[float, float, float]:
            values = config.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be a mapping")
            return tuple(
                float(values.get(key, default))
                for key, default in zip(("rg", "by", "y"), defaults)
            )

        brightness = config.get("brightness_weights", {})
        if not isinstance(brightness, dict):
            raise ValueError("brightness_weights must be a mapping")
        return {
            "device": str(config.get("device", "cuda")),
            "frame_downscale": float(config.get("frame_downscale", 0.5)),
            "coarse_template_scales": config.get("coarse_template_scales", [0.3, 0.4, 0.5]),
            "detail_template_scales": config.get("detail_template_scales", [1.0, 0.9, 0.8, 0.7, 0.6]),
            "coarse_stride": int(config.get("coarse_stride", 2)),
            "detail_stride_base": int(config.get("detail_stride_base", 4)),
            "detail_stride_min": int(config.get("detail_stride_min", 2)),
            "coarse_top_k": int(config.get("coarse_top_k", 3)),
            "coarse_candidates_per_template": int(config.get("coarse_candidates_per_template", 20)),
            "nms_distance_original_px": int(config.get("nms_distance_original_px", 40)),
            "roi_margin_px": int(config.get("roi_margin_px", 5)),
            "final_max_error": float(config.get("final_max_error", 25.0)),
            "brightness_weights": tuple(
                float(brightness.get(key, default))
                for key, default in (("r", 0.299), ("g", 0.587), ("b", 0.114))
            ),
            "channel_weights": channel_values("channel_weights", (0.5, 0.5, 1.0)),
            "variance_channel_weights": channel_values(
                "variance_channel_weights", (1.0, 1.0, 1.0)
            ),
            "variance_log_distance_max": float(config.get("variance_log_distance_max", 5.0)),
            "variance_epsilon": float(config.get("variance_epsilon", 1.0)),
            "min_chroma_variance": float(config.get("min_chroma_variance", 50.0)),
            "min_brightness_variance": float(config.get("min_brightness_variance", 1000.0)),
            "min_weight": float(config.get("min_weight", 0.05)),
        }

    @property
    def track_info(self):
        """従来のHumanDetectorと同様に、現在の追跡情報を参照できるようにする。"""
        return self.tracker.track_info

    @staticmethod
    def _draw_display_overlay(image: np.ndarray, overlay: _DisplayOverlay) -> None:
        """HUDを表示画像へ固定ピクセルサイズで描く。"""
        cv2.putText(
            image,
            f"Frame {overlay.status}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            overlay.colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            overlay.score_text,
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            overlay.colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"FPS: {overlay.fps:.1f}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            overlay.colour,
            2,
            cv2.LINE_AA,
        )

    def get_display_frame(self, display_width: int, display_height: int) -> np.ndarray:
        """検出枠を拡大し、HUDだけを固定フォントで描いた表示画像を返す。"""
        if self._display_base is None or self._display_overlay is None:
            raise RuntimeError("process_frame must be called before get_display_frame")
        interpolation = (
            cv2.INTER_LINEAR
            if display_width >= self._display_base.shape[1]
            else cv2.INTER_AREA
        )
        display = cv2.resize(
            self._display_base, (display_width, display_height), interpolation=interpolation
        )
        self._draw_display_overlay(display, self._display_overlay)
        return display

    def process_frame(
        self, frame: np.ndarray, source_timestamp_s: float | None = None
    ) -> tuple[np.ndarray, float | None, str | None]:
        """描画フレーム、MATCH時の中心x、互換クラス名 ``A`` を返す。"""
        del source_timestamp_s  # 互換性のため受け取るが、軽量推論では保存しない。
        self.frame_idx += 1
        started = perf_counter()
        result = self.matcher.process(
            frame,
            collect_coarse_image=False,
            collect_detail_variance_filters=False,
        )
        elapsed = perf_counter() - started
        output = frame.copy()
        best = result.best_match
        raw_match = result.detected and best is not None
        bbox_xyxy = (
            (best.x, best.y, best.x + best.template.width, best.y + best.template.height)
            if raw_match and best is not None
            else None
        )
        decision = self.tracker.update(bbox_xyxy)
        confirmed = raw_match and decision.confirmed
        if confirmed:
            status = "MATCH"
            colour = self.conf.confirmed_target_color
        elif raw_match:
            status = "MAYBE"
            colour = self.conf.maybe_target_color
        else:
            status = "NO MATCH"
            colour = self.conf.no_match_color

        if best is not None:
            cv2.rectangle(
                output,
                (best.x, best.y),
                (best.x + best.template.width, best.y + best.template.height),
                colour,
                2,
            )
            score_text = f"score={best.score:.2f} scale={best.template.scale:.2f}"
            if decision.track_id is not None:
                score_text += (
                    f" track={decision.track_id} history="
                    f"{decision.positive_frames}/{decision.history_window_frames}"
                )
        else:
            score_text = "score=N/A scale=N/A"

        fps = 1.0 / max(elapsed, 1e-6)
        overlay = _DisplayOverlay(
            status=f"{self.frame_idx}: {status}",
            score_text=score_text,
            fps=fps,
            colour=colour,
        )
        # 戻り値は従来どおり元解像度にHUDを描く。mainの拡大表示用にはHUDなしの画像も残す。
        self._display_base = output.copy()
        self._display_overlay = overlay
        self._draw_display_overlay(output, overlay)

        center_x = (
            float(best.x + best.template.width / 2.0)
            if confirmed and best is not None
            else None
        )
        return output, center_x, "A" if center_x is not None else None

    def close(self) -> None:
        """将来の推論リソース解放用の互換フック。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="粗探索・詳細照合マネキン検出")
    parser.add_argument("--config", default="config/coarse_to_fine_detector.yaml")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="処理する最大フレーム数。動作確認時に使う。未指定なら入力終了またはqまで処理する。",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="OpenCVウィンドウを表示しない。動画入力のヘッドレス動作確認に使う。",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    detector = CoarseToFineHumanDetector(args.config)
    input_source = str(detector.conf.input_source)
    display_width = getattr(detector.conf, "display_width", None)
    display_height = getattr(detector.conf, "display_height", None)
    if (display_width is None) != (display_height is None):
        raise ValueError("display_width and display_height must be specified together")
    if display_width is not None:
        display_width = int(display_width)
        display_height = int(display_height)
        if display_width <= 0 or display_height <= 0:
            raise ValueError("display_width and display_height must be positive")

    capture: cv2.VideoCapture | None = None
    sensor: ClsImageViewerUDP | None = None

    if input_source == "udp":
        sensor = ClsImageViewerUDP()

        def get_frame() -> tuple[bool, np.ndarray | None]:
            assert sensor is not None
            sensor.receive_one_set()
            frame = sensor.get_center_surround_bgr_image()
            return (frame is not None, frame)

    elif input_source in ("camera", "video"):
        source = int(detector.conf.camera_index) if input_source == "camera" else detector.conf.video_path
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {input_source} input: {source}")

        def get_frame() -> tuple[bool, np.ndarray | None]:
            assert capture is not None
            return capture.read()

    else:
        raise ValueError(f"unknown input_source: {input_source!r}")

    frame_number = 0
    try:
        while True:
            ok, frame = get_frame()
            if not ok or frame is None:
                break
            frame = cv2.resize(
                frame, (int(detector.conf.frame_width), int(detector.conf.frame_height))
            )
            timestamp = (
                capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if input_source == "video" and capture is not None
                else None
            )
            output, center_x, class_name = detector.process_frame(frame, timestamp)
            frame_number += 1
            print(f"center_x={center_x}, class_name={class_name}")
            if not args.no_display:
                # 表示サイズだけを変える。process_frameへ渡したフレームと検出結果は変えない。
                if display_width is None:
                    display = output
                else:
                    display = detector.get_display_frame(display_width, display_height)
                cv2.imshow(str(detector.conf.window_name), display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames is not None and frame_number >= args.max_frames:
                break
    finally:
        detector.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
