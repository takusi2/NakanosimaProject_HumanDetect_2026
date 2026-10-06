"""YAML設定を読み込み、実行に必要な値だけを保持する設定クラス。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .artifacts import ArtifactSaveOptions
from .performance import PerformanceOptions
from .readonly import ReadOnlySettings


def _resolve_path(value: str, config_path: Path) -> Path:
    """YAMLに書いた相対パスを、YAMLファイルから見た絶対パスにする。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _read_three_weights(
    config: Mapping[str, object],
    setting_name: str,
    keys: tuple[str, str, str],
    defaults: tuple[float, float, float],
) -> tuple[float, float, float]:
    """3要素の重み設定を、YAMLの辞書から読む。"""
    raw_value = config.get(setting_name, {})
    if raw_value is None:
        raw_value = {}
    if not isinstance(raw_value, Mapping):
        raise ValueError(f"{setting_name} must be a YAML mapping")

    first = float(raw_value.get(keys[0], defaults[0]))
    second = float(raw_value.get(keys[1], defaults[1]))
    third = float(raw_value.get(keys[2], defaults[2]))
    return first, second, third


class MatcherSettings(ReadOnlySettings):
    """CoarseToFineMatcherが使う照合設定。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.device = str(config.get("device", "cuda"))
        self.frame_downscale = float(config.get("frame_downscale", 0.25))
        self.coarse_template_scales = tuple(
            float(value) for value in config.get("coarse_template_scales", [0.2])
        )
        self.detail_template_scales = tuple(
            float(value)
            for value in config.get("detail_template_scales", [1.0, 0.8, 0.6, 0.4])
        )
        self.coarse_stride = int(config.get("coarse_stride", 2))
        self.detail_stride_base = int(
            config.get("detail_stride_base", config.get("detail_stride", 4))
        )
        self.detail_stride_min = int(config.get("detail_stride_min", 2))
        self.coarse_top_k = int(config.get("coarse_top_k", 3))
        self.coarse_candidates_per_template = int(
            config.get("coarse_candidates_per_template", 20)
        )
        self.nms_distance_original_px = int(config.get("nms_distance_original_px", 120))
        self.roi_margin_px = int(config.get("roi_margin_px", 32))
        self.final_max_error = float(config.get("final_max_error", 27.0))
        self.brightness_weights = _read_three_weights(
            config, "brightness_weights", ("r", "g", "b"), (0.299, 0.587, 0.114)
        )
        self.channel_weights = _read_three_weights(
            config, "channel_weights", ("rg", "by", "y"), (0.5, 0.5, 1.0)
        )
        self.variance_channel_weights = _read_three_weights(
            config, "variance_channel_weights", ("rg", "by", "y"), (1.0, 1.0, 1.0)
        )
        self.variance_log_distance_max = float(config.get("variance_log_distance_max", 3.0))
        self.variance_epsilon = float(config.get("variance_epsilon", 1.0))
        self.min_chroma_variance = float(config.get("min_chroma_variance", 0.0))
        self.min_brightness_variance = float(config.get("min_brightness_variance", 0.0))
        self.min_weight = float(config.get("min_weight", 0.05))
        self._lock_settings()


class TrackingSettings(ReadOnlySettings):
    """TemporalMatchTrackerが使う追跡・確定設定。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.window_frames = int(config.get("target_confirm_window_frames", 10))
        self.confirm_frames = int(config.get("target_confirm_frames", 5))
        self.match_iou_thresh = float(config.get("match_iou_thresh", 0.8))
        self.max_track_age = int(config.get("max_track_age", 30))
        self._lock_settings()


class InputSettings(ReadOnlySettings):
    """参照画像・入力動画／カメラ・フレームサイズの設定。"""

    def __init__(self, config: Mapping[str, object], config_path: Path) -> None:
        if "template_path" not in config:
            raise ValueError("template_path must be specified")
        self.template_path = _resolve_path(str(config["template_path"]), config_path)
        self.input_source = str(config.get("input_source", "video"))
        if self.input_source not in ("udp", "camera", "video"):
            raise ValueError("input_source must be 'udp', 'camera', or 'video'")

        self.video_path: Path | None = None
        if self.input_source == "video":
            if "video_path" not in config:
                raise ValueError("video_path must be specified when input_source is 'video'")
            self.video_path = _resolve_path(str(config["video_path"]), config_path)

        self.camera_index = int(config.get("camera_index", 0))
        width = config.get("input_width")
        height = config.get("input_height")
        if (width is None) or (height is None):
            raise ValueError("input_width and input_height must both be specified")
        self.input_size = (int(width), int(height))
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError("input_width and input_height must be positive")

        self.results_root = _resolve_path(
            str(config.get("results_root", "../results")), config_path
        )
        raw_processing_frame_limit = config.get("processing_frame_limit")
        self.processing_frame_limit = (
            None
            if raw_processing_frame_limit is None
            else int(raw_processing_frame_limit)
        )
        if (
            self.processing_frame_limit is not None
            and self.processing_frame_limit <= 0
        ):
            raise ValueError("processing_frame_limit must be positive when specified")
        self._lock_settings()


class DisplaySettings(ReadOnlySettings):
    """OpenCV表示ウィンドウの設定。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.show_window = bool(config.get("show_window", True))
        self.window_name = str(config.get("window_name", "Coarse-to-fine mannequin matcher"))
        width = config.get("display_width")
        height = config.get("display_height")
        if (width is None) or (height is None):
            raise ValueError("display_width and display_height must both be specified")
        self.display_size = (int(width), int(height))
        if self.display_size[0] <= 0 or self.display_size[1] <= 0:
            raise ValueError("display_width and display_height must be positive")
        self._lock_settings()


class RunSettings(ReadOnlySettings):
    """設定ファイルを一度だけ読み、各処理用設定をまとめる窓口。"""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, Mapping):
            raise ValueError("config must be a YAML mapping")

        self.matcher = MatcherSettings(raw_config)
        self.tracking = TrackingSettings(raw_config)
        self.input = InputSettings(raw_config, self.config_path)
        self.display = DisplaySettings(raw_config)
        self.save = ArtifactSaveOptions(raw_config)
        self.performance = PerformanceOptions(raw_config)
        self._lock_settings()
