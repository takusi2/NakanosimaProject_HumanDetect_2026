"""二十反対色テンプレート照合を動画またはカメラ入力で実行する。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from experiments.opponent_color_mannequin.src.matcher import OpponentColorMatcher  # noqa: E402


def _config_path(value: str, config_file: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_file.parent / path).resolve()


def _load_config(config_file: Path) -> dict:
    with config_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    return config


def _create_multiscale_matchers(template: np.ndarray, config: dict) -> list[tuple[float, OpponentColorMatcher]]:
    """元画像と縮小画像ごとに、GPUへ保持する照合器を作る。"""
    matchers: list[tuple[float, OpponentColorMatcher]] = []
    used_sizes: set[tuple[int, int]] = set()
    for scale in config.get("template_scales", [1.0]):
        scale = float(scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError("template_scales must be in the range (0, 1]")

        width = max(1, round(template.shape[1] * scale))
        height = max(1, round(template.shape[0] * scale))
        if (width, height) in used_sizes:
            continue
        used_sizes.add((width, height))
        resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        matcher = OpponentColorMatcher(
            resized_template,
            max_error=float(config["max_error"]),
            brightness_weights=tuple(config.get("brightness_weights", {}).get(k, d) for k, d in (
                ("r", 0.2126), ("g", 0.7152), ("b", 0.0722)
            )),
            channel_weights=tuple(config.get("channel_weights", {}).get(k, 1.0) for k in ("rg", "by", "y")),
            weight_mode=config.get("weight_mode", "inner_rectangle"),
            min_weight=float(config.get("min_weight", 0.05)),
            margins=tuple(float(config.get(f"margin_{side}", 0.15)) for side in ("top", "bottom", "left", "right")),
            device=str(config.get("device", "cuda")),
        )
        matchers.append((scale, matcher))
    return matchers


def main() -> None:
    parser = argparse.ArgumentParser(description="二十反対色テンプレート照合")
    parser.add_argument("--config", type=Path, required=True, help="YAML 設定ファイル")
    args = parser.parse_args()
    config_file = args.config.resolve()
    config = _load_config(config_file)
    display_width = int(config.get("display_width", 1280))
    display_height = int(config.get("display_height", 960))
    if display_width <= 0 or display_height <= 0:
        raise ValueError("display_width and display_height must be positive")

    template_path = _config_path(str(config["template_path"]), config_file)
    print(f"template_path={template_path}")
    # cv2.imread は Windows で日本語を含むパスを読めないことがあるため、
    # ファイルを Python 側で読み、OpenCV で画像として復号する。
    template_bytes = np.fromfile(str(template_path), dtype=np.uint8)
    template = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"template image cannot be read: {template_path}")

    matchers = _create_multiscale_matchers(template, config)
    print(f"template_scales={[scale for scale, _ in matchers]}")

    source = str(config.get("input_source", "video"))
    if source == "camera":
        capture = cv2.VideoCapture(int(config.get("camera_index", 0)))
    elif source == "video":
        video_path = _config_path(str(config["video_path"]), config_file)
        capture = cv2.VideoCapture(str(video_path))
    else:
        raise ValueError("input_source must be 'video' or 'camera'")
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source} input")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # 各縮小率でGPU照合し、最小スコアのテンプレートサイズを採用する。
            scale_results = []
            for template_scale, matcher in matchers:
                if matcher.template_height > frame.shape[0] or matcher.template_width > frame.shape[1]:
                    continue
                base_stride_x = config.get("stride_x")
                base_stride_y = config.get("stride_y")
                stride_x = None if base_stride_x is None else max(1, round(int(base_stride_x) * template_scale))
                stride_y = None if base_stride_y is None else max(1, round(int(base_stride_y) * template_scale))
                result = matcher.match(frame, stride_x=stride_x, stride_y=stride_y)
                scale_results.append((template_scale, matcher, result))
            if not scale_results:
                raise ValueError("all scaled templates are larger than the input frame")

            template_scale, matcher, result = min(scale_results, key=lambda item: item[2].score)

            # frame は元解像度のまま照合済み。ここからは表示専用の画像を作る。
            interpolation = cv2.INTER_LINEAR if display_width >= frame.shape[1] else cv2.INTER_AREA
            display = cv2.resize(frame, (display_width, display_height), interpolation=interpolation)
            scale_x = display_width / frame.shape[1]
            scale_y = display_height / frame.shape[0]
            x, y = result.top_left
            colour = (0, 255, 0) if result.detected else (0, 0, 255)
            cv2.rectangle(
                display,
                (round(x * scale_x), round(y * scale_y)),
                (
                    round((x + matcher.template_width) * scale_x),
                    round((y + matcher.template_height) * scale_y),
                ),
                colour,
                max(1, round(2 * min(scale_x, scale_y))),
            )
            label = (
                f"{'MANNEQUIN' if result.detected else 'NO MATCH'} "
                f"score={result.score:.2f} scale={template_scale:.1f}"
            )
            cv2.putText(
                display,
                label,
                # スコア表示は表示倍率に連動させない。拡大表示でも領域を占有しない。
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colour,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Opponent-color mannequin matcher", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
