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


def main() -> None:
    parser = argparse.ArgumentParser(description="二十反対色テンプレート照合")
    parser.add_argument("--config", type=Path, required=True, help="YAML 設定ファイル")
    args = parser.parse_args()
    config_file = args.config.resolve()
    config = _load_config(config_file)

    template_path = _config_path(str(config["template_path"]), config_file)
    # cv2.imread は Windows で日本語を含むパスを読めないことがあるため、
    # ファイルを Python 側で読み、OpenCV で画像として復号する。
    template_bytes = np.fromfile(str(template_path), dtype=np.uint8)
    template = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"template image cannot be read: {template_path}")

    matcher = OpponentColorMatcher(
        template,
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
            result = matcher.match(
                frame,
                stride_x=config.get("stride_x"),
                stride_y=config.get("stride_y"),
            )
            x, y = result.top_left
            colour = (0, 255, 0) if result.detected else (0, 0, 255)
            cv2.rectangle(
                frame,
                (x, y),
                (x + matcher.template_width, y + matcher.template_height),
                colour,
                2,
            )
            label = f"{'MANNEQUIN' if result.detected else 'NO MATCH'} score={result.score:.2f}"
            cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
            cv2.imshow("Opponent-color mannequin matcher", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
