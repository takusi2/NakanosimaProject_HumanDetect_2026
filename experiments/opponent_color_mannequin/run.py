"""二十反対色テンプレート照合を動画またはカメラ入力で実行する。"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))

from experiments.opponent_color_mannequin.src.matcher import OpponentColorMatcher  # noqa: E402

# 実行時に使う設定は、分かりやすさのためこのファイルに直接書く。
TEMPLATE_PATH = Path(
    r"C:\Users\takus\OneDrive\ドキュメント\ChatGPT\NakanosimaProject_HumanDetect_2026\humanA\masa_gc1_gs6.jpg"
)
INPUT_SOURCE = "video"  # "video" または "camera"
VIDEO_PATH = Path(
    r"C:\Users\takus\OneDrive\ドキュメント\ChatGPT\NakanosimaProject_HumanDetect_2026\output\2026-0826-1557-39_center_surround.mp4"
)
CAMERA_INDEX = 0

STRIDE_X = 8
STRIDE_Y = 8
MAX_ERROR = 40.0
BRIGHTNESS_WEIGHTS = (0.299, 0.587, 0.114)
CHANNEL_WEIGHTS = (1.0, 1.0, 1.0)
WEIGHT_MODE = "center_falloff"
MIN_WEIGHT = 0.05
MARGINS = (0.15, 0.15, 0.15, 0.15)
DEVICE = "cuda"


def main() -> None:
    print(f"template_path={TEMPLATE_PATH}")
    # cv2.imread は Windows で日本語を含むパスを読めないことがあるため、
    # ファイルを Python 側で読み、OpenCV で画像として復号する。
    template_bytes = np.fromfile(str(TEMPLATE_PATH), dtype=np.uint8)
    template = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"template image cannot be read: {TEMPLATE_PATH}")

    matcher = OpponentColorMatcher(
        template,
        max_error=MAX_ERROR,
        brightness_weights=BRIGHTNESS_WEIGHTS,
        channel_weights=CHANNEL_WEIGHTS,
        weight_mode=WEIGHT_MODE,
        min_weight=MIN_WEIGHT,
        margins=MARGINS,
        device=DEVICE,
    )

    if INPUT_SOURCE == "camera":
        capture = cv2.VideoCapture(CAMERA_INDEX)
    elif INPUT_SOURCE == "video":
        capture = cv2.VideoCapture(str(VIDEO_PATH))
    else:
        raise ValueError("input_source must be 'video' or 'camera'")
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {INPUT_SOURCE} input")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = matcher.match(
                frame,
                stride_x=STRIDE_X,
                stride_y=STRIDE_Y,
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
