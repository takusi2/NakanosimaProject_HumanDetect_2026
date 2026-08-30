"""正面直射で全色が明るく・鮮やかになるA-1用サンプル3を作成する。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


SAMPLE_ROOT = Path(__file__).resolve().parent / "data" / "sensor_quality_filename_samples_3"
SOURCE_IMAGE = (
    Path(__file__).resolve().parent
    / "data"
    / "sensor_quality_filename_samples"
    / "source_color_chart.png"
)
IMAGE_DIRECTORY = SAMPLE_ROOT / "images"
VIDEO_DIRECTORY = SAMPLE_ROOT / "videos"

# 直射時に明るさを増やす余地を残すため、日陰側は最大でも0.80にする。
CENTER_BRIGHTNESS = {"04": 0.50, "06": 0.60, "08": 0.70, "10": 0.80}
LIGHT_CONDITIONS = ("shade_none", "sun_back", "sun_front")


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return image


def adjust_hsv(image: np.ndarray, value_scale: float, saturation_scale: float) -> np.ndarray:
    """HSVの明度Vと彩度Sを独立に調整する。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * value_scale, 0, 255)
    return cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_conditions(source: np.ndarray, center_factor: float, light_condition: str) -> np.ndarray:
    """gCと照明条件を合成する。

    日陰は彩度を少し抑え、正面直射は明度・彩度をともに上げる。
    そのため、正面直射では赤・緑・青のradiusが増える設計になっている。
    """
    if light_condition == "shade_none":
        return adjust_hsv(source, center_factor, 0.72)
    if light_condition == "sun_back":
        return adjust_hsv(source, center_factor * 0.80, 0.68)
    if light_condition == "sun_front":
        return adjust_hsv(source, center_factor * 1.20, 0.72 * 1.30)
    raise ValueError(f"未対応の照明条件です: {light_condition}")


def sample_stem(center: str, light_condition: str) -> str:
    return (
        "20260827_1400_sample3_nowide_"
        f"gC{center}_gS06_iir040_nrA_{light_condition}_d1p0m_front_t01"
    )


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNGを作成できませんでした: {path.name}")
    path.write_bytes(encoded.tobytes())


def write_one_frame_video(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    with tempfile.TemporaryDirectory(prefix="sensor_quality_sample3_") as directory:
        temporary_path = Path(directory) / path.name
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"MP4を作成できませんでした: {path.name}")
        writer.write(image)
        writer.release()
        capture = cv2.VideoCapture(str(temporary_path))
        valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
        capture.release()
        if not valid:
            raise RuntimeError(f"作成したMP4を確認できませんでした: {path.name}")
        path.write_bytes(temporary_path.read_bytes())


def main() -> int:
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError("元画像がありません。先にサンプル1を作成してください。")
    source = read_image(SOURCE_IMAGE)
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for center, factor in CENTER_BRIGHTNESS.items():
        for light_condition in LIGHT_CONDITIONS:
            image = apply_conditions(source, factor, light_condition)
            stem = sample_stem(center, light_condition)
            write_png(IMAGE_DIRECTORY / f"{stem}.png", image)
            write_one_frame_video(VIDEO_DIRECTORY / f"{stem}.mp4", image)
    print("作成完了: PNG 12枚、MP4 12本")
    print("正面直射: HSVの明度・彩度をともに上げる")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
