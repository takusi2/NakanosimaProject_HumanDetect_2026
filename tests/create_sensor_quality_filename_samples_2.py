"""中心ガウシアン値に応じて段階的に明るくなるA-1用サンプル2を作成する。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


SAMPLE_ROOT = Path(__file__).resolve().parent / "data" / "sensor_quality_filename_samples_2"
SOURCE_IMAGE = (
    Path(__file__).resolve().parent
    / "data"
    / "sensor_quality_filename_samples"
    / "source_color_chart.png"
)
IMAGE_DIRECTORY = SAMPLE_ROOT / "images"
VIDEO_DIRECTORY = SAMPLE_ROOT / "videos"

# gS は 06 に固定し、gC の増加とともに画像全体を明るくする。
CENTER_BRIGHTNESS = {"04": 0.55, "06": 0.70, "08": 0.85, "10": 1.00}
LIGHT_CONDITIONS = ("shade_none", "sun_back", "sun_front")


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return image


def brightened_image(image: np.ndarray, factor: float) -> np.ndarray:
    """色相を保ったまま、各BGR成分を同じ比率で明るくする。"""
    return np.rint(image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def apply_light_condition(image: np.ndarray, light_condition: str) -> np.ndarray:
    """日陰を基準に、逆光・正面直射を区別する合成照明を加える。

    - shade_none: 基準画像のまま。
    - sun_back: 被写体面が少し暗くなり、わずかに青寄りになる逆光を模擬する。
    - sun_front: 明るさ増加、白寄り（彩度低下）、わずかな暖色寄りを模擬する。
    """
    floating = image.astype(np.float32)
    if light_condition == "shade_none":
        result = floating
    elif light_condition == "sun_back":
        # BGR順。暗くしたうえで青成分を少し加え、影・空光の影響を表現する。
        result = floating * 0.82 + np.array([5.0, 1.0, 0.0], dtype=np.float32)
    elif light_condition == "sun_front":
        # 直射では明るくなり、白へ少し近付くことで色の抜け・白飛びを模擬する。
        bright = floating * 1.10 + np.array([0.0, 3.0, 12.0], dtype=np.float32)
        result = bright * 0.88 + 255.0 * 0.12
    else:
        raise ValueError(f"未対応の照明条件です: {light_condition}")
    return np.rint(result).clip(0, 255).astype(np.uint8)


def sample_stem(center: str, light_condition: str) -> str:
    return (
        "20260827_1300_sample2_nowide_"
        f"gC{center}_gS06_iir040_nrA_{light_condition}_d1p0m_front_t01"
    )


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNGを作成できませんでした: {path.name}")
    path.write_bytes(encoded.tobytes())


def write_one_frame_video(path: Path, image: np.ndarray) -> None:
    """画像を1秒・1 fps・1フレームのMP4として保存する。"""
    height, width = image.shape[:2]
    with tempfile.TemporaryDirectory(prefix="sensor_quality_sample2_") as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"MP4を作成できませんでした: {path.name}")
        writer.write(image)
        writer.release()
        capture = cv2.VideoCapture(str(temporary_path))
        valid = (
            capture.isOpened()
            and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
            and abs(float(capture.get(cv2.CAP_PROP_FPS)) - 1.0) < 1e-6
        )
        capture.release()
        if not valid:
            raise RuntimeError(f"作成したMP4を1 fps・1フレームとして確認できませんでした: {path.name}")
        path.write_bytes(temporary_path.read_bytes())


def main() -> int:
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(
            "元画像がありません。先に create_sensor_quality_filename_samples.py を実行してください。"
        )
    source = read_image(SOURCE_IMAGE)
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)

    created = 0
    for center, factor in CENTER_BRIGHTNESS.items():
        base_image = brightened_image(source, factor)
        for light_condition in LIGHT_CONDITIONS:
            image = apply_light_condition(base_image, light_condition)
            stem = sample_stem(center, light_condition)
            write_png(IMAGE_DIRECTORY / f"{stem}.png", image)
            write_one_frame_video(VIDEO_DIRECTORY / f"{stem}.mp4", image)
            created += 1

    print(f"作成完了: PNG {created}枚、MP4 {created}本")
    print("明るさ倍率: " + ", ".join(f"gC{key}={value:.2f}" for key, value in CENTER_BRIGHTNESS.items()))
    print("照明効果: shade=基準、sun_back=暗化+青寄り、sun_front=明化+白寄り+暖色寄り")
    print(f"PNG: {IMAGE_DIRECTORY}")
    print(f"動画: {VIDEO_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
