"""同一カラーチャート画像から、ファイル名条件だけを変えたA-1用サンプルを作成する。"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np


SAMPLE_ROOT = Path(__file__).resolve().parent / "data" / "sensor_quality_filename_samples"
IMAGE_DIRECTORY = SAMPLE_ROOT / "images"
VIDEO_DIRECTORY = SAMPLE_ROOT / "videos"
SOURCE_IMAGE_NAME = "source_color_chart.png"

# measure_sensor_quality.py が読み取る表記に合わせる。中心側は 1 に固定し、
# surround 側と照明条件だけを変更する。
SURROUND_VALUES = ("04", "06", "08", "10")
LIGHT_CONDITIONS = ("shade_none", "sun_back", "sun_front")


def read_image(path: Path) -> np.ndarray:
    """Unicode を含むパスでも画像を読み込む。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return image


def sample_stem(surround: str, light_condition: str) -> str:
    return (
        "20260827_1200_sample_nowide_"
        f"gC01_gS{surround}_iir040_nrA_{light_condition}_d1p0m_front_t01"
    )


def write_one_frame_video(path: Path, image: np.ndarray) -> None:
    """画像を1秒・1 fps・1フレームのMP4として保存する。"""
    height, width = image.shape[:2]
    # OpenCVのVideoWriterはUnicodeパスを扱えない環境があるため、ASCIIの一時場所に
    # 生成してから目的のパスへコピーする。
    with tempfile.TemporaryDirectory(prefix="sensor_quality_sample_") as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0,
            (width, height),
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
    parser = argparse.ArgumentParser(description="同一画像のA-1ファイル名サンプルを作成する")
    parser.add_argument(
        "--source",
        type=Path,
        help="元画像のパス。最初の実行時だけ指定する。以後は保存済み元画像を利用できる。",
    )
    arguments = parser.parse_args()

    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    source_copy_path = SAMPLE_ROOT / SOURCE_IMAGE_NAME
    if arguments.source is not None:
        if not arguments.source.exists():
            raise FileNotFoundError(f"元画像が見つかりません: {arguments.source}")
        source_copy_path.write_bytes(arguments.source.read_bytes())
    if not source_copy_path.exists():
        raise FileNotFoundError(
            "元画像がありません。初回は --source <添付画像のパス> を指定してください。"
        )

    # PNGサンプルは元画像のバイト列をそのまま複製するため、画素だけでなくPNGとしても同一。
    source_bytes = source_copy_path.read_bytes()
    image = read_image(source_copy_path)
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)

    created = 0
    for surround in SURROUND_VALUES:
        for light_condition in LIGHT_CONDITIONS:
            stem = sample_stem(surround, light_condition)
            image_path = IMAGE_DIRECTORY / f"{stem}.png"
            video_path = VIDEO_DIRECTORY / f"{stem}.mp4"
            image_path.write_bytes(source_bytes)
            write_one_frame_video(video_path, image)
            created += 1

    print(f"作成完了: PNG {created}枚、MP4 {created}本")
    print(f"元画像コピー: {source_copy_path}")
    print(f"PNG: {IMAGE_DIRECTORY}")
    print(f"動画: {VIDEO_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
