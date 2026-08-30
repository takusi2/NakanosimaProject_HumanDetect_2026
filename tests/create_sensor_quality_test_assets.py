"""measure_sensor_quality.py 用の既知色テスト画像・動画を作成する。"""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np


DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
TILE_WIDTH = 60
TILE_HEIGHT = 60

# OpenCVのBGR順。上段: 白・灰・黒、下段: 赤・緑・青。
PATCH_COLORS_BGR = {
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def make_color_chart(red_patch_color: tuple[int, int, int]) -> np.ndarray:
    """余白のない6色パッチ画像を作成する。"""
    chart = np.zeros((TILE_HEIGHT * 2, TILE_WIDTH * 3, 3), dtype=np.uint8)
    colors = (
        PATCH_COLORS_BGR["white"],
        PATCH_COLORS_BGR["gray"],
        PATCH_COLORS_BGR["black"],
        red_patch_color,
        PATCH_COLORS_BGR["green"],
        PATCH_COLORS_BGR["blue"],
    )
    for index, color in enumerate(colors):
        row, column = divmod(index, 3)
        top = row * TILE_HEIGHT
        left = column * TILE_WIDTH
        chart[top : top + TILE_HEIGHT, left : left + TILE_WIDTH] = color
    return chart


def write_image(path: Path, image: np.ndarray) -> None:
    """OpenCVがUnicodeパスでもPNGを保存できるようにする。"""
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"画像のエンコードに失敗しました: {path}")
    path.write_bytes(encoded.tobytes())


def main() -> int:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    representative = make_color_chart(PATCH_COLORS_BGR["red"])
    # 動画の後続フレームでは、赤パッチを青へ置換する。
    # 代表フレームだけを使う実装なら、red ROI は赤のまま評価されるはずである。
    changed_later = make_color_chart(PATCH_COLORS_BGR["blue"])

    representative_path = DATA_DIRECTORY / "representative_color_chart.png"
    changed_later_path = DATA_DIRECTORY / "later_changed_color_chart.png"
    write_image(representative_path, representative)
    write_image(changed_later_path, changed_later)

    video_path = DATA_DIRECTORY / "representative_then_changed.avi"
    # VideoWriterもUnicodeパスを扱えない環境があるため、一時的なASCIIパスへ
    # 出力してからプロジェクト内のテストデータへコピーする。
    with tempfile.TemporaryDirectory(prefix="sensor_quality_test_") as temporary_directory:
        temporary_video_path = Path(temporary_directory) / "representative_then_changed.avi"
        writer = cv2.VideoWriter(
            str(temporary_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (representative.shape[1], representative.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"テスト動画を作成できませんでした: {temporary_video_path}")
        writer.write(representative)
        writer.write(changed_later)
        writer.release()

        capture = cv2.VideoCapture(str(temporary_video_path))
        readable = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        capture.release()
        if not readable:
            raise RuntimeError(f"保存したテスト動画を読み込めませんでした: {temporary_video_path}")
        video_path.write_bytes(temporary_video_path.read_bytes())
    print(f"作成完了: {DATA_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
