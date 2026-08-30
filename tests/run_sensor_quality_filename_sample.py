"""ファイル名サンプル動画をA-1評価・グラフ化し、期待どおりか確認する。"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import measure_sensor_quality as quality
from tools import plot_sensor_quality as plot


VIDEO_DIRECTORY = Path("tests/data/sensor_quality_filename_samples/videos")
OUTPUT_DIRECTORY = Path("analysis/sensor_quality_filename_sample")


def color_chart_patch_rois(width: int, height: int) -> dict[str, list[int]]:
    """3列×2行のカラーチャートの各タイル中央をROIとして返す。"""
    if width < 12 or height < 8 or width % 3 != 0 or height % 2 != 0:
        raise ValueError(f"サンプル画像が3列×2行のチャートとして扱えません: {width}x{height}")
    tile_width, tile_height = width // 3, height // 2
    margin = max(1, min(tile_width, tile_height) // 12)
    names = ("white", "gray", "black", "red", "green", "blue")
    rois: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        row, column = divmod(index, 3)
        rois[name] = [
            column * tile_width + margin,
            row * tile_height + margin,
            tile_width - margin * 2,
            tile_height - margin * 2,
        ]
    return rois


def prepare_rois(video_paths: list[Path]) -> None:
    """GUI選択なしで、全サンプル動画に同じチャートROIを登録する。"""
    first_capture = cv2.VideoCapture(str(video_paths[0]))
    width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_capture.release()
    rois = color_chart_patch_rois(width, height)
    roi_data = quality.new_roi_data()
    for video_path in video_paths:
        roi_data["videos"][str(video_path.resolve())] = {
            "frame_width": width,
            "frame_height": height,
            "representative_frame_index": 0,
            "patch_rois_xywh": rois,
        }
    quality.atomic_json_dump(
        OUTPUT_DIRECTORY / f"{quality.ROI_PROFILE_NAME}_rois.json", roi_data
    )


def verify_results() -> None:
    """同一画像のため、12本の結果と日陰との差0を確認する。"""
    summary_path = OUTPUT_DIRECTORY / "sensor_quality_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload["results"]
    if len(results) != 12:
        raise AssertionError(f"評価結果は12本のはずですが、{len(results)}本です。")

    for result in results:
        if result["reference_video"] == "":
            raise AssertionError(f"日陰基準が見つかりません: {result['video_name']}")
        for patch_name, difference in result["patch_opponent_polar_difference_from_shade"].items():
            if abs(difference["radius_change"]) > 1e-6:
                raise AssertionError(f"{patch_name} のradius差が0ではありません: {difference}")
            theta_difference = difference["theta_difference_deg"]
            if theta_difference is not None and abs(theta_difference) > 1e-6:
                raise AssertionError(f"{patch_name} の色相差が0ではありません: {difference}")

    manifest = json.loads((OUTPUT_DIRECTORY / "plots" / "plot_manifest.json").read_text(encoding="utf-8"))
    svg_paths = [Path(path) for path in manifest["outputs"] if path.endswith(".svg")]
    if len(svg_paths) != 3:
        raise AssertionError(f"今回の実行で作成したSVGは3枚のはずですが、{len(svg_paths)}枚です。")
    for svg_path in svg_paths:
        element_tree.parse(svg_path)
    print("確認完了: 12本を評価し、すべての日陰との差が0であることとSVG 3枚を確認しました。")


def main() -> int:
    video_paths = sorted(VIDEO_DIRECTORY.glob("*.mp4"))
    if len(video_paths) != 12:
        raise FileNotFoundError(
            "サンプル動画が12本ありません。先に tests/create_sensor_quality_filename_samples.py を実行してください。"
        )

    quality.VIDEO_GLOB = "tests/data/sensor_quality_filename_samples/videos/*.mp4"
    quality.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    quality.ROI_PROFILE_NAME = "sample_color_chart"
    quality.RESELECT_ROIS = False
    quality.REPRESENTATIVE_FRAME_INDEX = 0
    prepare_rois(video_paths)
    if quality.main() != 0:
        raise RuntimeError("measure_sensor_quality.py のサンプル実行に失敗しました。")

    plot.INPUT_SUMMARY_JSON = OUTPUT_DIRECTORY / "sensor_quality_summary.json"
    plot.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
    if plot.main() != 0:
        raise RuntimeError("plot_sensor_quality.py のサンプル実行に失敗しました。")
    verify_results()
    print(f"結果フォルダ: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
