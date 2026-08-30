"""サンプル3を評価・グラフ化し、正面直射のradius増加を検証する。"""

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


VIDEO_DIRECTORY = Path("tests/data/sensor_quality_filename_samples_3/videos")
OUTPUT_DIRECTORY = Path("analysis/sensor_quality_filename_sample_3")


def make_rois(width: int, height: int) -> dict[str, list[int]]:
    tile_width, tile_height = width // 3, height // 2
    if width % 3 or height % 2 or tile_width < 4 or tile_height < 4:
        raise ValueError(f"3列×2行のカラーチャートではありません: {width}x{height}")
    margin = max(1, min(tile_width, tile_height) // 12)
    return {
        name: [
            (index % 3) * tile_width + margin,
            (index // 3) * tile_height + margin,
            tile_width - margin * 2,
            tile_height - margin * 2,
        ]
        for index, name in enumerate(("white", "gray", "black", "red", "green", "blue"))
    }


def prepare_rois(video_paths: list[Path]) -> None:
    capture = cv2.VideoCapture(str(video_paths[0]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    roi_data = quality.new_roi_data()
    for video_path in video_paths:
        roi_data["videos"][str(video_path.resolve())] = {
            "frame_width": width,
            "frame_height": height,
            "representative_frame_index": 0,
            "patch_rois_xywh": make_rois(width, height),
        }
    quality.atomic_json_dump(OUTPUT_DIRECTORY / "sample3_color_chart_rois.json", roi_data)


def verify_results() -> None:
    results = json.loads(
        (OUTPUT_DIRECTORY / "sensor_quality_summary.json").read_text(encoding="utf-8")
    )["results"]
    if len(results) != 12:
        raise AssertionError(f"評価結果は12本のはずですが、{len(results)}本です。")
    for result in results:
        if result["conditions"]["light_condition"] != "sun-front":
            continue
        for patch in ("red", "green", "blue"):
            change = result["patch_opponent_polar_difference_from_shade"][patch]["radius_change"]
            if change <= 0:
                raise AssertionError(
                    f"正面直射で{patch}のradiusが増えていません: "
                    f"gC{result['conditions']['center_gaussian']} / {change}"
                )
    manifest = json.loads((OUTPUT_DIRECTORY / "plots" / "plot_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("x_axis_field") != "center_gaussian":
        raise AssertionError("横軸がgCではありません。")
    svg_paths = [Path(path) for path in manifest["outputs"] if path.endswith(".svg")]
    if len(svg_paths) != 3:
        raise AssertionError("今回作成したSVGが3枚ではありません。")
    for path in svg_paths:
        element_tree.parse(path)
    print("確認完了: 正面直射で赤・緑・青すべてのradiusが増加しています。")


def main() -> int:
    video_paths = sorted(VIDEO_DIRECTORY.glob("*.mp4"))
    if len(video_paths) != 12:
        raise FileNotFoundError("サンプル3動画が12本ありません。先に作成プログラムを実行してください。")
    quality.VIDEO_GLOB = "tests/data/sensor_quality_filename_samples_3/videos/*.mp4"
    quality.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    quality.ROI_PROFILE_NAME = "sample3_color_chart"
    quality.RESELECT_ROIS = False
    quality.REPRESENTATIVE_FRAME_INDEX = 0
    prepare_rois(video_paths)
    if quality.main() != 0:
        raise RuntimeError("サンプル3の評価に失敗しました。")
    plot.INPUT_SUMMARY_JSON = OUTPUT_DIRECTORY / "sensor_quality_summary.json"
    plot.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
    if plot.main() != 0:
        raise RuntimeError("サンプル3のグラフ化に失敗しました。")
    verify_results()
    print(f"結果フォルダ: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
