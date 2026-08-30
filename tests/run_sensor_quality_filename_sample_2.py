"""サンプル2をA-1評価・グラフ化し、明るさと照明効果を検証する。"""

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


VIDEO_DIRECTORY = Path("tests/data/sensor_quality_filename_samples_2/videos")
OUTPUT_DIRECTORY = Path("analysis/sensor_quality_filename_sample_2")


def color_chart_patch_rois(width: int, height: int) -> dict[str, list[int]]:
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
    capture = cv2.VideoCapture(str(video_paths[0]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
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
    payload = json.loads((OUTPUT_DIRECTORY / "sensor_quality_summary.json").read_text(encoding="utf-8"))
    results = payload["results"]
    if len(results) != 12:
        raise AssertionError(f"評価結果は12本のはずですが、{len(results)}本です。")

    shade_results = []
    light_results: dict[tuple[str, str], dict] = {}
    for result in results:
        conditions = result["conditions"]
        center = conditions["center_gaussian"]
        light_condition = conditions["light_condition"]
        light_results[(center, light_condition)] = result
        if light_condition == "shade-none":
            shade_results.append(result)

    gray_l_values = [
        (int(result["conditions"]["center_gaussian"]), result["patches"]["gray"]["l_mean"])
        for result in shade_results
    ]
    gray_l_values.sort()
    if not all(next_value[1] > value[1] for value, next_value in zip(gray_l_values, gray_l_values[1:])):
        raise AssertionError(f"gC増加に伴う明るさ増加を確認できません: {gray_l_values}")

    for center, shade_result in ((r["conditions"]["center_gaussian"], r) for r in shade_results):
        shade_gray_l = shade_result["patches"]["gray"]["l_mean"]
        back_gray_l = light_results[(center, "sun-back")]["patches"]["gray"]["l_mean"]
        front_gray_l = light_results[(center, "sun-front")]["patches"]["gray"]["l_mean"]
        if not back_gray_l < shade_gray_l < front_gray_l:
            raise AssertionError(
                f"照明による灰色パッチ明度の順序が不正です (gC{center}): "
                f"back={back_gray_l}, shade={shade_gray_l}, front={front_gray_l}"
            )
        front_red_difference = light_results[(center, "sun-front")][
            "patch_opponent_polar_difference_from_shade"
        ]["red"]
        if (
            abs(front_red_difference["radius_change"]) <= 1e-6
            and abs(front_red_difference["theta_difference_deg"] or 0.0) <= 1e-6
        ):
            raise AssertionError(f"sun-frontの赤パッチに変化がありません (gC{center})")

    manifest = json.loads((OUTPUT_DIRECTORY / "plots" / "plot_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("x_axis_field") != "center_gaussian":
        raise AssertionError(f"横軸がgCになっていません: {manifest.get('x_axis_field')}")
    svg_paths = [Path(path) for path in manifest["outputs"] if path.endswith(".svg")]
    if len(svg_paths) != 3:
        raise AssertionError(f"今回の実行で作成したSVGは3枚のはずですが、{len(svg_paths)}枚です。")
    for svg_path in svg_paths:
        element_tree.parse(svg_path)
    print("確認完了: gCとともに明るさが増加し、逆光は暗化、正面直射は明化・色変化、横軸はgCです。")


def main() -> int:
    video_paths = sorted(VIDEO_DIRECTORY.glob("*.mp4"))
    if len(video_paths) != 12:
        raise FileNotFoundError(
            "サンプル2動画が12本ありません。先に create_sensor_quality_filename_samples_2.py を実行してください。"
        )
    quality.VIDEO_GLOB = "tests/data/sensor_quality_filename_samples_2/videos/*.mp4"
    quality.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    quality.ROI_PROFILE_NAME = "sample2_color_chart"
    quality.RESELECT_ROIS = False
    quality.REPRESENTATIVE_FRAME_INDEX = 0
    prepare_rois(video_paths)
    if quality.main() != 0:
        raise RuntimeError("measure_sensor_quality.py のサンプル2実行に失敗しました。")

    plot.INPUT_SUMMARY_JSON = OUTPUT_DIRECTORY / "sensor_quality_summary.json"
    plot.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
    if plot.main() != 0:
        raise RuntimeError("plot_sensor_quality.py のサンプル2実行に失敗しました。")
    verify_results()
    print(f"結果フォルダ: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
