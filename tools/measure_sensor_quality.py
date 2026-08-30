"""A-1: Center-Surround のガウシアン値を、モデル非依存で比較するツール。

このファイル先頭の「利用者設定」を編集してから実行する。

    python tools/measure_sensor_quality.py
    python tools/measure_sensor_quality.py --patches red green
    python tools/measure_sensor_quality.py --patches white,gray,black

最初の実行時には、動画ごとの代表フレームで同じカラーチャートのROIを指定する。
  white / gray / black / red / green / blue

カラーチャートの位置や背景が動画ごとに異なっても、各動画でROIを選択するため
比較できる。背景や服領域を共通座標で扱わず、以下を保存する。
  - 飽和画素率
  - 各カラーパッチの色（Lab）と彩度の目安
  - 反対色表現の極座標（論文 3.5 節）: 半径 r と色相角 theta
  - チャート内の白黒明度差（ダイナミックレンジの目安）
  - 同一センサ設定内の shade-none 動画との差分

色評価は、手振れによるROIずれを混入させないため、代表フレーム1枚だけで行う。
ブレ・IIRの時間的な効果は、このツールでは評価しない。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# 利用者設定：撮影した動画に合わせて、ここだけを書き換える
# ============================================================
# A-1用の動画だけに絞る。例: "./output/a1/*.mp4"
VIDEO_GLOB = "./output/*.mp4"

# 出力先。ROI定義、動画ごとの詳細JSON、一覧CSVを保存する。
OUTPUT_DIRECTORY = Path("./analysis/sensor_quality_a1")

# ROI定義ファイルを識別する名前。ROIは動画ごとに保存される。
ROI_PROFILE_NAME = "a1_color_chart"

# Trueにすると、既存のROI定義があっても選び直す。
RESELECT_ROIS = True

# 色評価に使う代表フレーム（0始まり）。通常は先頭フレームの 0 を使う。
# 撮影開始直後に自動露出・ホワイトバランスが安定しない場合は、安定後の番号へ変更する。
# 値を変更すると、その動画のROIを代表フレーム上で自動的に選び直す。
REPRESENTATIVE_FRAME_INDEX = 0

# shade-none の基準動画を探す際、撮影場所(site)も一致させるか。
# A-1では場所の異なる日陰・直射動画も比較するため、通常は False のまま使う。
REFERENCE_REQUIRE_SAME_SITE = False

# 飽和と見なす画素値。動画を正規化している場合も、すべての設定で同じ値を使う。
SATURATION_LOW = 5
SATURATION_HIGH = 250

# カラーチャートで選択するパッチ。パッチは十分な画素数が得られる大きさで撮影する。
PATCH_NAMES = ("white", "gray", "black", "red", "green", "blue")
PATCH_PROMPTS = {
    "white": "白パッチ",
    "gray": "灰色パッチ",
    "black": "黒パッチ",
    "red": "赤パッチ",
    "green": "緑パッチ",
    "blue": "青パッチ",
}


def parse_patch_names(values: list[str] | None) -> tuple[str, ...]:
    """--patches の指定を検証し、定義済みの色順に整列して返す。"""
    if values is None:
        return PATCH_NAMES

    requested = {
        patch_name.strip().lower()
        for value in values
        for patch_name in value.split(",")
        if patch_name.strip()
    }
    unknown_names = requested.difference(PATCH_NAMES)
    if unknown_names:
        valid_names = ", ".join(PATCH_NAMES)
        unknown_text = ", ".join(sorted(unknown_names))
        raise ValueError(
            f"未対応の色が指定されました: {unknown_text}（指定可能: {valid_names}）"
        )
    if not requested:
        raise ValueError("--patches には少なくとも1色を指定してください。")
    return tuple(name for name in PATCH_NAMES if name in requested)


def atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def new_roi_data() -> dict[str, Any]:
    """動画ごとのカラーチャートROI定義の初期データを返す。"""
    return {"profile": ROI_PROFILE_NAME, "roi_mode": "per_video_color_chart", "videos": {}}


def load_roi_data(roi_path: Path) -> dict[str, Any]:
    """既存ROIを常に読み込む。RESELECT_ROIS は読み込み可否に影響させない。"""
    if roi_path.exists():
        return read_json(roi_path)
    return new_roi_data()


def parse_filename_conditions(video_path: Path) -> dict[str, str]:
    """実験計画のファイル名規則から、比較に使う条件を抽出する。"""
    tokens = video_path.stem.split("_")
    conditions: dict[str, str] = {"file_stem": video_path.stem}
    if len(tokens) >= 3 and re.fullmatch(r"\d{8}", tokens[0]):
        conditions["date"] = tokens[0]
        conditions["time"] = tokens[1]
        conditions["site"] = tokens[2]

    token_patterns = {
        "lens": r"^(nowide|wide)$",
        "center_gaussian": r"^gC(.+)$",
        "surround_gaussian": r"^gS(.+)$",
        "iir": r"^(iir.+)$",
        "naka_rushton": r"^(nr.+)$",
        "light": r"^(shade|sun|cloudy)-.+$",
        "sun_direction": r"^(?:shade|sun|cloudy)-(.+)$",
        "distance": r"^(d.+m)$",
        "pose": r"^(front|diag|side)$",
        "trial": r"^(t\d+)$",
    }
    for token in tokens:
        for key, pattern in token_patterns.items():
            match = re.fullmatch(pattern, token)
            if match:
                conditions[key] = match.group(1) if match.groups() else token
                break

        light_match = re.fullmatch(r"^(shade|sun|cloudy)-(.+)$", token)
        if light_match:
            conditions["light"] = light_match.group(1)
            conditions["sun_direction"] = light_match.group(2)

    # ファイル名に ``shade_none`` / ``sun_back`` のようにアンダースコアを
    # 使う場合、split("_") 後は二つのトークンになるため、隣接トークンからも
    # 照明条件を復元する。従来の ``shade-none`` 形式にも対応したままとする。
    for index, token in enumerate(tokens[:-1]):
        if token in {"shade", "sun", "cloudy"}:
            conditions["light"] = token
            conditions["sun_direction"] = tokens[index + 1]

    # 日時・照明・繰返しを除いた、同一センサ設定の比較キー。
    # 通常は場所の異なる日陰・直射動画も比較できるよう、siteは含めない。
    comparison_keys = [
        "lens",
        "center_gaussian",
        "surround_gaussian",
        "iir",
        "naka_rushton",
        "distance",
        "pose",
    ]
    if REFERENCE_REQUIRE_SAME_SITE:
        comparison_keys.insert(0, "site")
    key_parts = [conditions.get(key, "unknown") for key in comparison_keys]
    conditions["comparison_group"] = "|".join(key_parts)
    conditions["light_condition"] = (
        f"{conditions.get('light', 'unknown')}-{conditions.get('sun_direction', 'unknown')}"
    )
    return conditions


def validate_roi(roi: list[int], width: int, height: int, name: str) -> None:
    if len(roi) != 4:
        raise ValueError(f"{name} ROIの形式が不正です: {roi}")
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 1 or h <= 1 or x + w > width or y + h > height:
        raise ValueError(
            f"{name} ROI {roi} が動画サイズ {width}x{height} の範囲外です。ROIを選び直してください。"
        )


def roi_from_top_left_drag(
    start: tuple[int, int], end: tuple[int, int]
) -> list[int] | None:
    """左上から右下へドラッグした場合だけ、ROIのxywhを返す。"""
    start_x, start_y = start
    end_x, end_y = end
    if end_x <= start_x or end_y <= start_y:
        return None
    return [start_x, start_y, end_x - start_x, end_y - start_y]


def select_roi_top_left_to_bottom_right(
    window_name: str, image: np.ndarray, initial_roi: list[int] | None = None
) -> list[int]:
    """左上始点・右下終点のみを受け付けるROI選択画面を表示する。

    initial_roi がある場合は最初から緑枠で表示し、Enter/Spaceでそのまま採用できる。
    新たに正しい方向でドラッグした場合だけ、そのROIへ置き換える。
    """
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    retained_roi = list(initial_roi) if initial_roi is not None else None
    selected_roi = list(initial_roi) if initial_roi is not None else None

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        nonlocal start, current, selected_roi
        if event == cv2.EVENT_LBUTTONDOWN:
            # 既存ROIは残したまま、新しい選択のドラッグを開始する。
            start = (x, y)
            current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            current = (x, y)
            selected_roi = roi_from_top_left_drag(start, current)
            if selected_roi is None:
                print("左上から右下へドラッグしてください。選択をやり直してください。")
            start = None
            current = None

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = image.copy()
        if start is not None and current is not None:
            # ドラッグ中は緑: 正しい方向、赤: 逆方向を示す。
            dragging_roi = roi_from_top_left_drag(start, current)
            color = (0, 255, 0) if dragging_roi is not None else (0, 0, 255)
            cv2.rectangle(preview, start, current, color, 2)
        if selected_roi is not None:
            x, y, width, height = selected_roi
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            if selected_roi is not None:
                cv2.destroyWindow(window_name)
                return selected_roi
            print("ROIを左上から右下へドラッグしてから Enter または Space を押してください。")
        elif key in (ord("r"), ord("c")):
            start = None
            current = None
            selected_roi = list(retained_roi) if retained_roi is not None else None
            if retained_roi is not None:
                print("変更中のROIを取り消し、登録済みのROIへ戻しました。")
            else:
                print("未確定のROIを取り消しました。もう一度選択してください。")
        elif key == 27:  # Esc
            cv2.destroyWindow(window_name)
            raise RuntimeError("ROI選択をキャンセルしました。")


def select_rois_for_video(
    representative_frame: np.ndarray,
    video_path: Path,
    patch_names: tuple[str, ...],
    existing_patch_rois: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """各動画の代表フレームで、カラーチャートのパッチを選択する。"""
    height, width = representative_frame.shape[:2]
    patch_rois: dict[str, list[int]] = {}
    for patch_name in patch_names:
        initial_roi = (existing_patch_rois or {}).get(patch_name)
        if initial_roi is not None:
            try:
                validate_roi(initial_roi, width, height, patch_name)
            except ValueError:
                print(
                    f"{video_path.name}: 登録済みの{PATCH_PROMPTS[patch_name]}ROIは"
                    "現在のフレーム範囲外のため表示しません。"
                )
                initial_roi = None
        print(
            f"{video_path.name}: {PATCH_PROMPTS[patch_name]}を選択してください。"
            "左上から右下へドラッグし、Enter または Space で確定します。"
            "R / C で未確定の選択を取り消せます。"
        )
        if initial_roi is not None:
            print("登録済みROIを緑枠で表示しています。変更しなければ Enter / Space を押してください。")
        roi = select_roi_top_left_to_bottom_right(
            f"Select {patch_name} patch", representative_frame, initial_roi
        )
        validate_roi(roi, width, height, patch_name)
        patch_rois[patch_name] = roi
    cv2.destroyAllWindows()
    return {
        "frame_width": width,
        "frame_height": height,
        "representative_frame_index": REPRESENTATIVE_FRAME_INDEX,
        "patch_rois_xywh": patch_rois,
    }


def get_video_rois(
    roi_data: dict[str, Any],
    video_path: Path,
    representative_frame: np.ndarray,
    patch_names: tuple[str, ...],
) -> dict[str, Any]:
    """動画ごとのROIを読み込み、必要な場合だけ代表フレーム上で指定する。"""
    videos = roi_data.setdefault("videos", {})
    video_key = str(video_path.resolve())
    existing_video_rois = videos.get(video_key)
    saved_frame_index = (existing_video_rois or {}).get("representative_frame_index", 0)
    existing_patch_rois = (existing_video_rois or {}).get("patch_rois_xywh", {})
    missing_requested_roi = any(name not in existing_patch_rois for name in patch_names)
    if (
        RESELECT_ROIS
        or video_key not in videos
        or saved_frame_index != REPRESENTATIVE_FRAME_INDEX
        or missing_requested_roi
    ):
        # 別の代表フレームで登録したROIは位置がずれるため、表示せず選び直す。
        existing_patch_rois = (
            existing_video_rois.get("patch_rois_xywh")
            if existing_video_rois is not None
            and saved_frame_index == REPRESENTATIVE_FRAME_INDEX
            else None
        )
        selected_rois = select_rois_for_video(
            representative_frame, video_path, patch_names, existing_patch_rois
        )
        # 指定していない色の既存ROIは残す。後から別の色を追加調査できるようにする。
        patch_rois = dict(existing_patch_rois or {})
        patch_rois.update(selected_rois["patch_rois_xywh"])
        selected_rois["patch_rois_xywh"] = patch_rois
        videos[video_key] = selected_rois
    return videos[video_key]


def crop(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, width, height = roi
    return frame[y : y + height, x : x + width]


def saturation_ratio(image: np.ndarray) -> float:
    saturated = np.any(
        (image <= SATURATION_LOW) | (image >= SATURATION_HIGH), axis=2
    )
    return float(np.mean(saturated))


def lab_statistics(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Lab平均、明度L平均、彩度の目安（a/bからの距離）を返す。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    mean_l = float(mean_lab[0])
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    return mean_lab, mean_l, float(chroma.mean())


def opponent_polar_sums(image: np.ndarray) -> dict[str, float]:
    """ROI内の全画素を反対色表現の極座標へ変換し、平均用の合計値を返す。

    論文 3.5 節の定義を用いる。
      O_RG = R - G
      O_YB = (R + G) / 2 - B
      r = sqrt(O_RG^2 + O_YB^2)
      theta = atan2(O_YB, O_RG)

    theta は 0 度と 360 度が連続する量なので、あとで sin/cos の合計から
    円平均を求める。r=0 の画素では theta が定義できないため、角度平均だけ
    から除外する。ただし r、O_RG、O_YB の平均には全画素を含める。
    """
    bgr = image.astype(np.float32)
    blue = bgr[..., 0]
    green = bgr[..., 1]
    red = bgr[..., 2]
    opponent_rg = red - green
    opponent_yb = (red + green) / 2.0 - blue
    radius = np.hypot(opponent_rg, opponent_yb)
    theta = np.arctan2(opponent_yb, opponent_rg)
    valid_theta = radius > 0.0

    return {
        "pixel_count": float(radius.size),
        "opponent_rg_sum": float(np.sum(opponent_rg)),
        "opponent_yb_sum": float(np.sum(opponent_yb)),
        "radius_sum": float(np.sum(radius)),
        "theta_cos_sum": float(np.sum(np.cos(theta[valid_theta]))),
        "theta_sin_sum": float(np.sum(np.sin(theta[valid_theta]))),
        "theta_pixel_count": float(np.count_nonzero(valid_theta)),
    }


def opponent_polar_statistics(sums: dict[str, float]) -> dict[str, float | None]:
    """ROI内の全画素から反対色極座標の平均を返す。"""
    pixel_count = sums["pixel_count"]
    theta_pixel_count = sums["theta_pixel_count"]
    if pixel_count <= 0:
        raise ValueError("反対色表現を計算する画素がありません。")

    statistics: dict[str, float | None] = {
        "opponent_rg_mean": sums["opponent_rg_sum"] / pixel_count,
        "opponent_yb_mean": sums["opponent_yb_sum"] / pixel_count,
        "radius_mean": sums["radius_sum"] / pixel_count,
        "pixels_averaged": int(pixel_count),
        "theta_pixels_averaged": int(theta_pixel_count),
        "theta_deg_circular_mean": None,
        "theta_resultant_length": None,
        "theta_deg_circular_std": None,
    }
    if theta_pixel_count <= 0:
        return statistics

    mean_cos = sums["theta_cos_sum"] / theta_pixel_count
    mean_sin = sums["theta_sin_sum"] / theta_pixel_count
    # 理論上は 0〜1。浮動小数点の丸めでわずかに1を超える場合を補正する。
    resultant_length = min(1.0, float(np.hypot(mean_cos, mean_sin)))
    theta_degrees = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
    # 円標準偏差。R=0 に近いほど角度がばらつき、値が大きくなる。
    circular_std = float(
        np.degrees(np.sqrt(-2.0 * np.log(max(resultant_length, 1e-12))))
    )
    statistics.update(
        {
            "theta_deg_circular_mean": theta_degrees,
            "theta_resultant_length": resultant_length,
            "theta_deg_circular_std": circular_std,
        }
    )
    return statistics


def circular_angle_difference_degrees(first: float | None, second: float | None) -> float | None:
    """二つの角度の最短差を 0〜180 度で返す。"""
    if first is None or second is None:
        return None
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def read_representative_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """指定番号のフレームを読み込む。フレーム番号は0始まり。"""
    if frame_index < 0:
        raise ValueError("REPRESENTATIVE_FRAME_INDEX は 0 以上を指定してください。")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"代表フレーム {frame_index} を読み込めませんでした。動画の長さを確認してください。"
        )
    return frame


def analyze_video(
    video_path: Path, video_rois: dict[str, Any], patch_names: tuple[str, ...]
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    patch_rois = video_rois["patch_rois_xywh"]
    for patch_name in patch_names:
        if patch_name not in patch_rois:
            raise ValueError(f"{patch_name} パッチのROIがありません。")
        validate_roi(patch_rois[patch_name], width, height, patch_name)

    capture.set(cv2.CAP_PROP_POS_FRAMES, REPRESENTATIVE_FRAME_INDEX)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"代表フレーム {REPRESENTATIVE_FRAME_INDEX} を読み込めませんでした。"
        )

    patches: dict[str, dict[str, Any]] = {}
    for patch_name in patch_names:
        patch_image = crop(frame, patch_rois[patch_name])
        lab_mean, mean_l, chroma_mean = lab_statistics(patch_image)
        patches[patch_name] = {
            "lab_mean": [float(v) for v in lab_mean],
            "l_mean": mean_l,
            "chroma_mean": chroma_mean,
            "saturation_ratio": saturation_ratio(patch_image),
            "opponent_polar": opponent_polar_statistics(opponent_polar_sums(patch_image)),
        }

    white_black_range = (
        patches["white"]["l_mean"] - patches["black"]["l_mean"]
        if {"white", "black"}.issubset(patches)
        else None
    )
    primary_patch_names = ("red", "green", "blue")
    primary_chroma_mean = (
        float(np.mean([patches[name]["chroma_mean"] for name in primary_patch_names]))
        if set(primary_patch_names).issubset(patches)
        else None
    )
    return {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "video_width": width,
        "video_height": height,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "evaluation_mode": "single_representative_frame",
        "representative_frame_index": REPRESENTATIVE_FRAME_INDEX,
        "representative_frame_time_seconds": (
            REPRESENTATIVE_FRAME_INDEX / fps if fps > 0 else None
        ),
        "frames_analyzed": 1,
        "saturation_ratio_full": saturation_ratio(frame),
        "patches": patches,
        "white_black_l_range": (
            float(white_black_range) if white_black_range is not None else None
        ),
        "primary_chroma_mean": primary_chroma_mean,
    }


def delta_e_approximate(first_lab: list[float], second_lab: list[float]) -> float:
    """OpenCVの8-bit Lab値での色差の近似値。相対比較用として用いる。"""
    return float(np.linalg.norm(np.asarray(first_lab) - np.asarray(second_lab)))


def add_reference_differences(
    results: list[dict[str, Any]], patch_names: tuple[str, ...]
) -> None:
    """同一センサ設定群の shade-none を基準に、チャートの色差を計算する。"""
    references: dict[str, dict[str, Any]] = {}
    for result in results:
        conditions = result["conditions"]
        if conditions.get("light_condition") == "shade-none":
            references.setdefault(conditions["comparison_group"], result)

    for result in results:
        reference = references.get(result["conditions"]["comparison_group"])
        if reference is None:
            result["reference_video"] = ""
            result["patch_delta_lab_from_shade"] = {}
            result["patch_opponent_polar_difference_from_shade"] = {}
            result["white_black_l_range_change_from_shade"] = None
            continue
        result["reference_video"] = reference["video_name"]
        result["patch_delta_lab_from_shade"] = {
            patch_name: delta_e_approximate(
                result["patches"][patch_name]["lab_mean"],
                reference["patches"][patch_name]["lab_mean"],
            )
            for patch_name in patch_names
        }
        if (
            result["white_black_l_range"] is not None
            and reference["white_black_l_range"] is not None
        ):
            result["white_black_l_range_change_from_shade"] = (
                result["white_black_l_range"] - reference["white_black_l_range"]
            )
        else:
            result["white_black_l_range_change_from_shade"] = None
        result["patch_opponent_polar_difference_from_shade"] = {
            patch_name: {
                "radius_change": (
                    result["patches"][patch_name]["opponent_polar"]["radius_mean"]
                    - reference["patches"][patch_name]["opponent_polar"]["radius_mean"]
                ),
                "theta_difference_deg": circular_angle_difference_degrees(
                    result["patches"][patch_name]["opponent_polar"][
                        "theta_deg_circular_mean"
                    ],
                    reference["patches"][patch_name]["opponent_polar"][
                        "theta_deg_circular_mean"
                    ],
                ),
            }
            for patch_name in patch_names
        }


def write_summary_csv(
    results: list[dict[str, Any]], path: Path, patch_names: tuple[str, ...]
) -> None:
    fields = [
        "video_name",
        "comparison_group",
        "light_condition",
        "center_gaussian",
        "surround_gaussian",
        "iir",
        "naka_rushton",
        "evaluation_mode",
        "representative_frame_index",
        "frames_analyzed",
        "saturation_ratio_full",
        "white_black_l_range",
        "primary_chroma_mean",
        "reference_video",
    ]
    fields.extend(f"{patch_name}_delta_lab_from_shade" for patch_name in patch_names)
    fields.append("white_black_l_range_change_from_shade")
    for patch_name in patch_names:
        fields.extend(
            [
                f"{patch_name}_opponent_radius_mean",
                f"{patch_name}_opponent_theta_deg_circular_mean",
                f"{patch_name}_opponent_theta_resultant_length",
                f"{patch_name}_opponent_radius_change_from_shade",
                f"{patch_name}_opponent_theta_difference_from_shade_deg",
            ]
        )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            conditions = result["conditions"]
            row = {field: result.get(field, conditions.get(field, "")) for field in fields}
            patch_deltas = result.get("patch_delta_lab_from_shade", {})
            for patch_name in patch_names:
                row[f"{patch_name}_delta_lab_from_shade"] = patch_deltas.get(
                    patch_name, ""
                )
                opponent_polar = result["patches"][patch_name]["opponent_polar"]
                row[f"{patch_name}_opponent_radius_mean"] = opponent_polar[
                    "radius_mean"
                ]
                row[f"{patch_name}_opponent_theta_deg_circular_mean"] = opponent_polar[
                    "theta_deg_circular_mean"
                ]
                row[f"{patch_name}_opponent_theta_resultant_length"] = opponent_polar[
                    "theta_resultant_length"
                ]
                polar_differences = result.get(
                    "patch_opponent_polar_difference_from_shade", {}
                ).get(patch_name, {})
                row[f"{patch_name}_opponent_radius_change_from_shade"] = (
                    polar_differences.get("radius_change", "")
                )
                row[f"{patch_name}_opponent_theta_difference_from_shade_deg"] = (
                    polar_differences.get("theta_difference_deg", "")
                )
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="カラーチャートの指定色だけをROI選択して計測します。"
    )
    parser.add_argument(
        "--patches",
        nargs="+",
        metavar="COLOR",
        help=(
            "調査する色（white gray black red green blue）。"
            "空白区切りまたはカンマ区切りで指定。省略時は全色。"
        ),
    )
    args = parser.parse_args()
    try:
        patch_names = parse_patch_names(args.patches)
    except ValueError as error:
        parser.error(str(error))

    print(f"調査対象の色: {', '.join(patch_names)}")
    video_paths = sorted(Path().glob(VIDEO_GLOB))
    if not video_paths:
        print(f"動画が見つかりません: {VIDEO_GLOB}", file=sys.stderr)
        return 1
    if REPRESENTATIVE_FRAME_INDEX < 0:
        print("REPRESENTATIVE_FRAME_INDEX は 0 以上を指定してください。", file=sys.stderr)
        return 1

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    roi_path = OUTPUT_DIRECTORY / f"{ROI_PROFILE_NAME}_rois.json"
    # RESELECT_ROIS=True でも既存ROIを読み込み、初期表示に利用する。
    roi_data = load_roi_data(roi_path)
    # 旧版の共通ROI定義があれば、新しい動画ごとのROI定義へ切り替える。
    if roi_data.get("roi_mode") != "per_video_color_chart":
        roi_data = new_roi_data()

    results: list[dict[str, Any]] = []
    for index, video_path in enumerate(video_paths, start=1):
        print(f"[{index}/{len(video_paths)}] 解析中: {video_path.name}")
        try:
            representative_frame = read_representative_frame(
                video_path, REPRESENTATIVE_FRAME_INDEX
            )
            video_rois = get_video_rois(
                roi_data, video_path, representative_frame, patch_names
            )
            atomic_json_dump(roi_path, roi_data)
            result = analyze_video(video_path, video_rois, patch_names)
            result["conditions"] = parse_filename_conditions(video_path)
            results.append(result)
            detail_path = OUTPUT_DIRECTORY / f"{video_path.stem}_metrics.json"
            atomic_json_dump(detail_path, result)
        except Exception as error:
            print(f"  スキップ: {error}", file=sys.stderr)

    if not results:
        print("解析できた動画がありません。", file=sys.stderr)
        return 1
    atomic_json_dump(roi_path, roi_data)
    add_reference_differences(results, patch_names)
    for result in results:
        detail_path = OUTPUT_DIRECTORY / f"{Path(result['video_name']).stem}_metrics.json"
        atomic_json_dump(detail_path, result)
    atomic_json_dump(OUTPUT_DIRECTORY / "sensor_quality_summary.json", {"results": results})
    write_summary_csv(
        results, OUTPUT_DIRECTORY / "sensor_quality_summary.csv", patch_names
    )
    print(f"完了: {OUTPUT_DIRECTORY / 'sensor_quality_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
