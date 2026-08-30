"""A-1色評価のJSONから、反対色評価グラフをSVGで出力する。

このファイル先頭の「利用者設定」を編集してから実行する。

    python tools/plot_sensor_quality.py

出力するグラフ:
  1. 日陰基準からの色相差 theta_difference_deg（測定結果に含まれるパッチ）
  2. 日陰基準からの色成分強さの変化 radius_change（同上）
  3. 色相の信頼性 theta_resultant_length（同上）

theta_resultant_length はROI内の色相のまとまり度で、1に近いほど
ROI内の画素が同じ色相方向に揃う。小さい場合は、thetaの代表値や
日陰との差を強く解釈しない。
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


# ============================================================
# 利用者設定：通常はここだけを書き換える
# ============================================================
INPUT_SUMMARY_JSON = Path("./analysis/sensor_quality_a1/sensor_quality_summary.json")
OUTPUT_DIRECTORY = Path("./analysis/sensor_quality_a1/plots")

# パッチは測定結果JSONの ``patches`` から自動抽出する。
# すべての結果に共通して存在するパッチだけをグラフ化するため、結果によって
# 選択したカラーチャートの色数・色名が異なっても扱える。
PATCH_LABELS = {
    "white": "白",
    "gray": "灰",
    "black": "黒",
    "red": "赤",
    "green": "緑",
    "blue": "青",
}

# 日陰との差を描く照明条件。shade-noneは差が常に0なので、差分グラフには含めない。
DIFFERENCE_LIGHT_CONDITIONS = ("sun-back", "sun-front")
# 信頼性グラフには日陰も含め、照明条件ごとにRを比較する。
RELIABILITY_LIGHT_CONDITIONS = ("shade-none", "sun-back", "sun-front")

PATCH_COLORS = {
    "white": "#805ad5",
    "gray": "#718096",
    "black": "#1a202c",
    "red": "#c53030",
    "green": "#27824a",
    "blue": "#2563b8",
    "yellow": "#b7791f",
    "cyan": "#008b8b",
    "magenta": "#b83280",
}
FALLBACK_PATCH_COLORS = ("#9f7aea", "#dd6b20", "#319795", "#d53f8c")
LIGHT_COLORS = {
    "shade-none": "#4a5568",
    "sun-back": "#2b6cb0",
    "sun-front": "#dd6b20",
}
LIGHT_LABELS = {"shade-none": "日陰", "sun-back": "直射（背面）", "sun-front": "直射（正面）"}

SVG_WIDTH = 1200
SVG_HEIGHT = 680
FONT_FAMILY = "Yu Gothic, Meiryo, sans-serif"


def numeric_sort_key(value: str) -> tuple[int, float | str]:
    """gS04などを数字順に並べる。"""
    match = re.search(r"\d+(?:\.\d+)?", value)
    return (0, float(match.group(0))) if match else (1, value)


def safe_slug(value: str) -> str:
    """出力ファイル名に使えるASCII文字列へ変換する。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def select_x_axis_field(results: list[dict[str, Any]]) -> str:
    """変化させたガウシアン値を、グラフ横軸として自動選択する。"""
    center_values = {
        result.get("conditions", {}).get("center_gaussian")
        for result in results
        if result.get("conditions", {}).get("center_gaussian")
    }
    surround_values = {
        result.get("conditions", {}).get("surround_gaussian")
        for result in results
        if result.get("conditions", {}).get("surround_gaussian")
    }
    if len(center_values) > 1 and len(surround_values) <= 1:
        return "center_gaussian"
    if len(surround_values) > 1 and len(center_values) <= 1:
        return "surround_gaussian"
    if len(center_values) <= 1 and len(surround_values) <= 1:
        raise ValueError("横軸にするgCまたはgSの変化がありません。")
    raise ValueError(
        "gCとgSの両方が同時に変化しています。1次元グラフにするため、"
        "どちらか一方を固定した結果を入力してください。"
    )


def experiment_key(result: dict[str, Any], x_axis_field: str) -> tuple[str, ...]:
    """横軸値と照明を除く、同一実験系列を識別するキー。"""
    conditions = result["conditions"]
    return tuple(
        conditions.get(key, "unknown")
        for key in (
            "lens",
            *(key for key in ("center_gaussian", "surround_gaussian") if key != x_axis_field),
            "iir",
            "naka_rushton",
            "distance",
            "pose",
        )
    )


def experiment_label(key: tuple[str, ...], x_axis_field: str) -> str:
    labels = ["lens"]
    if x_axis_field != "center_gaussian":
        labels.append("center_gaussian")
    if x_axis_field != "surround_gaussian":
        labels.append("surround_gaussian")
    labels.extend(("iir", "naka_rushton", "distance", "pose"))
    values = dict(zip(labels, key, strict=True))
    parts = [values["lens"]]
    if "center_gaussian" in values:
        parts.append(f"gC{values['center_gaussian']}")
    if "surround_gaussian" in values:
        parts.append(f"gS{values['surround_gaussian']}")
    parts.extend(values[name] for name in ("iir", "naka_rushton", "distance", "pose"))
    return " / ".join(parts)


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"結果JSONが見つかりません: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"結果が空です: {path}")
    return results


def select_plot_patches(results: list[dict[str, Any]]) -> tuple[str, ...]:
    """すべての結果に共通して記録されたパッチ名を、グラフ対象として返す。"""
    patch_sets = [set(result.get("patches", {})) for result in results]
    common_patches = set.intersection(*patch_sets) if patch_sets else set()
    if not common_patches:
        raise ValueError("全結果に共通するパッチがありません。測定結果のpatchesを確認してください。")
    known_order = tuple(PATCH_LABELS)
    return tuple(
        sorted(
            common_patches,
            key=lambda name: (known_order.index(name) if name in known_order else len(known_order), name),
        )
    )


def patch_labels(patch_names: tuple[str, ...]) -> dict[str, str]:
    """既知色は日本語名、任意のパッチ名はそのまま凡例に使う。"""
    return {name: PATCH_LABELS.get(name, name) for name in patch_names}


def patch_colors(patch_names: tuple[str, ...]) -> dict[str, str]:
    """既知色は専用色、任意のパッチ名には見分けやすい代替色を割り当てる。"""
    return {
        name: PATCH_COLORS.get(name, FALLBACK_PATCH_COLORS[index % len(FALLBACK_PATCH_COLORS)])
        for index, name in enumerate(patch_names)
    }


def select_groups(
    results: list[dict[str, Any]], x_axis_field: str
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        conditions = result.get("conditions", {})
        if not conditions.get(x_axis_field):
            continue
        groups[experiment_key(result, x_axis_field)].append(result)
    if not groups:
        raise ValueError("横軸にするガウシアン値を含む評価結果がありません。")
    return dict(groups)


def svg_text(x: float, y: float, value: str, size: int, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{weight}" '
        f'fill="#1a202c">{html.escape(value)}</text>'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" />'
    )


def value_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def padded_domain(values: list[float], include_zero: bool = True) -> tuple[float, float]:
    if include_zero:
        values = [*values, 0.0]
    if not values:
        return (-1.0, 1.0)
    lower, upper = min(values), max(values)
    if math.isclose(lower, upper):
        padding = max(1.0, abs(lower) * 0.1)
    else:
        padding = (upper - lower) * 0.12
    return lower - padding, upper + padding


def value_to_y(value: float, lower: float, upper: float, top: float, bottom: float) -> float:
    return bottom - (value - lower) / (upper - lower) * (bottom - top)


def x_positions(values: list[str], left: float, right: float) -> dict[str, float]:
    if len(values) == 1:
        return {values[0]: (left + right) / 2}
    step = (right - left) / (len(values) - 1)
    return {value: left + index * step for index, value in enumerate(values)}


def build_series(
    results: list[dict[str, Any]],
    light_condition: str,
    patch_name: str,
    accessor: Callable[[dict[str, Any]], float | None],
    x_axis_field: str,
) -> dict[str, float]:
    """gSごとの値を返す。同一条件の重複があれば最初の1件を使う。"""
    series: dict[str, float] = {}
    for result in results:
        conditions = result["conditions"]
        if conditions.get("light_condition") != light_condition:
            continue
        value = accessor(result)
        if value is not None:
            series.setdefault(conditions[x_axis_field], value)
    return series


def draw_multi_panel_chart(
    title: str,
    subtitle: str,
    panels: list[tuple[str, dict[str, dict[str, float]]]],
    x_values: list[str],
    y_domain: tuple[float, float],
    y_label: str,
    series_colors: dict[str, str],
    series_labels: dict[str, str],
    output_path: Path,
    x_axis_label: str,
    x_tick_prefix: str,
    baseline_zero: bool = False,
) -> None:
    """同じy軸を持つ複数パネルの折れ線グラフをSVGへ保存する。"""
    panel_count = len(panels)
    outer_left, outer_right = 86.0, 42.0
    title_bottom, bottom = 108.0, 78.0
    gap = 44.0
    panel_width = (SVG_WIDTH - outer_left - outer_right - gap * (panel_count - 1)) / panel_count
    chart_top, chart_bottom = 155.0, SVG_HEIGHT - bottom
    lower, upper = y_domain
    # R のように 0〜1 を扱うグラフは、小数目盛りを残す。
    # 整数へ丸めると 0.2/0.4 が 0、0.6/0.8 が 1 と表示されてしまう。
    tick_format = ".1f" if upper - lower <= 1.0 else ".0f"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(42, 46, title, 28, weight="500"),
        svg_text(42, 76, subtitle, 16),
    ]

    legend_x = SVG_WIDTH - 42.0
    for index, key in enumerate(series_labels):
        position_x = legend_x - index * 170.0
        parts.append(svg_line(position_x - 18, 103, position_x + 4, 103, series_colors[key], 3))
        parts.append(svg_text(position_x + 10, 108, series_labels[key], 14))

    for panel_index, (panel_title, panel_series) in enumerate(panels):
        left = outer_left + panel_index * (panel_width + gap)
        right = left + panel_width
        positions = x_positions(x_values, left + 24, right - 16)
        parts.append(svg_text((left + right) / 2, 135, panel_title, 18, anchor="middle", weight="500"))
        parts.append(f'<rect x="{left:.1f}" y="{chart_top:.1f}" width="{panel_width:.1f}" height="{chart_bottom - chart_top:.1f}" fill="none" stroke="#a0aec0" stroke-width="1"/>')

        for tick_index in range(6):
            tick_value = lower + (upper - lower) * tick_index / 5
            tick_y = value_to_y(tick_value, lower, upper, chart_top, chart_bottom)
            parts.append(svg_line(left, tick_y, right, tick_y, "#e2e8f0"))
            if panel_index == 0:
                parts.append(svg_text(left - 10, tick_y + 5, format(tick_value, tick_format), 13, anchor="end"))
            elif panel_index == panel_count - 1:
                # 左右パネルで同一の縦軸範囲を使うことを、右側の数値でも確認できるようにする。
                parts.append(svg_text(right + 10, tick_y + 5, format(tick_value, tick_format), 13))
        if baseline_zero and lower <= 0 <= upper:
            zero_y = value_to_y(0.0, lower, upper, chart_top, chart_bottom)
            parts.append(svg_line(left, zero_y, right, zero_y, "#4a5568", 1.5))

        for x_value, position_x in positions.items():
            parts.append(svg_line(position_x, chart_bottom, position_x, chart_bottom + 5, "#4a5568"))
            parts.append(svg_text(position_x, chart_bottom + 26, f"{x_tick_prefix}{x_value}", 13, anchor="middle"))

        for series_name, values_by_gs in panel_series.items():
            points = [
                (positions[gs], value_to_y(value, lower, upper, chart_top, chart_bottom))
                for gs, value in values_by_gs.items()
                if gs in positions
            ]
            if not points:
                continue
            if len(points) > 1:
                parts.append(
                    '<polyline fill="none" stroke="{}" stroke-width="3" points="{}"/>'.format(
                        series_colors[series_name],
                        " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
                    )
                )
            for point_x, point_y in points:
                parts.append(f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="5" fill="{series_colors[series_name]}"/>')

    # 縦書きではなく、左側の縦軸の直上に横書きで置く。細い左余白で文字が欠けないようにする。
    parts.append(svg_text(outer_left, chart_top - 8, y_label, 15, weight="500"))
    parts.append(svg_text(SVG_WIDTH / 2, SVG_HEIGHT - 24, x_axis_label, 15, anchor="middle"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_plot_data_csv(
    results: list[dict[str, Any]], path: Path, patch_names: tuple[str, ...]
) -> None:
    fields = [
        "video_name",
        "light_condition",
        "center_gaussian",
        "surround_gaussian",
        "patch",
        "theta_difference_deg",
        "radius_change",
        "theta_resultant_length",
        "reference_video",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            conditions = result["conditions"]
            for patch_name in patch_names:
                polar = result["patches"][patch_name]["opponent_polar"]
                difference = result.get("patch_opponent_polar_difference_from_shade", {}).get(
                    patch_name, {}
                )
                writer.writerow(
                    {
                        "video_name": result["video_name"],
                        "light_condition": conditions.get("light_condition", ""),
                        "center_gaussian": conditions.get("center_gaussian", ""),
                        "surround_gaussian": conditions.get("surround_gaussian", ""),
                        "patch": patch_name,
                        "theta_difference_deg": difference.get("theta_difference_deg", ""),
                        "radius_change": difference.get("radius_change", ""),
                        "theta_resultant_length": polar.get("theta_resultant_length", ""),
                        "reference_video": result.get("reference_video", ""),
                    }
                )


def create_graphs_for_group(
    key: tuple[str, ...], results: list[dict[str, Any]], x_axis_field: str
) -> list[Path]:
    group_directory = OUTPUT_DIRECTORY / safe_slug(experiment_label(key, x_axis_field))
    group_directory.mkdir(parents=True, exist_ok=True)
    gaussian_values = sorted(
        {result["conditions"][x_axis_field] for result in results},
        key=numeric_sort_key,
    )
    x_tick_prefix = "gC" if x_axis_field == "center_gaussian" else "gS"
    x_axis_label = (
        "center Gaussian value"
        if x_axis_field == "center_gaussian"
        else "surround Gaussian value"
    )
    patch_names = select_plot_patches(results)
    selected_patch_labels = patch_labels(patch_names)
    selected_patch_colors = patch_colors(patch_names)
    patch_description = "・".join(selected_patch_labels[name] for name in patch_names)
    outputs: list[Path] = []

    theta_accessor = lambda result, patch: value_or_none(
        result.get("patch_opponent_polar_difference_from_shade", {})
        .get(patch, {})
        .get("theta_difference_deg")
    )
    theta_panels = [
        (
            LIGHT_LABELS[light],
            {
                patch: build_series(
                    results, light, patch, lambda r, p=patch: theta_accessor(r, p), x_axis_field
                )
                for patch in patch_names
            },
        )
        for light in DIFFERENCE_LIGHT_CONDITIONS
    ]
    theta_output = group_directory / "theta_difference_from_shade.svg"
    draw_multi_panel_chart(
        "日陰基準からの色相差",
        f"0°に近いほど、日陰と同じ色味の方向を保てている。対象パッチ: {patch_description}",
        theta_panels,
        gaussian_values,
        (0.0, 180.0),
        "色相差 [deg]",
        selected_patch_colors,
        selected_patch_labels,
        theta_output,
        x_axis_label,
        x_tick_prefix,
    )
    outputs.append(theta_output)

    radius_accessor = lambda result, patch: value_or_none(
        result.get("patch_opponent_polar_difference_from_shade", {})
        .get(patch, {})
        .get("radius_change")
    )
    radius_panels = [
        (
            LIGHT_LABELS[light],
            {
                patch: build_series(
                    results, light, patch, lambda r, p=patch: radius_accessor(r, p), x_axis_field
                )
                for patch in patch_names
            },
        )
        for light in DIFFERENCE_LIGHT_CONDITIONS
    ]
    radius_values = [
        value
        for _, panel in radius_panels
        for values_by_gs in panel.values()
        for value in values_by_gs.values()
    ]
    radius_output = group_directory / "radius_change_from_shade.svg"
    draw_multi_panel_chart(
        "日陰基準からの色成分強さの変化",
        "0に近いほど、日陰と比べた色成分の強さの変化が小さい。",
        radius_panels,
        gaussian_values,
        padded_domain(radius_values),
        "色成分の強さの変化（radius change）",
        selected_patch_colors,
        selected_patch_labels,
        radius_output,
        x_axis_label,
        x_tick_prefix,
        baseline_zero=True,
    )
    outputs.append(radius_output)

    reliability_accessor = lambda result, patch: value_or_none(
        result["patches"][patch]["opponent_polar"].get("theta_resultant_length")
    )
    reliability_panels = [
        (
            f"{selected_patch_labels[patch]}パッチ",
            {
                light: build_series(
                    results, light, patch, lambda r, p=patch: reliability_accessor(r, p), x_axis_field
                )
                for light in RELIABILITY_LIGHT_CONDITIONS
            },
        )
        for patch in patch_names
    ]
    reliability_output = group_directory / "theta_resultant_length.svg"
    draw_multi_panel_chart(
        "反対色の信頼性：色相のまとまり度 R",
        "1に近いほどROI内の色相が揃い、代表色相θと色相差を信頼しやすい。",
        reliability_panels,
        gaussian_values,
        (0.0, 1.0),
        "R (0–1)",
        LIGHT_COLORS,
        {light: LIGHT_LABELS[light] for light in RELIABILITY_LIGHT_CONDITIONS},
        reliability_output,
        x_axis_label,
        x_tick_prefix,
    )
    outputs.append(reliability_output)

    data_output = group_directory / "plot_data_used.csv"
    write_plot_data_csv(results, data_output, patch_names)
    outputs.append(data_output)
    return outputs


def main() -> int:
    try:
        results = read_results(INPUT_SUMMARY_JSON)
        x_axis_field = select_x_axis_field(results)
        groups = select_groups(results, x_axis_field)
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        all_outputs: list[str] = []
        for key, group_results in groups.items():
            print(f"グラフ作成中: {experiment_label(key, x_axis_field)}")
            all_outputs.extend(
                str(path)
                for path in create_graphs_for_group(key, group_results, x_axis_field)
            )
        with (OUTPUT_DIRECTORY / "plot_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"input": str(INPUT_SUMMARY_JSON), "x_axis_field": x_axis_field, "outputs": all_outputs},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"完了: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
