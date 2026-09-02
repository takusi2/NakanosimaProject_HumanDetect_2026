"""点正解データと HumanDetector の保存結果から精度を算出する。

このファイル先頭の利用者設定を確認してから実行する。

    python tools/evaluate_detection_results.py

出力:
    annotations/evaluation_results/<動画名>__<検出実行名>/
      - evaluation_summary.json : 集計値・設定・入力元
      - frame_evaluation.csv    : フレームごとの判定結果
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


# ============================================================
# 利用者設定：通常はここだけを書き換える
# ============================================================
GROUND_TRUTH_PATH = Path("./annotations/ground_truth/2026-0826-1602-46_center_surround/point_ground_truth.json")
DETECTION_RESULTS_ROOT = Path("./annotations/detection_results")
# None の場合は、正解データの動画名に対応する最新の検出結果フォルダを自動選択する。
DETECTION_RESULT_DIRECTORY: Path | None = None
OUTPUT_ROOT = Path("./annotations/evaluation_results")

EVALUATION_VERSION = 1
FINAL_STATUS = "MATCH"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def point_in_bbox(point_xy: list[int] | tuple[int, int], bbox_xyxy: list[int]) -> bool:
    """境界上もBBoxに含まれるものとして、対象点が枠内か判定する。"""
    x, y = point_xy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def latest_detection_directory(
    root: Path, ground_truth_video_path: str
) -> Path:
    """正解データと同名の入力動画に対する最新の検出結果を見つける。"""
    expected_stem = Path(ground_truth_video_path).stem
    candidates: list[Path] = []
    for directory in root.iterdir() if root.exists() else []:
        metadata_path = directory / "run_metadata.json"
        if not directory.is_dir() or not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if Path(metadata.get("source_path", "")).stem == expected_stem:
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            f"動画 '{expected_stem}' に対応する検出結果が見つかりません: {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_detection_frames(detection_directory: Path) -> dict[int, dict[str, Any]]:
    jsonl_path = detection_directory / "detections.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"検出結果JSONLがありません: {jsonl_path}")
    frames: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        frame = json.loads(line)
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int) or frame_index <= 0:
            raise ValueError(f"{jsonl_path}:{line_number} の frame_index が不正です。")
        if frame_index in frames:
            raise ValueError(f"フレーム {frame_index} が重複しています: {jsonl_path}")
        frames[frame_index] = frame
    return frames


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def frame_timestamp(
    frame_index: int, detection_frame: dict[str, Any] | None, video_fps: float
) -> float:
    if detection_frame is not None:
        timestamp = finite_number(detection_frame.get("source_timestamp_s"))
        if timestamp is not None:
            return timestamp
    return (frame_index - 1) / video_fps if video_fps > 0 else float(frame_index - 1)


def longest_streak(values: list[bool], desired: bool) -> int:
    best = 0
    current = 0
    for value in values:
        if value == desired:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_processing(frame_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    fps_values = [
        value
        for row in frame_rows
        if (value := finite_number(row.get("processing_fps"))) is not None and value > 0
    ]
    time_values = [
        value
        for row in frame_rows
        if (value := finite_number(row.get("processing_time_s"))) is not None and value >= 0
    ]
    return {
        "frames_with_fps": len(fps_values),
        "mean_fps": statistics.mean(fps_values) if fps_values else None,
        "median_fps": statistics.median(fps_values) if fps_values else None,
        "min_fps": min(fps_values) if fps_values else None,
        "mean_processing_time_s": statistics.mean(time_values) if time_values else None,
    }


def evaluate(
    ground_truth: dict[str, Any],
    detection_frames: dict[int, dict[str, Any]],
    detection_width: int | None = None,
    detection_height: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """単一target点の正解データに対する各種指標とフレーム別結果を返す。"""
    if ground_truth.get("annotation_type") != "single_target_point":
        raise ValueError("single_target_point 形式の正解データだけを評価できます。")
    if ground_truth.get("frame_index_base") != 1:
        raise ValueError("フレーム番号が1始まりの正解データだけを評価できます。")

    video = ground_truth["video"]
    frame_count = int(video["frame_count"])
    ground_truth_width = int(video["width"])
    ground_truth_height = int(video["height"])
    video_fps = float(video.get("fps", 0.0))
    annotations = ground_truth.get("frames", {})
    detection_width = detection_width or ground_truth_width
    detection_height = detection_height or ground_truth_height
    x_scale = detection_width / ground_truth_width
    y_scale = detection_height / ground_truth_height

    tp = fp = fn = 0
    target_frames = 0
    no_target_frames = 0
    ignored_frames = 0
    yolo_covered = reid_covered = maybe_covered = match_covered = 0
    evaluation_detection_frames: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    episodes: list[list[tuple[int, bool, float]]] = []
    current_episode: list[tuple[int, bool, float]] = []

    for frame_index in range(1, frame_count + 1):
        annotation = annotations.get(str(frame_index))
        if annotation is None:
            raise ValueError(f"フレーム {frame_index} に正解データがありません。")
        state = annotation.get("state")
        detection_frame = detection_frames.get(frame_index)
        detections = detection_frame.get("detections", []) if detection_frame else []
        match_detections = [d for d in detections if d.get("status") == FINAL_STATUS]
        timestamp = frame_timestamp(frame_index, detection_frame, video_fps)

        row: dict[str, Any] = {
            "frame_index": frame_index,
            "ground_truth_state": state,
            "target_x": None,
            "target_y": None,
            "evaluation_target_x": None,
            "evaluation_target_y": None,
            "yolo_contains_target": None,
            "reid_input_contains_target": None,
            "maybe_contains_target": None,
            "match_contains_target": None,
            "match_prediction_count": len(match_detections),
            "tp_count": 0,
            "fp_count": 0,
            "fn_count": 0,
            "source_timestamp_s": timestamp,
            "processing_time_s": detection_frame.get("processing_time_s") if detection_frame else None,
            "processing_fps": detection_frame.get("processing_fps") if detection_frame else None,
        }

        if state == "ignore":
            ignored_frames += 1
            if current_episode:
                episodes.append(current_episode)
                current_episode = []
            frame_rows.append(row)
            continue

        if state == "no_target":
            no_target_frames += 1
            row["fp_count"] = len(match_detections)
            fp += row["fp_count"]
            if current_episode:
                episodes.append(current_episode)
                current_episode = []
            evaluation_detection_frames.append(detection_frame or {})
            frame_rows.append(row)
            continue

        if state != "target":
            raise ValueError(f"フレーム {frame_index} の正解状態が不正です: {state}")
        point = annotation.get("target_point_xy")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"フレーム {frame_index} の target_point_xy が不正です。")

        target_frames += 1
        row["target_x"], row["target_y"] = point
        # HumanDetectorはconfigのframe_width / frame_heightへリサイズしてから
        # BBoxを保存するため、正解点も同じ座標系へ変換して照合する。
        evaluation_point = [point[0] * x_scale, point[1] * y_scale]
        row["evaluation_target_x"], row["evaluation_target_y"] = evaluation_point
        containing = [
            detection
            for detection in detections
            if isinstance(detection.get("bbox_xyxy"), list)
            and len(detection["bbox_xyxy"]) == 4
            and point_in_bbox(evaluation_point, detection["bbox_xyxy"])
        ]
        yolo_hit = bool(containing)
        reid_hit = any(detection.get("status") != "SKIP" for detection in containing)
        maybe_hit = any(
            detection.get("status") in {"MAYBE", FINAL_STATUS} for detection in containing
        )
        correct_matches = [
            detection for detection in match_detections if detection in containing
        ]
        match_hit = bool(correct_matches)

        row["yolo_contains_target"] = yolo_hit
        row["reid_input_contains_target"] = reid_hit
        row["maybe_contains_target"] = maybe_hit
        row["match_contains_target"] = match_hit
        if yolo_hit:
            yolo_covered += 1
        if reid_hit:
            reid_covered += 1
        if maybe_hit:
            maybe_covered += 1
        if match_hit:
            match_covered += 1
            tp += 1
            row["tp_count"] = 1
        else:
            fn += 1
            row["fn_count"] = 1

        # 対象点を含むMATCHは最大1件だけTP。重複MATCHと対象外MATCHはFP。
        row["fp_count"] = len(match_detections) - row["tp_count"]
        fp += row["fp_count"]
        current_episode.append((frame_index, match_hit, timestamp))
        evaluation_detection_frames.append(detection_frame or {})
        frame_rows.append(row)

    if current_episode:
        episodes.append(current_episode)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    episode_summaries: list[dict[str, Any]] = []
    for episode in episodes:
        first_frame, _, start_time = episode[0]
        correct_frames = [item for item in episode if item[1]]
        first_correct = correct_frames[0] if correct_frames else None
        values = [item[1] for item in episode]
        episode_summaries.append(
            {
                "start_frame": first_frame,
                "end_frame": episode[-1][0],
                "target_frames": len(episode),
                "correct_match_frames": sum(values),
                "longest_miss_streak_frames": longest_streak(values, False),
                "longest_match_streak_frames": longest_streak(values, True),
                "first_correct_match_frame": first_correct[0] if first_correct else None,
                "first_correct_match_latency_frames": (
                    first_correct[0] - first_frame if first_correct else None
                ),
                "first_correct_match_latency_s": (
                    first_correct[2] - start_time if first_correct else None
                ),
            }
        )

    first_episode_with_match = next(
        (episode for episode in episode_summaries if episode["first_correct_match_frame"] is not None),
        None,
    )
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_scope": {
            "final_status": FINAL_STATUS,
            "matching_rule": "MATCH BBoxがtarget点を含む場合にTP。1 target点につきTPは最大1件。",
            "ignored_frames_excluded": True,
            "ground_truth_coordinate_size": [ground_truth_width, ground_truth_height],
            "detection_coordinate_size": [detection_width, detection_height],
            "ground_truth_to_detection_scale": [x_scale, y_scale],
        },
        "frame_counts": {
            "video_frames": frame_count,
            "target_frames": target_frames,
            "no_target_frames": no_target_frames,
            "ignored_frames": ignored_frames,
            "evaluated_frames": target_frames + no_target_frames,
            "detection_result_frames_present": sum(
                1 for row in frame_rows if row["frame_index"] in detection_frames
            ),
        },
        "final_match": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score(precision, recall),
        },
        "target_stage_recall": {
            "denominator_target_frames": target_frames,
            "yolo_person_bbox": ratio(yolo_covered, target_frames),
            "reid_input_not_skip": ratio(reid_covered, target_frames),
            "maybe_or_match": ratio(maybe_covered, target_frames),
            "final_match": ratio(match_covered, target_frames),
            "covered_frames": {
                "yolo_person_bbox": yolo_covered,
                "reid_input_not_skip": reid_covered,
                "maybe_or_match": maybe_covered,
                "final_match": match_covered,
            },
        },
        "continuity": {
            "target_episodes": episode_summaries,
            "longest_miss_streak_frames": max(
                (episode["longest_miss_streak_frames"] for episode in episode_summaries),
                default=0,
            ),
            "longest_match_streak_frames": max(
                (episode["longest_match_streak_frames"] for episode in episode_summaries),
                default=0,
            ),
            "first_correct_match": first_episode_with_match,
        },
        "processing": summarize_processing(evaluation_detection_frames),
    }
    return summary, frame_rows


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "ground_truth_state",
        "target_x",
        "target_y",
        "evaluation_target_x",
        "evaluation_target_y",
        "yolo_contains_target",
        "reid_input_contains_target",
        "maybe_contains_target",
        "match_contains_target",
        "match_prediction_count",
        "tp_count",
        "fp_count",
        "fn_count",
        "source_timestamp_s",
        "processing_time_s",
        "processing_fps",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ground_truth = read_json(GROUND_TRUTH_PATH)
    detection_directory = (
        DETECTION_RESULT_DIRECTORY
        if DETECTION_RESULT_DIRECTORY is not None
        else latest_detection_directory(
            DETECTION_RESULTS_ROOT, ground_truth["video"]["path"]
        )
    )
    detection_frames = load_detection_frames(detection_directory)
    detection_metadata = read_json(detection_directory / "run_metadata.json")
    detection_config = detection_metadata.get("config", {})
    detection_width = detection_config.get("frame_width")
    detection_height = detection_config.get("frame_height")
    if not isinstance(detection_width, int) or not isinstance(detection_height, int):
        detection_width = detection_height = None
    summary, frame_rows = evaluate(
        ground_truth, detection_frames, detection_width, detection_height
    )

    output_directory = OUTPUT_ROOT / f"{Path(ground_truth['video']['path']).stem}__{detection_directory.name}"
    output_directory.mkdir(parents=True, exist_ok=True)
    summary["inputs"] = {
        "ground_truth_path": str(GROUND_TRUTH_PATH),
        "detection_result_directory": str(detection_directory),
    }
    write_json(output_directory / "evaluation_summary.json", summary)
    write_frame_csv(output_directory / "frame_evaluation.csv", frame_rows)
    print(f"評価結果の保存先: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
