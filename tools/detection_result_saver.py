"""HumanDetector のフレームごとの検出結果を保存するための補助クラス。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class DetectionResultSaver:
    """JSONL・CSV・実行メタデータを1実行分のフォルダへ保存する。"""

    CSV_FIELDNAMES = (
        "frame_index",
        "source_timestamp_s",
        "processing_time_s",
        "processing_fps",
        "detection_index",
        "track_id",
        "status",
        "is_target_frame",
        "is_match",
        "is_number1",
        "yolo_confidence",
        "similarity",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "bbox_area",
    )

    def __init__(
        self,
        output_root: str | Path,
        source_type: str,
        source_path: str | None,
        config_path: str | Path,
        config: dict[str, Any],
    ) -> None:
        source_name = self._source_name(source_type, source_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_directory = self._create_run_directory(
            Path(output_root), f"{source_name}_{timestamp}"
        )

        self.jsonl_path = self.run_directory / "detections.jsonl"
        self.csv_path = self.run_directory / "detections.csv"
        self.metadata_path = self.run_directory / "run_metadata.json"
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8", newline="\n")
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDNAMES)
        self._csv_writer.writeheader()
        self._frames_written = 0
        self._detections_written = 0
        self._closed = False

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_type": source_type,
            "source_path": source_path or "",
            "config_path": str(config_path),
            "config": self._json_safe(config),
            "files": {
                "frame_results_jsonl": self.jsonl_path.name,
                "detections_csv": self.csv_path.name,
            },
        }
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"検出結果の保存先: {self.run_directory}")

    @staticmethod
    def _source_name(source_type: str, source_path: str | None) -> str:
        if source_path:
            name = Path(source_path).stem
            if name:
                return name
        return source_type or "input"

    @staticmethod
    def _create_run_directory(output_root: Path, base_name: str) -> Path:
        """同じ秒に再実行しても既存結果を上書きしない保存先を作る。"""
        output_root.mkdir(parents=True, exist_ok=True)
        for index in range(1, 10_000):
            suffix = "" if index == 1 else f"_{index:02d}"
            candidate = output_root / f"{base_name}{suffix}"
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("検出結果の保存フォルダを作成できませんでした。")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): DetectionResultSaver._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [DetectionResultSaver._json_safe(item) for item in value]
        return value

    def write_frame(
        self,
        frame_index: int,
        source_timestamp_s: float | None,
        processing_time_s: float,
        detections: list[dict[str, Any]],
    ) -> None:
        """1フレーム分の判定をJSONLとCSVへ追記する。"""
        if self._closed:
            raise RuntimeError("閉じた検出結果保存先には書き込めません。")

        processing_fps = 1.0 / max(processing_time_s, 1e-9)
        frame_result = {
            "frame_index": frame_index,
            "source_timestamp_s": source_timestamp_s,
            "processing_time_s": processing_time_s,
            "processing_fps": processing_fps,
            "detections": detections,
        }
        self._jsonl_file.write(json.dumps(frame_result, ensure_ascii=False) + "\n")

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            self._csv_writer.writerow(
                {
                    "frame_index": frame_index,
                    "source_timestamp_s": source_timestamp_s,
                    "processing_time_s": processing_time_s,
                    "processing_fps": processing_fps,
                    "detection_index": detection["detection_index"],
                    "track_id": detection.get("track_id"),
                    "status": detection["status"],
                    "is_target_frame": detection.get("is_target_frame"),
                    "is_match": detection["is_match"],
                    "is_number1": detection["is_number1"],
                    "yolo_confidence": detection["yolo_confidence"],
                    "similarity": detection.get("similarity"),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": detection["width"],
                    "bbox_height": detection["height"],
                    "bbox_area": detection["area"],
                }
            )

        self._frames_written += 1
        self._detections_written += len(detections)

    def close(self) -> None:
        """ファイルを閉じ、最終件数をメタデータへ追記する。"""
        if self._closed:
            return
        self._jsonl_file.close()
        self._csv_file.close()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["completed_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["frames_written"] = self._frames_written
        metadata["detections_written"] = self._detections_written
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._closed = True
