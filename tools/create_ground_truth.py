"""単一マネキンの中心点を動画へ付与し、正解データを作成する。

使い方:
    1. このファイル先頭の VIDEO_PATH を対象動画に変更する。
    2. python tools/create_ground_truth.py を実行する。

操作:
    左クリック                   : 対象マネキンの胴体中心点を保存し、次フレームへ
    左ボタンを長押し             : 押している間、対象点を連続保存してフレームを進める
    Enter                       : 対象なし (no_target) を保存し、次フレームへ
    I                           : 評価除外 (ignore) を保存し、次フレームへ
    B / Backspace               : 前フレームへ
    R                           : 現在フレームの記録を削除
    Q / Esc                     : 保存して終了

この基本版は、常に1フレームずつ進み、対象マネキンは1体だけを扱う。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2


# ============================================================
# 利用者設定：通常はここだけを書き換える
# ============================================================
VIDEO_PATH = Path("./output/2026-0819-1713-38_center_surround.mp4")
OUTPUT_ROOT = Path("./annotations/ground_truth")
RESUME_EXISTING_ANNOTATION = True

# この時間を超えて押し続けると連続保存を開始する（秒）。
LONG_PRESS_SECONDS = 0.25
# 長押し中の保存間隔（秒）。カーソルを対象の胴体中心に合わせ続けて使う。
CONTINUOUS_SAVE_INTERVAL_SECONDS = 0.05
# 表示ウィンドウの最大幅。元動画の座標へ自動変換して保存する。
DISPLAY_MAX_WIDTH = 1280
WINDOW_NAME = "Point Ground Truth"

SCHEMA_VERSION = 1
FRAME_INDEX_BASE = 1  # HumanDetector の detections.jsonl と揃える。


def create_document(
    video_path: Path, width: int, height: int, fps: float, frame_count: int
) -> dict[str, Any]:
    """1本の動画に対応する空の点アノテーション文書を作る。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_type": "single_target_point",
        "frame_index_base": FRAME_INDEX_BASE,
        "video": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "frames": {},
        "completed": False,
    }


def set_frame_annotation(
    document: dict[str, Any],
    frame_index: int,
    state: str,
    point_xy: tuple[int, int] | None = None,
) -> None:
    """1フレームを target / no_target / ignore のいずれかで確定する。"""
    if state not in {"target", "no_target", "ignore"}:
        raise ValueError(f"不正な状態です: {state}")
    if state == "target" and point_xy is None:
        raise ValueError("target には中心点が必要です。")
    if state != "target" and point_xy is not None:
        raise ValueError(f"{state} に中心点は指定できません。")

    annotation: dict[str, Any] = {"state": state}
    if point_xy is not None:
        annotation["target_point_xy"] = [int(point_xy[0]), int(point_xy[1])]
    document["frames"][str(frame_index)] = annotation


def save_document(output_path: Path, document: dict[str, Any]) -> None:
    """途中終了でも壊れたJSONを残さないよう、一時ファイル経由で保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)


class PointGroundTruthAnnotator:
    def __init__(self, video_path: Path, output_path: Path) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {video_path}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise RuntimeError("動画のサイズまたはフレーム数を取得できませんでした。")

        self.scale = min(1.0, DISPLAY_MAX_WIDTH / self.width)
        self.document = self._load_or_create_document()
        self.current_frame_index = self._first_unlabeled_frame()
        self.current_frame = None
        self.is_mouse_down = False
        self.mouse_down_time = 0.0
        self.last_continuous_save_time = 0.0
        self.continuous_save_active = False
        self.held_point: tuple[int, int] | None = None
        self.preview_point: tuple[int, int] | None = None

    def _load_or_create_document(self) -> dict[str, Any]:
        if self.output_path.exists() and RESUME_EXISTING_ANNOTATION:
            document = json.loads(self.output_path.read_text(encoding="utf-8"))
            video = document.get("video", {})
            expected = (self.width, self.height, self.frame_count)
            actual = (
                video.get("width"),
                video.get("height"),
                video.get("frame_count"),
            )
            if actual != expected:
                raise RuntimeError(
                    "既存の正解データが別の動画のものです。"
                    "出力フォルダを変更するか、既存JSONを移動してください。"
                )
            print(f"既存の正解データを再開します: {self.output_path}")
            return document
        return create_document(
            self.video_path, self.width, self.height, self.fps, self.frame_count
        )

    def _first_unlabeled_frame(self) -> int:
        for frame_index in range(FRAME_INDEX_BASE, self.frame_count + FRAME_INDEX_BASE):
            if str(frame_index) not in self.document["frames"]:
                return frame_index
        return self.frame_count + FRAME_INDEX_BASE - 1

    def _read_frame(self, frame_index: int):
        raw_index = frame_index - FRAME_INDEX_BASE
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, raw_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"フレームを読み込めませんでした: {frame_index}")
        return frame

    def _display_to_original(self, x: int, y: int) -> tuple[int, int]:
        original_x = int(round(x / self.scale))
        original_y = int(round(y / self.scale))
        return (
            min(max(original_x, 0), self.width - 1),
            min(max(original_y, 0), self.height - 1),
        )

    def _render(self) -> None:
        self.current_frame = self._read_frame(self.current_frame_index)
        display = self.current_frame.copy()
        current = self.document["frames"].get(str(self.current_frame_index))

        if current is not None:
            state = current["state"]
            if state == "target":
                x, y = current["target_point_xy"]
                cv2.drawMarker(
                    display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2
                )
                status = "TARGET (saved)"
            elif state == "no_target":
                status = "NO TARGET (saved)"
            else:
                status = "IGNORE (saved)"
        else:
            status = "UNLABELED"

        if self.preview_point is not None:
            cv2.drawMarker(
                display, self.preview_point, (0, 255, 255), cv2.MARKER_CROSS, 18, 2
            )

        overlay = f"Frame {self.current_frame_index}/{self.frame_count}  {status}"
        controls = "Click: target | Hold left: continuous | Enter: no target | I: ignore | B: back | R: clear | Q: quit"
        cv2.rectangle(display, (0, 0), (min(display.shape[1], 1200), 54), (0, 0, 0), -1)
        cv2.putText(display, overlay, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, controls, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

        if self.scale < 1.0:
            display = cv2.resize(
                display,
                (int(self.width * self.scale), int(self.height * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(WINDOW_NAME, display)

    def _save_and_next(self, state: str, point_xy: tuple[int, int] | None = None) -> None:
        set_frame_annotation(self.document, self.current_frame_index, state, point_xy)
        save_document(self.output_path, self.document)
        self.preview_point = None
        if self.current_frame_index < self.frame_count:
            self.current_frame_index += 1
        self._render()

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        point = self._display_to_original(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_mouse_down = True
            self.mouse_down_time = time.monotonic()
            self.last_continuous_save_time = 0.0
            self.continuous_save_active = False
            self.held_point = point
            self.preview_point = point
            self._render()
        elif event == cv2.EVENT_MOUSEMOVE and self.is_mouse_down:
            self.held_point = point
            self.preview_point = point
            self._render()
        elif event == cv2.EVENT_LBUTTONUP and self.is_mouse_down:
            self.is_mouse_down = False
            self.held_point = None
            if self.continuous_save_active:
                self.continuous_save_active = False
                self.preview_point = None
                self._render()
            else:
                # 短いクリックは、離した位置を対象点として1回だけ保存する。
                self._save_and_next("target", point)

    def _save_continuously_while_held(self) -> None:
        """長押し中は、設定した間隔で対象点を保存して1フレームずつ進める。"""
        if not self.is_mouse_down or self.held_point is None:
            return
        now = time.monotonic()
        if not self.continuous_save_active:
            if now - self.mouse_down_time < LONG_PRESS_SECONDS:
                return
            self.continuous_save_active = True
            self.last_continuous_save_time = 0.0
        if now - self.last_continuous_save_time < CONTINUOUS_SAVE_INTERVAL_SECONDS:
            return

        self.last_continuous_save_time = now
        self.preview_point = self.held_point
        self._save_and_next("target", self.held_point)

    def _clear_current_annotation(self) -> None:
        self.document["frames"].pop(str(self.current_frame_index), None)
        self.document["completed"] = False
        save_document(self.output_path, self.document)
        self._render()

    def run(self) -> None:
        print("正解データ保存先:", self.output_path)
        print("対象マネキンの胴体中心をクリックしてください。長押し中は連続保存します。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._render()
        try:
            while True:
                key = cv2.waitKey(20) & 0xFF
                self._save_continuously_while_held()
                if key in (ord("q"), 27):
                    break
                if key in (13, 10):
                    self._save_and_next("no_target")
                elif key in (ord("i"), ord("I")):
                    self._save_and_next("ignore")
                elif key in (ord("b"), ord("B"), 8):
                    self.current_frame_index = max(FRAME_INDEX_BASE, self.current_frame_index - 1)
                    self.preview_point = None
                    self._render()
                elif key in (ord("r"), ord("R")):
                    self._clear_current_annotation()
        finally:
            self.document["completed"] = len(self.document["frames"]) == self.frame_count
            save_document(self.output_path, self.document)
            self.capture.release()
            cv2.destroyAllWindows()


def main() -> int:
    output_path = OUTPUT_ROOT / VIDEO_PATH.stem / "point_ground_truth.json"
    annotator = PointGroundTruthAnnotator(VIDEO_PATH, output_path)
    annotator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
