"""YOLO候補を補助に、動画の人物正解データを作成するツール。

使い方:
    このファイル先頭の「利用者設定」を編集してから、次を実行する。
    python tools/create_ground_truth.py

YOLO の検出結果は proposal（下書き）として ground_truth.json と分けて保存する。
候補にない人物は BBox を手動で追加できるため、YOLO の見逃しも正解データに含められる。

操作:
    1 / 2 / 0 : 選択中の領域を target / other / ignore にする
    B           : BBox追加モード（マウスでドラッグして追加）
    D           : 選択中のYOLO候補を誤検出として除外する
    Delete      : 選択中の正解領域を削除する
    N / P       : 次 / 前のフレームへ移動
    Space       : 現フレームを確認済みにして次へ移動
    V           : YOLO候補の表示を切り替える（見逃し確認用）
    Z           : 直前の編集を元に戻す
    S           : 保存
    Q / Esc     : 保存して終了
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PERSON_CLASS_ID = 0
FORMAT_VERSION = "ground-truth-v1"
PROPOSAL_FORMAT_VERSION = "yolo-proposals-v1"
LABEL_COLORS = {
    "target": (0, 200, 0),
    "other": (0, 165, 255),
    "ignore": (128, 128, 128),
    "unlabeled": (0, 255, 255),
}
PROPOSAL_COLOR = (255, 180, 0)
LABEL_DISPLAY_NAMES = {
    "target": "対象",
    "other": "対象外",
    "ignore": "除外",
    "unlabeled": "未分類",
}
JAPANESE_FONT_PATHS = (
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\msgothic.ttc"),
)


# ============================================================
# 利用者設定：動画やモデルを変えるときは、ここだけを書き換える
# ============================================================
VIDEO_PATH = "./output/2-6-hiroto.mp4"
YOLO_MODEL_PATH = "yolo11x.pt"
DETECTION_MODE = "bbox"  # "bbox" または "seg"
YOLO_CONFIDENCE = 0.25
OUTPUT_DIRECTORY: str | None = None  # Noneなら annotations/<動画名> に保存
PRECOMPUTE_ALL_FRAMES = False  # Trueで開始前に全フレームのYOLO候補を生成


@dataclass(frozen=True)
class AnnotationConfig:
    video: str
    model: str
    mode: str
    conf: float
    output_dir: str | None
    precompute: bool


def load_config() -> AnnotationConfig:
    if DETECTION_MODE not in {"bbox", "seg"}:
        raise ValueError('DETECTION_MODE は "bbox" または "seg" を指定してください。')
    if not 0.0 <= YOLO_CONFIDENCE <= 1.0:
        raise ValueError("YOLO_CONFIDENCE は 0.0〜1.0 の範囲で指定してください。")
    return AnnotationConfig(
        video=VIDEO_PATH,
        model=YOLO_MODEL_PATH,
        mode=DETECTION_MODE,
        conf=YOLO_CONFIDENCE,
        output_dir=OUTPUT_DIRECTORY,
        precompute=PRECOMPUTE_ALL_FRAMES,
    )


def atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    """途中終了してもJSONを壊しにくいよう、一時ファイル経由で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def read_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clipped_xywh(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[float] | None:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def xywh_to_xyxy(box: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


class GroundTruthCreator:
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.label_font = self.load_japanese_font(17)
        self.status_font = self.load_japanese_font(18)
        self.video_path = Path(config.video).resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(f"動画が見つかりません: {self.video_path}")

        self.output_dir = (
            Path(config.output_dir).resolve()
            if config.output_dir
            else Path("annotations") / self.video_path.stem
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_path = self.output_dir / "ground_truth.json"
        self.proposals_path = self.output_dir / "proposals.json"
        self.progress_path = self.output_dir / "progress.json"

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {self.video_path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise RuntimeError("動画のサイズまたはフレーム数を取得できませんでした。")

        self.model = YOLO(config.model)
        self.ground_truth = read_json_or_default(
            self.ground_truth_path,
            {
                "format": FORMAT_VERSION,
                "video": {
                    "path": str(self.video_path),
                    "sha256": sha256_of_file(self.video_path),
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "frame_count": self.frame_count,
                },
                "labels": ["target", "other", "ignore"],
                "frames": {},
            },
        )
        self.proposals = read_json_or_default(
            self.proposals_path,
            {
                "format": PROPOSAL_FORMAT_VERSION,
                "video": str(self.video_path),
                "model": str(config.model),
                "mode": config.mode,
                "confidence_threshold": config.conf,
                "frames": {},
            },
        )
        self.progress = read_json_or_default(self.progress_path, {"current_frame": 0})
        self.current_frame_index = min(
            max(int(self.progress.get("current_frame", 0)), 0), self.frame_count - 1
        )
        self.current_frame: np.ndarray | None = None
        self.selected: tuple[str, int] | None = None  # ("proposal" | "gt", index)
        self.show_proposals = True
        self.draw_mode = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.undo_stack: list[tuple[int, dict[str, Any]]] = []
        self.window_name = "Create Ground Truth"

    @staticmethod
    def load_japanese_font(size: int) -> ImageFont.FreeTypeFont:
        """Windows標準の日本語フォントを明示指定し、文字化けを防ぐ。"""
        for font_path in JAPANESE_FONT_PATHS:
            if font_path.is_file():
                return ImageFont.truetype(str(font_path), size=size)
        raise RuntimeError(
            "日本語表示用フォントが見つかりません。Yu Gothic または Meiryo を導入してください。"
        )

    def frame_key(self, frame_index: int | None = None) -> str:
        return str(self.current_frame_index if frame_index is None else frame_index)

    def frame_data(self, frame_index: int | None = None) -> dict[str, Any]:
        key = self.frame_key(frame_index)
        return self.ground_truth["frames"].setdefault(
            key,
            {"verified": False, "instances": [], "rejected_proposals": []},
        )

    def save(self) -> None:
        self.progress["current_frame"] = self.current_frame_index
        self.progress["updated_at_unix"] = time.time()
        atomic_json_dump(self.ground_truth_path, self.ground_truth)
        atomic_json_dump(self.proposals_path, self.proposals)
        atomic_json_dump(self.progress_path, self.progress)

    def read_frame(self, frame_index: int) -> np.ndarray:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"フレーム {frame_index} を読み込めませんでした。")
        return frame

    def proposals_for_current_frame(self) -> list[dict[str, Any]]:
        key = self.frame_key()
        cached = self.proposals["frames"].get(key)
        if cached is not None:
            return cached["detections"]
        if self.current_frame is None:
            raise RuntimeError("フレームが読み込まれていません。")

        results = self.model.predict(
            self.current_frame,
            conf=self.config.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        result = results[0] if results else None
        detections: list[dict[str, Any]] = []
        if result is not None and result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            polygons = None
            if self.config.mode == "seg" and getattr(result, "masks", None) is not None:
                polygons = result.masks.xy
            for index, (box, score) in enumerate(zip(boxes, scores)):
                bbox = clipped_xywh(*box, self.width, self.height)
                if bbox is None:
                    continue
                detection: dict[str, Any] = {
                    "id": len(detections),
                    "bbox": bbox,
                    "score": round(float(score), 6),
                }
                if polygons is not None and index < len(polygons):
                    polygon = np.asarray(polygons[index], dtype=np.float32)
                    if len(polygon) >= 3:
                        detection["segmentation"] = np.round(polygon, 2).tolist()
                detections.append(detection)

        self.proposals["frames"][key] = {"detections": detections}
        return detections

    def precompute_proposals(self) -> None:
        print(f"YOLO候補を全 {self.frame_count} フレームで生成します。")
        original_frame = self.current_frame_index
        for frame_index in range(self.frame_count):
            if self.frame_key(frame_index) in self.proposals["frames"]:
                continue
            self.current_frame_index = frame_index
            self.current_frame = self.read_frame(frame_index)
            self.proposals_for_current_frame()
            if (frame_index + 1) % 50 == 0 or frame_index + 1 == self.frame_count:
                print(f"  {frame_index + 1}/{self.frame_count}")
                self.save()
        self.current_frame_index = original_frame
        self.current_frame = self.read_frame(self.current_frame_index)
        self.save()

    def snapshot_current_frame(self) -> None:
        self.undo_stack.append(
            (self.current_frame_index, copy.deepcopy(self.frame_data()))
        )
        self.undo_stack = self.undo_stack[-50:]

    def undo(self) -> None:
        if not self.undo_stack:
            return
        frame_index, previous = self.undo_stack.pop()
        self.ground_truth["frames"][self.frame_key(frame_index)] = previous
        self.current_frame_index = frame_index
        self.load_current_frame()
        self.save()

    def is_proposal_used(self, proposal_index: int) -> bool:
        for instance in self.frame_data()["instances"]:
            source = instance.get("source", {})
            if source.get("proposal_index") == proposal_index:
                return True
        return False

    def visible_proposal_indices(self) -> list[int]:
        frame_data = self.frame_data()
        rejected = set(frame_data.get("rejected_proposals", []))
        return [
            index
            for index, _ in enumerate(self.proposals_for_current_frame())
            if index not in rejected and not self.is_proposal_used(index)
        ]

    def add_instance(
        self,
        bbox: list[float],
        source: dict[str, Any],
        label: str = "unlabeled",
    ) -> int:
        self.snapshot_current_frame()
        instance = {
            "id": f"gt_{uuid.uuid4().hex[:10]}",
            "label": label,
            "bbox": bbox,
            "source": source,
        }
        self.frame_data()["instances"].append(instance)
        return len(self.frame_data()["instances"]) - 1

    def label_selected(self, label: str) -> None:
        if self.selected is None:
            return
        kind, index = self.selected
        if kind == "gt":
            instances = self.frame_data()["instances"]
            if 0 <= index < len(instances):
                self.snapshot_current_frame()
                instances[index]["label"] = label
        elif kind == "proposal":
            proposals = self.proposals_for_current_frame()
            if 0 <= index < len(proposals):
                gt_index = self.add_instance(
                    proposals[index]["bbox"],
                    source={"type": "yolo_proposal", "proposal_index": index},
                    label=label,
                )
                self.selected = ("gt", gt_index)
        self.save()

    def reject_selected_proposal(self) -> None:
        if self.selected is None or self.selected[0] != "proposal":
            return
        index = self.selected[1]
        self.snapshot_current_frame()
        rejected = self.frame_data().setdefault("rejected_proposals", [])
        if index not in rejected:
            rejected.append(index)
        self.selected = None
        self.save()

    def delete_selected_instance(self) -> None:
        if self.selected is None or self.selected[0] != "gt":
            return
        index = self.selected[1]
        instances = self.frame_data()["instances"]
        if 0 <= index < len(instances):
            self.snapshot_current_frame()
            instances.pop(index)
        self.selected = None
        self.save()

    def go_to_frame(self, target_frame_index: int) -> None:
        self.current_frame_index = min(max(target_frame_index, 0), self.frame_count - 1)
        self.selected = None
        self.draw_mode = False
        self.drag_start = None
        self.drag_end = None
        self.load_current_frame()
        self.save()

    def load_current_frame(self) -> None:
        self.current_frame = self.read_frame(self.current_frame_index)
        self.proposals_for_current_frame()

    @staticmethod
    def contains(box: list[float], x: int, y: int) -> bool:
        bx, by, bw, bh = box
        return bx <= x <= bx + bw and by <= y <= by + bh

    def select_at(self, x: int, y: int) -> None:
        instances = self.frame_data()["instances"]
        for index in reversed(range(len(instances))):
            if self.contains(instances[index]["bbox"], x, y):
                self.selected = ("gt", index)
                return
        for index in reversed(self.visible_proposal_indices()):
            proposal = self.proposals_for_current_frame()[index]
            if self.contains(proposal["bbox"], x, y):
                self.selected = ("proposal", index)
                return
        self.selected = None

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if self.draw_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
                self.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
                self.drag_end = (x, y)
                x1, y1 = self.drag_start
                x2, y2 = self.drag_end
                bbox = clipped_xywh(
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self.width, self.height
                )
                if bbox is not None:
                    gt_index = self.add_instance(bbox, source={"type": "manual"})
                    self.selected = ("gt", gt_index)
                    self.save()
                self.draw_mode = False
                self.drag_start = None
                self.drag_end = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.select_at(x, y)

    def draw_box(
        self,
        canvas: np.ndarray,
        box: list[float],
        color: tuple[int, int, int],
        text: str,
        text_items: list[tuple[str, int, int, tuple[int, int, int], ImageFont.FreeTypeFont]],
        selected: bool = False,
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = xywh_to_xyxy(box)
        height, width = canvas.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return
        if selected:
            color, thickness = (255, 255, 255), 3
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA
        )
        # 上端の領域が画面外・ステータス欄と重なる場合は、ラベルを枠の下へ表示する。
        text_y = y1 - 21 if y1 >= 24 else min(height - 21, y2 + 3)
        text_items.append((text, x1 + 2, text_y, color, self.label_font))

    @staticmethod
    def draw_unicode_text(
        canvas: np.ndarray,
        text_items: list[tuple[str, int, int, tuple[int, int, int], ImageFont.FreeTypeFont]],
    ) -> np.ndarray:
        """Pillowで文字を描画し、OpenCVの日本語文字化けを回避する。"""
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil_image)
        for text, x, y, bgr_color, font in text_items:
            rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
            drawer.text(
                (x, y),
                text,
                font=font,
                fill=rgb_color,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def render(self) -> np.ndarray:
        if self.current_frame is None:
            raise RuntimeError("フレームが読み込まれていません。")
        canvas = self.current_frame.copy()
        text_items: list[tuple[str, int, int, tuple[int, int, int], ImageFont.FreeTypeFont]] = []
        state = self.frame_data()
        status = (
            f"フレーム {self.current_frame_index + 1}/{self.frame_count}  "
            f"確認済み: {'はい' if state['verified'] else 'いいえ'}  "
            f"候補表示: {'オン' if self.show_proposals else 'オフ'}  "
            f"操作: {'BBox追加' if self.draw_mode else '選択'}"
        )
        # ステータス欄を先に描く。後からBBoxを描画するため、枠線が隠れない。
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 900), 28), (0, 0, 0), -1)
        text_items.append((status, 8, 4, (255, 255, 255), self.status_font))
        if self.show_proposals:
            for index in self.visible_proposal_indices():
                proposal = self.proposals_for_current_frame()[index]
                polygon = proposal.get("segmentation")
                if polygon:
                    points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [points], True, PROPOSAL_COLOR, 1, lineType=cv2.LINE_AA)
                self.draw_box(
                    canvas,
                    proposal["bbox"],
                    PROPOSAL_COLOR,
                    f"候補 {index}  信頼度 {proposal['score']:.2f}",
                    text_items,
                    selected=self.selected == ("proposal", index),
                    thickness=1,
                )

        for index, instance in enumerate(self.frame_data()["instances"]):
            label = instance.get("label", "unlabeled")
            self.draw_box(
                canvas,
                instance["bbox"],
                LABEL_COLORS.get(label, LABEL_COLORS["unlabeled"]),
                f"{LABEL_DISPLAY_NAMES.get(label, '未分類')} ({instance['id']})",
                text_items,
                selected=self.selected == ("gt", index),
            )

        if self.drag_start is not None and self.drag_end is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                LABEL_COLORS["unlabeled"],
                2,
                lineType=cv2.LINE_AA,
            )

        return self.draw_unicode_text(canvas, text_items)

    def run(self) -> None:
        if self.config.precompute:
            self.precompute_proposals()
        self.load_current_frame()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        print(__doc__)

        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("1"), ord("2"), ord("0")):
                self.label_selected({ord("1"): "target", ord("2"): "other", ord("0"): "ignore"}[key])
            elif key in (ord("b"), ord("B")):
                self.draw_mode = not self.draw_mode
                self.drag_start = None
                self.drag_end = None
            elif key in (ord("d"), ord("D")):
                self.reject_selected_proposal()
            elif key in (8, 46, 3014656):  # Backspace / Delete (environment dependent)
                self.delete_selected_instance()
            elif key in (ord("n"), ord("N"), 83, 2555904):  # N / right arrow
                self.go_to_frame(self.current_frame_index + 1)
            elif key in (ord("p"), ord("P"), 81, 2424832):  # P / left arrow
                self.go_to_frame(self.current_frame_index - 1)
            elif key == ord(" "):
                self.snapshot_current_frame()
                self.frame_data()["verified"] = True
                self.save()
                self.go_to_frame(self.current_frame_index + 1)
            elif key in (ord("v"), ord("V")):
                self.show_proposals = not self.show_proposals
            elif key in (ord("z"), ord("Z")):
                self.undo()
            elif key in (ord("s"), ord("S")):
                self.save()

        self.save()
        self.capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    try:
        GroundTruthCreator(load_config()).run()
    except Exception as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
