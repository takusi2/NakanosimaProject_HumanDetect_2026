"""粗探索・詳細照合の生検出を時系列で確定する追跡器。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .types import FrameResult

if TYPE_CHECKING:
    from .settings import TrackingSettings


@dataclass
class TemporalTrack:
    """同じ位置に連続して現れた生検出の履歴。"""

    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    target_history: list[bool]
    last_frame_idx: int


@dataclass(frozen=True)
class TrackingDecision:
    """現在フレームに対する時系列確定結果。"""

    status: str
    confirmed: bool
    track_id: int | None
    positive_frames: int
    history_window_frames: int


class TemporalMatchTracker:
    """IoUで同じ位置を対応付け、直近履歴からMATCH/MAYBEを決める。"""

    def __init__(
        self,
        settings: TrackingSettings | None = None,
        *,
        window_frames: int | None = None,
        confirm_frames: int | None = None,
        match_iou_thresh: float | None = None,
        max_track_age: int | None = None,
    ) -> None:
        """型付き設定から追跡器を作る。"""
        if settings is None:
            # 既存の直接引数呼出しを、テストと過去の利用コードのために維持する。
            from .settings import TrackingSettings

            legacy_config = {
                "target_confirm_window_frames": 10 if window_frames is None else window_frames,
                "target_confirm_frames": 5 if confirm_frames is None else confirm_frames,
                "match_iou_thresh": 0.8 if match_iou_thresh is None else match_iou_thresh,
                "max_track_age": 30 if max_track_age is None else max_track_age,
            }
            settings = TrackingSettings(legacy_config)
        elif any(value is not None for value in (window_frames, confirm_frames, match_iou_thresh, max_track_age)):
            raise TypeError("settings と個別の設定値を同時に指定することはできません")
        window_frames = settings.window_frames
        confirm_frames = settings.confirm_frames
        match_iou_thresh = settings.match_iou_thresh
        max_track_age = settings.max_track_age
        if window_frames <= 0 or confirm_frames <= 0 or confirm_frames > window_frames:
            raise ValueError("confirm_frames must be in 1..window_frames")
        if not 0.0 <= match_iou_thresh <= 1.0:
            raise ValueError("match_iou_thresh must be in the range [0, 1]")
        if max_track_age < 0:
            raise ValueError("max_track_age must be non-negative")
        self.window_frames = window_frames
        self.confirm_frames = confirm_frames
        self.match_iou_thresh = match_iou_thresh
        self.max_track_age = max_track_age
        self.track_info: list[TemporalTrack] = []
        self.next_track_id = 0
        self.frame_idx = 0

    @staticmethod
    def _iou_xyxy(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        return intersection / max(first_area + second_area - intersection, 1e-6)

    def _append_history(self, track: TemporalTrack, is_target_frame: bool) -> None:
        track.target_history.append(is_target_frame)
        track.target_history = track.target_history[-self.window_frames :]

    def update(self, bbox_xyxy: tuple[int, int, int, int] | None) -> TrackingDecision:
        """現在の生検出位置を追加し、MATCH/MAYBE/NO MATCHを返す。"""
        self.frame_idx += 1
        updated_index: int | None = None
        updated_track: TemporalTrack | None = None
        if bbox_xyxy is not None:
            best_iou, best_index = 0.0, -1
            for index, track in enumerate(self.track_info):
                iou = self._iou_xyxy(bbox_xyxy, track.bbox_xyxy)
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_iou >= self.match_iou_thresh:
                updated_track = self.track_info[best_index]
                updated_track.bbox_xyxy = bbox_xyxy
                self._append_history(updated_track, True)
                updated_track.last_frame_idx = self.frame_idx
                updated_index = best_index
            else:
                updated_track = TemporalTrack(
                    track_id=self.next_track_id,
                    bbox_xyxy=bbox_xyxy,
                    target_history=[True],
                    last_frame_idx=self.frame_idx,
                )
                self.next_track_id += 1
                self.track_info.append(updated_track)
                updated_index = len(self.track_info) - 1

        for index, track in enumerate(self.track_info):
            if index != updated_index:
                self._append_history(track, False)
        self.track_info[:] = [
            track
            for track in self.track_info
            if self.frame_idx - track.last_frame_idx <= self.max_track_age
        ]

        if updated_track is None:
            return TrackingDecision("NO MATCH", False, None, 0, self.window_frames)
        positive_frames = sum(updated_track.target_history)
        confirmed = positive_frames >= self.confirm_frames
        return TrackingDecision(
            "MATCH" if confirmed else "MAYBE",
            confirmed,
            updated_track.track_id,
            positive_frames,
            self.window_frames,
        )


def apply_temporal_tracking(
    raw_result: FrameResult, tracker: TemporalMatchTracker
) -> FrameResult:
    """生の照合結果を履歴で MATCH / MAYBE / NO MATCH に確定する。"""
    best = raw_result.best_match
    raw_match = raw_result.detected and best is not None
    bbox_xyxy = (
        (best.x, best.y, best.x + best.template.width, best.y + best.template.height)
        if raw_match and best is not None
        else None
    )
    decision = tracker.update(bbox_xyxy)
    return replace(
        raw_result,
        detected=raw_match and decision.confirmed,
        temporal_status=decision.status,
        track_id=decision.track_id,
        positive_frames=decision.positive_frames,
        history_window_frames=decision.history_window_frames,
    )
