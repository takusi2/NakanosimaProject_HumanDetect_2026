"""粗探索・詳細照合の生検出を時系列で確定する追跡器。"""

from __future__ import annotations

from dataclasses import dataclass


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
        *,
        window_frames: int = 10,
        confirm_frames: int = 5,
        match_iou_thresh: float = 0.8,
        max_track_age: int = 30,
    ) -> None:
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
