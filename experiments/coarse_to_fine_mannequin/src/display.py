"""検出結果をOpenCVウィンドウへ表示するための描画処理。"""

from __future__ import annotations

import cv2
import numpy as np

from .types import FrameResult


def draw_display_frame(
    frame_number: int,
    frame: np.ndarray,
    result: FrameResult,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    fps: float | None = None,
) -> np.ndarray:
    """検出枠と状態・スコア・FPSを描いた表示用フレームを返す。"""
    image = frame.copy()
    status = result.temporal_status or ("MATCH" if result.detected else "NO MATCH")
    colour = (
        (0, 255, 0)
        if status == "MATCH"
        else ((0, 165, 255) if status == "MAYBE" else (0, 0, 255))
    )
    best = result.best_match
    if best is not None:
        cv2.rectangle(
            image,
            (round(best.x * scale_x), round(best.y * scale_y)),
            (
                round((best.x + best.template.width) * scale_x),
                round((best.y + best.template.height) * scale_y),
            ),
            colour,
            2,
        )
        score_text = f"score={best.score:.2f} scale={best.template.scale:.2f}"
    else:
        score_text = "score=N/A scale=N/A"
    if result.track_id is not None:
        score_text += (
            f" track={result.track_id} "
            f"history={result.positive_frames}/{result.history_window_frames}"
        )

    status_text = f"{frame_number}: {status}"
    hud_rows: list[tuple[str, int]] = [(status_text, 28), (score_text, 52)]
    if fps is not None:
        hud_rows.append((f"FPS: {fps:.1f}", 76))
    for text, y in hud_rows:
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            image,
            (5, y - height - baseline - 3),
            (width + 11, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA
        )
    return image
