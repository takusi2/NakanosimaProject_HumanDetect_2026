"""表示用フレームの描画が元フレームを変更しないことを確認する。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

from src.display import draw_display_frame
from src.types import FrameResult


class DisplayTests(unittest.TestCase):
    def test_draws_match_box_without_modifying_input_frame(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        best_match = SimpleNamespace(
            x=100,
            y=90,
            score=1.0,
            template=SimpleNamespace(width=12, height=16, scale=1.0),
        )
        result = FrameResult(
            coarse_image_bgr=None,
            coarse_candidates=[],
            rois=[],
            detail_matches=[],
            detail_variance_filters=[],
            best_match=best_match,
            detected=True,
            temporal_status="MATCH",
        )

        displayed = draw_display_frame(1, frame, result, fps=30.0)

        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))
        self.assertFalse(np.array_equal(displayed, frame))
        self.assertTrue(np.array_equal(displayed[90, 100], np.array([0, 255, 0])))


if __name__ == "__main__":
    unittest.main()
