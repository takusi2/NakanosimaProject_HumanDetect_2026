from __future__ import annotations

import unittest

import numpy as np

from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher


class CoarseToFineMatcherTests(unittest.TestCase):
    def test_finds_template_after_coarse_roi_selection(self) -> None:
        rng = np.random.default_rng(123)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        frame = np.zeros((80, 96, 3), dtype=np.uint8)
        frame[40:56, 48:60] = template
        matcher = CoarseToFineMatcher(
            template,
            device="cpu",
            frame_downscale=0.25,
            coarse_template_scales=[0.25],
            detail_template_scales=[1.0],
            coarse_stride=1,
            detail_stride=1,
            coarse_top_k=1,
            nms_distance_original_px=10,
            roi_margin_px=8,
            final_max_error=0.1,
        )

        result = matcher.process(frame)

        self.assertEqual((result.best_match.x, result.best_match.y), (48, 40))
        self.assertTrue(result.detected)
        self.assertEqual(result.coarse_image_bgr.shape[:2], (20, 24))
        self.assertEqual(len(result.rois), 1)


if __name__ == "__main__":
    unittest.main()
