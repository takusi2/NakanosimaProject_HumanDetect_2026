from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.coarse_to_fine_mannequin.src.pipeline import CoarseToFineMatcher
from experiments.coarse_to_fine_mannequin.src.artifacts import ArtifactWriter


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

        self.assertIsNotNone(result.best_match)
        self.assertEqual((result.best_match.x, result.best_match.y), (48, 40))
        self.assertTrue(result.detected)
        self.assertEqual(result.coarse_image_bgr.shape[:2], (20, 24))
        self.assertEqual(len(result.rois), 1)

    def test_rejects_uniform_wall_by_variance(self) -> None:
        rng = np.random.default_rng(456)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        wall_frame = np.full((80, 96, 3), 230, dtype=np.uint8)
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
            variance_log_distance_max=3.0,
        )

        result = matcher.process(wall_frame)

        self.assertFalse(result.detected)
        self.assertIsNone(result.best_match)
        self.assertEqual(len(result.detail_variance_filters), 1)
        variance_filter = result.detail_variance_filters[0]
        self.assertEqual(variance_filter.passed_count, 0)
        self.assertEqual(variance_filter.rejected_count, variance_filter.candidate_count)

    def test_saves_variance_filter_visualisation(self) -> None:
        rng = np.random.default_rng(789)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        wall_frame = np.full((80, 96, 3), 230, dtype=np.uint8)
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
            variance_log_distance_max=3.0,
        )
        result = matcher.process(wall_frame)

        with tempfile.TemporaryDirectory() as temporary_directory:
            writer = ArtifactWriter(Path(temporary_directory))
            writer.save_frame(1, wall_frame, result)
            writer.close()

            run_directory = next(Path(temporary_directory).glob("run_*"))
            filter_images = list(
                (run_directory / "05_detail_match").glob("*_variance_filter.png")
            )
            self.assertEqual(len(filter_images), 1)
            self.assertGreater(filter_images[0].stat().st_size, 0)
            score_data = json.loads(
                (run_directory / "06_scores" / "frame_000001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(score_data["variance_filters"][0]["candidates_passed"], 0)
            self.assertGreater(score_data["variance_filters"][0]["candidates_rejected"], 0)


if __name__ == "__main__":
    unittest.main()
