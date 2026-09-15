from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml

from coarse_to_fine_human_detector import CoarseToFineHumanDetector


class CoarseToFineHumanDetectorTests(unittest.TestCase):
    def _make_detector(self, directory: Path, *, final_max_error: float) -> tuple[CoarseToFineHumanDetector, np.ndarray]:
        assets = directory / "assets"
        assets.mkdir()
        rng = np.random.default_rng(123)
        template = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
        template_path = assets / "template.png"
        self.assertTrue(cv2.imwrite(str(template_path), template))
        config = {
            "input_source": "video",
            "video_path": "assets/input.mp4",
            "camera_index": 0,
            "frame_width": 96,
            "frame_height": 100,
            "window_name": "test detector",
            "template_path": "assets/template.png",
            "device": "cpu",
            "frame_downscale": 0.25,
            "coarse_template_scales": [0.25],
            "detail_template_scales": [1.0],
            "coarse_stride": 1,
            "detail_stride_base": 1,
            "detail_stride_min": 1,
            "coarse_top_k": 1,
            "coarse_candidates_per_template": 20,
            "nms_distance_original_px": 10,
            "roi_margin_px": 8,
            "final_max_error": final_max_error,
            "variance_log_distance_max": 100.0,
            "min_chroma_variance": 0.0,
            "min_brightness_variance": 0.0,
        }
        config_path = directory / "detector.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return CoarseToFineHumanDetector(str(config_path)), template

    def test_match_returns_center_class_and_annotated_frame_without_artifact_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, template = self._make_detector(Path(temporary_directory), final_max_error=0.1)
            frame = np.zeros((100, 96, 3), dtype=np.uint8)
            frame[60:76, 48:60] = template

            with mock.patch.object(detector.matcher, "process", wraps=detector.matcher.process) as process:
                output, center_x, class_name = detector.process_frame(frame)

            process.assert_called_once_with(
                frame,
                collect_coarse_image=False,
                collect_detail_variance_filters=False,
            )
            self.assertEqual(center_x, 54.0)
            self.assertEqual(class_name, "A")
            self.assertEqual(output.shape, frame.shape)
            self.assertFalse(np.array_equal(output, frame))
            self.assertEqual(Path(detector.conf.template_path), Path(temporary_directory) / "assets" / "template.png")

    def test_no_match_returns_none_and_draws_best_candidate_in_red(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector, template = self._make_detector(Path(temporary_directory), final_max_error=-0.1)
            frame = np.zeros((100, 96, 3), dtype=np.uint8)
            frame[60:76, 48:60] = template

            output, center_x, class_name = detector.process_frame(frame)

            self.assertIsNone(center_x)
            self.assertIsNone(class_name)
            self.assertTrue(np.any(np.all(output == (0, 0, 255), axis=2)))


if __name__ == "__main__":
    unittest.main()
