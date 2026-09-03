from __future__ import annotations

import unittest

import numpy as np

from experiments.opponent_color_mannequin.src.matcher import OpponentColorMatcher
from experiments.opponent_color_mannequin.src.opponent_color import bgr_to_opponent_features
from experiments.opponent_color_mannequin.src.weight_map import inner_rectangle


class OpponentColorFeatureTests(unittest.TestCase):
    def test_bgr_conversion_uses_expected_opponent_axes(self) -> None:
        # BGR: blue=30, green=20, red=10
        image = np.array([[[30, 20, 10]]], dtype=np.uint8)
        features = bgr_to_opponent_features(image)

        self.assertAlmostEqual(float(features[0, 0, 0]), -10.0)
        self.assertAlmostEqual(float(features[0, 0, 1]), -15.0)
        self.assertAlmostEqual(float(features[0, 0, 2]), 18.596, places=3)

    def test_inner_rectangle_zeros_template_edges(self) -> None:
        weights = inner_rectangle(10, 10, (0.2, 0.2, 0.2, 0.2))
        self.assertEqual(float(weights[0, 5]), 0.0)
        self.assertEqual(float(weights[5, 0]), 0.0)
        self.assertEqual(float(weights[5, 5]), 1.0)


class MatcherTests(unittest.TestCase):
    def test_finds_exact_template_at_stride_aligned_position(self) -> None:
        rng = np.random.default_rng(42)
        template = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        frame[16:24, 24:32] = template
        matcher = OpponentColorMatcher(template, max_error=0.1, weight_mode="center_falloff")

        result = matcher.match(frame, stride_x=4, stride_y=4)

        self.assertEqual(result.top_left, (24, 16))
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.detected)

    def test_checks_last_position_when_stride_does_not_divide_extent(self) -> None:
        template = np.full((4, 4, 3), 200, dtype=np.uint8)
        frame = np.zeros((10, 11, 3), dtype=np.uint8)
        frame[6:10, 7:11] = template
        matcher = OpponentColorMatcher(template, max_error=0.1, weight_mode="center_falloff")

        result = matcher.match(frame, stride_x=3, stride_y=4)

        self.assertEqual(result.top_left, (7, 6))
        self.assertTrue(result.detected)

    def test_rejects_minimum_when_it_exceeds_threshold(self) -> None:
        template = np.full((4, 4, 3), 255, dtype=np.uint8)
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        matcher = OpponentColorMatcher(template, max_error=1.0, weight_mode="center_falloff")

        result = matcher.match(frame, stride_x=4, stride_y=4)

        self.assertFalse(result.detected)
        self.assertGreater(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
