"""スマホ参照画像のNaka-Rushton / Center-Surround前処理を確認する。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import prepare_smartphone_reference as preprocessing


class SmartphoneReferencePreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[:180, :240]
        self.image = np.dstack((
            ((x * 3 + y) % 256).astype(np.uint8),
            ((x + y * 2) % 256).astype(np.uint8),
            ((x * 2 + y * 3) % 256).astype(np.uint8),
        ))

    def test_naka_rushton_is_uint8_and_has_valid_range(self) -> None:
        result = preprocessing.apply_naka_rushton_bgr(self.image)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(int(result.max()), int(result.min()))

    def test_uniform_image_is_handled_without_non_finite_values(self) -> None:
        image = np.full((120, 160, 3), 100, dtype=np.uint8)
        result = preprocessing.apply_naka_rushton_bgr(image)
        self.assertTrue(np.all(result == 0))

    def test_aspect_expansion_keeps_selection_inside_four_by_three_frame(self) -> None:
        roi = preprocessing.expand_roi_to_aspect((80, 30, 40, 100), 240, 180, 4 / 3)
        x, y, width, height = roi
        self.assertLessEqual(x, 80)
        self.assertLessEqual(y, 30)
        self.assertGreaterEqual(x + width, 120)
        self.assertGreaterEqual(y + height, 130)
        # 整数画素のため完全な4:3にはならない場合があるが、160x120へ
        # リサイズする前提では十分に近い比率であることを確認する。
        self.assertAlmostEqual(width / height, 4 / 3, delta=0.01)

    def test_center_surround_of_uniform_image_is_black(self) -> None:
        image = np.full((120, 160, 3), 90, dtype=np.uint8)
        _, _, result = preprocessing.apply_center_surround_bgr(
            image, center_sigma=1.0, surround_sigma=6.0
        )
        self.assertTrue(np.all(result == 0))

    def test_full_pipeline_outputs_sensor_sized_center_surround_frame(self) -> None:
        result = preprocessing.prepare_sensor_like_frame(
            self.image,
            (70, 25, 80, 120),
            sensor_width=160,
            sensor_height=120,
            center_sigma=1.0,
            surround_sigma=6.0,
        )
        self.assertEqual(result["sensor_frame_nr_bgr"].shape, (120, 160, 3))
        self.assertEqual(result["center_surround_bgr"].shape, (120, 160, 3))
        self.assertEqual(result["center_surround_bgr"].dtype, np.uint8)
        self.assertEqual(result["pseudo_sensor_roi_xywh"][2:], (160, 120))


if __name__ == "__main__":
    unittest.main(verbosity=2)
