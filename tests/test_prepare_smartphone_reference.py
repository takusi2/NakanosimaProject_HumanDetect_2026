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

    def test_portrait_image_is_padded_with_black_on_left_and_right(self) -> None:
        portrait = np.full((240, 120, 3), 200, dtype=np.uint8)
        result = preprocessing.fit_to_sensor_frame_with_black_padding(portrait)
        self.assertEqual(result.shape, (120, 160, 3))
        self.assertTrue(np.all(result[:, :50] == 0))
        self.assertTrue(np.all(result[:, 50:110] == 200))
        self.assertTrue(np.all(result[:, 110:] == 0))

    def test_vertical_crop_removes_only_top_and_bottom(self) -> None:
        image = np.arange(10 * 4 * 3, dtype=np.uint8).reshape(10, 4, 3)
        result = preprocessing.crop_vertical_region(
            image, top_pixels=2, bottom_pixels=3
        )
        self.assertEqual(result.shape, (5, 4, 3))
        self.assertTrue(np.array_equal(result, image[2:7, :]))

    def test_saturation_scale_zero_makes_final_image_achromatic(self) -> None:
        image = np.array(
            [[[10, 90, 240], [200, 30, 50]]], dtype=np.uint8
        )
        result = preprocessing.adjust_saturation_bgr(image, saturation_scale=0.0)
        result_hsv = preprocessing.cv2.cvtColor(result, preprocessing.cv2.COLOR_BGR2HSV)
        self.assertTrue(np.all(result_hsv[:, :, 1] == 0))

    def test_saturation_scale_one_keeps_pixels_unchanged(self) -> None:
        result = preprocessing.adjust_saturation_bgr(self.image, saturation_scale=1.0)
        self.assertTrue(np.array_equal(result, self.image))

    def test_center_surround_of_uniform_image_is_black(self) -> None:
        image = np.full((120, 160, 3), 90, dtype=np.uint8)
        _, _, linear_result, log_result = preprocessing.apply_center_surround_bgr(
            image, center_sigma=1.0, surround_sigma=6.0
        )
        self.assertTrue(np.all(linear_result == 0))
        self.assertTrue(np.all(log_result == 0))

    def test_center_surround_uses_logarithms_before_subtraction(self) -> None:
        center_source = self.image
        center, surround, _, log_result = preprocessing.apply_center_surround_bgr(
            center_source, center_sigma=1.0, surround_sigma=6.0
        )
        expected_log = preprocessing.compute_center_surround(
            preprocessing.logarithm_for_center_surround(center[..., 0], 1.0),
            preprocessing.logarithm_for_center_surround(surround[..., 0], 1.0),
        )
        linear_difference = preprocessing.compute_center_surround(
            center[..., 0], surround[..., 0]
        )
        self.assertTrue(np.array_equal(log_result[..., 0], expected_log))
        self.assertFalse(np.array_equal(log_result[..., 0], linear_difference))

    def test_full_pipeline_outputs_sensor_sized_center_surround_frame(self) -> None:
        result = preprocessing.prepare_sensor_like_frame(
            self.image,
            sensor_width=160,
            sensor_height=120,
            center_sigma=1.0,
            surround_sigma=6.0,
            vertical_crop_top_px=20,
            vertical_crop_bottom_px=30,
        )
        self.assertEqual(result["sensor_frame_nr_bgr"].shape, (120, 160, 3))
        self.assertEqual(result["naka_rushton_vertical_crop_bgr"].shape, (130, 240, 3))
        self.assertEqual(result["linear_center_surround_bgr"].shape, (120, 160, 3))
        self.assertEqual(result["log_center_surround_bgr"].shape, (120, 160, 3))
        self.assertEqual(
            result["linear_center_surround_saturation_adjusted_bgr"].shape,
            (120, 160, 3),
        )
        self.assertEqual(
            result["log_center_surround_saturation_adjusted_bgr"].shape,
            (120, 160, 3),
        )
        self.assertEqual(result["linear_center_surround_bgr"].dtype, np.uint8)
        self.assertEqual(result["log_center_surround_bgr"].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
