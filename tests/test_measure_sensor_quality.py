"""代表フレーム方式の色評価を、既知色のテスト画像・動画で検証する。"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


# このファイルを VS Code の「Pythonファイルを実行」などで直接起動しても、
# プロジェクト直下の tools を絶対インポートできるようにする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import measure_sensor_quality as quality


DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
REPRESENTATIVE_IMAGE_PATH = DATA_DIRECTORY / "representative_color_chart.png"
# OpenCVのVideoCaptureには、プロジェクト直下からの相対ASCIIパスを渡す。
TEST_VIDEO_PATH = Path("tests/data/representative_then_changed.avi")

# 各60x60タイルの中央40x40だけをROIにする。パッチ境界を含めないため、
# 画素値が既知の一様な色領域だけを評価できる。
PATCH_ROIS = {
    "white": [10, 10, 40, 40],
    "gray": [70, 10, 40, 40],
    "black": [130, 10, 40, 40],
    "red": [10, 70, 40, 40],
    "green": [70, 70, 40, 40],
    "blue": [130, 70, 40, 40],
}


class SensorQualityRepresentativeFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # VideoCaptureにはUnicodeを含まない相対パスを渡すため、作業場所を固定する。
        os.chdir(PROJECT_ROOT)

    def setUp(self) -> None:
        if not REPRESENTATIVE_IMAGE_PATH.exists() or not TEST_VIDEO_PATH.exists():
            self.fail(
                "テストデータがありません。"
                "python tests/create_sensor_quality_test_assets.py を実行してください。"
            )
        self.original_representative_frame_index = quality.REPRESENTATIVE_FRAME_INDEX

    def tearDown(self) -> None:
        quality.REPRESENTATIVE_FRAME_INDEX = self.original_representative_frame_index

    @staticmethod
    def read_test_image(path: Path):
        """Unicodeを含むパスでも画像を読み込めるようにする。"""
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def test_known_color_patch_has_expected_opponent_values(self) -> None:
        image = self.read_test_image(REPRESENTATIVE_IMAGE_PATH)
        self.assertIsNotNone(image)
        assert image is not None

        red = quality.opponent_polar_statistics(
            quality.opponent_polar_sums(quality.crop(image, PATCH_ROIS["red"]))
        )
        expected_red_theta = math.degrees(math.atan2(127.5, 255.0))
        self.assertAlmostEqual(red["opponent_rg_mean"], 255.0, places=4)
        self.assertAlmostEqual(red["opponent_yb_mean"], 127.5, places=4)
        self.assertAlmostEqual(
            red["radius_mean"], math.hypot(255.0, 127.5), places=4
        )
        self.assertAlmostEqual(
            red["theta_deg_circular_mean"], expected_red_theta, places=4
        )
        self.assertAlmostEqual(red["theta_resultant_length"], 1.0, places=6)
        self.assertEqual(red["pixels_averaged"], 1600)

        black = quality.opponent_polar_statistics(
            quality.opponent_polar_sums(quality.crop(image, PATCH_ROIS["black"]))
        )
        self.assertAlmostEqual(black["radius_mean"], 0.0, places=6)
        self.assertIsNone(black["theta_deg_circular_mean"])
        self.assertEqual(black["theta_pixels_averaged"], 0)

    def test_roi_selection_accepts_only_a_top_left_to_bottom_right_drag(self) -> None:
        self.assertEqual(
            quality.roi_from_top_left_drag((10, 20), (50, 80)), [10, 20, 40, 60]
        )
        self.assertIsNone(quality.roi_from_top_left_drag((50, 20), (10, 80)))
        self.assertIsNone(quality.roi_from_top_left_drag((10, 80), (50, 20)))
        self.assertIsNone(quality.roi_from_top_left_drag((10, 20), (10, 80)))

    def test_reselect_mode_can_load_previously_saved_rois(self) -> None:
        existing_roi_data = quality.new_roi_data()
        existing_roi_data["videos"]["sample-video"] = {
            "representative_frame_index": 0,
            "patch_rois_xywh": PATCH_ROIS,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            roi_path = Path(temporary_directory) / "rois.json"
            quality.atomic_json_dump(roi_path, existing_roi_data)
            loaded_roi_data = quality.load_roi_data(roi_path)
        self.assertEqual(loaded_roi_data, existing_roi_data)

    def test_reference_group_ignores_site_by_default(self) -> None:
        shade = quality.parse_filename_conditions(
            Path(
                "20260825_1509_smkArea_nowide_gC01_gS06_iir040_nrA_"
                "shade_none_d1p0m_front_t01.mp4"
            )
        )
        sun = quality.parse_filename_conditions(
            Path(
                "20260825_1510_2fparking_nowide_gC01_gS06_iir040_nrA_"
                "sun_back_d1p0m_front_t01.mp4"
            )
        )
        self.assertEqual(shade["light_condition"], "shade-none")
        self.assertEqual(sun["light_condition"], "sun-back")
        self.assertEqual(shade["comparison_group"], sun["comparison_group"])

    def test_video_evaluation_uses_only_the_selected_representative_frame(self) -> None:
        quality.REPRESENTATIVE_FRAME_INDEX = 0
        result = quality.analyze_video(
            TEST_VIDEO_PATH, {"patch_rois_xywh": PATCH_ROIS}
        )
        red_polar = result["patches"]["red"]["opponent_polar"]

        self.assertEqual(result["evaluation_mode"], "single_representative_frame")
        self.assertEqual(result["representative_frame_index"], 0)
        self.assertEqual(result["frames_analyzed"], 1)
        # MJPG圧縮後でも、先頭フレームの赤はおよそ26.6度のままである。
        self.assertAlmostEqual(red_polar["theta_deg_circular_mean"], 26.565, delta=3.0)

    def test_changing_representative_frame_changes_the_result(self) -> None:
        quality.REPRESENTATIVE_FRAME_INDEX = 1
        result = quality.analyze_video(
            TEST_VIDEO_PATH, {"patch_rois_xywh": PATCH_ROIS}
        )
        red_polar = result["patches"]["red"]["opponent_polar"]

        self.assertEqual(result["representative_frame_index"], 1)
        # 2フレーム目ではred ROIを意図的に青へ置き換えている。
        self.assertAlmostEqual(red_polar["theta_deg_circular_mean"], 270.0, delta=3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
