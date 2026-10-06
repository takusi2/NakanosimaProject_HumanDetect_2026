"""設定クラスがYAMLの値を読み、初期化後に書き換えられないことを確認する。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

from src.pipeline import CoarseToFineMatcher
from src.settings import MatcherSettings, RunSettings, TrackingSettings
from src.tracking import TemporalMatchTracker


class RunSettingsTests(unittest.TestCase):
    def test_reads_each_class_setting_from_yaml(self) -> None:
        yaml_text = """
template_path: ../assets/template.jpg
input_source: video
video_path: ../assets/input.mp4
results_root: ../results
device: cpu
frame_downscale: 0.5
coarse_template_scales: [0.3, 0.5]
detail_template_scales: [1.0, 0.6]
target_confirm_window_frames: 8
target_confirm_frames: 4
match_iou_thresh: 0.7
input_width: 160
input_height: 120
processing_frame_limit: 12
display_width: 640
display_height: 480
save:
  enabled: false
performance:
  enabled: false
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            settings = RunSettings(config_path)

        self.assertEqual(settings.matcher.device, "cpu")
        self.assertEqual(settings.matcher.coarse_template_scales, (0.3, 0.5))
        self.assertEqual(settings.tracking.window_frames, 8)
        self.assertEqual(settings.input.input_size, (160, 120))
        self.assertEqual(settings.input.processing_frame_limit, 12)
        self.assertEqual(settings.display.display_size, (640, 480))
        self.assertEqual(settings.input.template_path, config_path.parent.parent / "assets/template.jpg")
        self.assertFalse(settings.save.enabled)
        self.assertFalse(settings.performance.enabled)

    def test_settings_cannot_be_changed_after_initialisation(self) -> None:
        matcher_settings = MatcherSettings({"device": "cpu"})

        with self.assertRaises(AttributeError):
            matcher_settings.device = "cuda"

    def test_matcher_and_tracker_accept_settings_objects(self) -> None:
        template = np.full((10, 8, 3), 128, dtype=np.uint8)
        matcher = CoarseToFineMatcher(
            template,
            MatcherSettings(
                {
                    "device": "cpu",
                    "coarse_template_scales": [1.0],
                    "detail_template_scales": [1.0],
                }
            ),
        )
        tracker = TemporalMatchTracker(
            TrackingSettings(
                {
                    "target_confirm_window_frames": 3,
                    "target_confirm_frames": 2,
                    "match_iou_thresh": 0.7,
                }
            )
        )

        self.assertEqual(str(matcher.device), "cpu")
        self.assertEqual(len(matcher.coarse_templates), 1)
        self.assertEqual(tracker.window_frames, 3)
        self.assertEqual(tracker.confirm_frames, 2)


if __name__ == "__main__":
    unittest.main()
