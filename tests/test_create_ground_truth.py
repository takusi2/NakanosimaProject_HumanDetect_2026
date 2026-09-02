import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.create_ground_truth import (
    clamp_display_scale,
    create_document,
    save_document,
    set_frame_annotation,
)


class CreateGroundTruthTest(unittest.TestCase):
    def test_saves_target_no_target_and_ignore_annotations(self):
        document = create_document(Path("output/example.mp4"), 640, 480, 30.0, 3)
        set_frame_annotation(document, 1, "target", (320, 240))
        set_frame_annotation(document, 2, "no_target")
        set_frame_annotation(document, 3, "ignore")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "point_ground_truth.json"
            save_document(output_path, document)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["annotation_type"], "single_target_point")
        self.assertEqual(saved["frame_index_base"], 1)
        self.assertEqual(saved["frames"]["1"], {"state": "target", "target_point_xy": [320, 240]})
        self.assertEqual(saved["frames"]["2"], {"state": "no_target"})
        self.assertEqual(saved["frames"]["3"], {"state": "ignore"})

    def test_rejects_invalid_annotation_combinations(self):
        document = create_document(Path("output/example.mp4"), 640, 480, 30.0, 1)
        with self.assertRaises(ValueError):
            set_frame_annotation(document, 1, "target")
        with self.assertRaises(ValueError):
            set_frame_annotation(document, 1, "no_target", (1, 1))

    def test_display_scale_can_enlarge_and_is_limited_by_max_width(self):
        self.assertEqual(clamp_display_scale(2.0, 640), 2.0)
        self.assertEqual(clamp_display_scale(10.0, 640), 6.0)
        self.assertEqual(clamp_display_scale(10.0, 1000), 4.0)
        self.assertEqual(clamp_display_scale(0.01, 640), 0.25)


if __name__ == "__main__":
    unittest.main()
