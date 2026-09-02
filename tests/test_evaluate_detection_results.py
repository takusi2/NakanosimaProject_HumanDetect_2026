import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_detection_results import evaluate


def detection(status, bbox, fps=10.0):
    return {
        "frame_index": 1,
        "source_timestamp_s": 0.0,
        "processing_time_s": 0.1,
        "processing_fps": fps,
        "detections": [{"status": status, "bbox_xyxy": bbox}],
    }


class EvaluateDetectionResultsTest(unittest.TestCase):
    def test_calculates_final_stage_and_continuity_metrics(self):
        ground_truth = {
            "annotation_type": "single_target_point",
            "frame_index_base": 1,
            "video": {"width": 100, "height": 100, "frame_count": 5, "fps": 10.0},
            "frames": {
                "1": {"state": "target", "target_point_xy": [5, 5]},
                "2": {"state": "target", "target_point_xy": [5, 5]},
                "3": {"state": "no_target"},
                "4": {"state": "target", "target_point_xy": [5, 5]},
                "5": {"state": "ignore"},
            },
        }
        frame1 = detection("MATCH", [0, 0, 10, 10])
        frame1["detections"].append({"status": "MATCH", "bbox_xyxy": [20, 20, 30, 30]})
        frame2 = detection("person", [0, 0, 10, 10])
        frame3 = detection("MATCH", [20, 20, 30, 30])
        frame5 = detection("MATCH", [20, 20, 30, 30])
        for index, frame in enumerate((frame1, frame2, frame3, frame5), start=1):
            frame["frame_index"] = index if index < 4 else 5
            frame["source_timestamp_s"] = (frame["frame_index"] - 1) / 10.0
        summary, rows = evaluate(
            ground_truth, {1: frame1, 2: frame2, 3: frame3, 5: frame5}
        )

        final_match = summary["final_match"]
        self.assertEqual(final_match["true_positive"], 1)
        self.assertEqual(final_match["false_positive"], 2)
        self.assertEqual(final_match["false_negative"], 2)
        self.assertAlmostEqual(final_match["precision"], 1 / 3)
        self.assertAlmostEqual(final_match["recall"], 1 / 3)
        self.assertAlmostEqual(final_match["f1_score"], 1 / 3)

        stage = summary["target_stage_recall"]
        self.assertEqual(stage["covered_frames"], {
            "yolo_person_bbox": 2,
            "reid_input_not_skip": 2,
            "maybe_or_match": 1,
            "final_match": 1,
        })
        self.assertEqual(summary["continuity"]["longest_miss_streak_frames"], 1)
        self.assertEqual(summary["continuity"]["longest_match_streak_frames"], 1)
        self.assertEqual(len(rows), 5)

    def test_scales_point_to_detection_bbox_coordinate_system(self):
        ground_truth = {
            "annotation_type": "single_target_point",
            "frame_index_base": 1,
            "video": {"width": 640, "height": 480, "frame_count": 1, "fps": 30.0},
            "frames": {"1": {"state": "target", "target_point_xy": [320, 240]}},
        }
        frame = detection("MATCH", [70, 50, 90, 70])
        summary, rows = evaluate(ground_truth, {1: frame}, 160, 120)

        self.assertEqual(summary["final_match"]["true_positive"], 1)
        self.assertEqual(summary["evaluation_scope"]["ground_truth_to_detection_scale"], [0.25, 0.25])
        self.assertEqual(rows[0]["evaluation_target_x"], 80.0)
        self.assertEqual(rows[0]["evaluation_target_y"], 60.0)


if __name__ == "__main__":
    unittest.main()
