import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.detection_result_saver import DetectionResultSaver


class DetectionResultSaverTest(unittest.TestCase):
    def test_saves_frame_data_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            saver = DetectionResultSaver(
                output_root=temporary_directory,
                source_type="video",
                source_path="output/example.mp4",
                config_path="config/config2.yaml",
                config={"model_pt": "yolo11x.pt", "person_color": (255, 0, 0)},
            )
            saver.write_frame(
                frame_index=1,
                source_timestamp_s=0.25,
                processing_time_s=0.05,
                detections=[
                    {
                        "detection_index": 0,
                        "bbox_xyxy": [10, 20, 50, 80],
                        "yolo_confidence": 0.91,
                        "width": 40,
                        "height": 60,
                        "area": 2400,
                        "status": "MATCH",
                        "is_target_frame": True,
                        "is_match": True,
                        "is_number1": True,
                        "similarity": 0.82,
                        "track_id": 3,
                    }
                ],
            )
            run_directory = saver.run_directory
            saver.close()

            jsonl_rows = [
                json.loads(line)
                for line in (run_directory / "detections.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(jsonl_rows[0]["frame_index"], 1)
            self.assertEqual(jsonl_rows[0]["detections"][0]["status"], "MATCH")
            self.assertTrue(jsonl_rows[0]["detections"][0]["is_number1"])

            with (run_directory / "detections.csv").open(encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(csv_rows[0]["track_id"], "3")

            metadata = json.loads(
                (run_directory / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["frames_written"], 1)
            self.assertEqual(metadata["detections_written"], 1)


if __name__ == "__main__":
    unittest.main()
